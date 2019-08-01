import os
from typing import List, Tuple

import numpy as np
import tensorflow.keras as K

from agents.featurizers import FullFeaturizer
from game_engine.card import Card

logdir = '/logs'


class PredictorNetwork(K.models.Model):
    def __init__(self, x_dim, y_dim):
        super().__init__(name='predictor_network', dynamic=False)

        self.model_layers = [
            K.layers.Dense(128, input_dim=x_dim, activation='relu'),
            K.layers.BatchNormalization(),
            K.layers.Dense(64, activation='relu'),
            K.layers.BatchNormalization(),
            K.layers.Dense(32, activation='relu'),
            K.layers.Dense(y_dim, activation='softmax'),
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x


class NNPredictor:
    """Neural Network Predictor, predicts the number of tricks achieved in a round.

    Attributes:
        x_dim (int): The input shape of the NN. Equal to
            - 4 * 13 = 52 for numbered color cards
            - 2 for wizards & jesters
            - 5 for trump colors (4 colors + no trump)
            - 1 for the prediction
        max_num_tricks (int): Determines the output shape of the NN and
            therefore restricts the possible number of tricks
            which can be predicted
        y_dim (int): The output shape of the NN.
            Equal to max_num_tricks + 1 (as 0 tricks can be predicted)
        x_batch (np.array[train_batch_size][x_dim]): Because we don't want to
            train the NN after each round, we store the data
            in these batch arrays and train it after train_batch_size rounds
        y_batch (np.array[train_batch_size][y_dim]):
            Same as x_batch for the labels
        batch_position (int): Determines our current position
            in x_batch, y_batch. Resets to 0 after train_batch_size rounds.
        model_path (str): The path to the file where the parameters etc.
            of the NN are stored
        model (keras.models.Model): The NN
        train_step (int): How many samples should be recorded before a training step is executed.
        verbose (bool): Determines if information about the prediction performance should be printed
        keep_models_fixed: If set to true, the NN is not trained
    """

    def __init__(self, model_path='prediction_model', max_num_tricks=15,
                 train_batch_size=1000, train_step=300,
                 verbose=True, keep_models_fixed=False):
        self.max_num_tricks = max_num_tricks

        self.y_dim = self.max_num_tricks + 1
        self.x_dim = 59 + max_num_tricks + 1

        self._build_prediction_to_expected_num_points_matrix()

        self.train_step = train_step
        self.buffer_filled = False

        self.x_batch = np.zeros((train_batch_size, self.x_dim))
        self.y_batch = np.zeros((train_batch_size, self.y_dim))
        self.batch_position = 0
        self.train_batch_size = train_batch_size
        self.verbose = verbose
        self.keep_models_fixed = keep_models_fixed

        # keep track of current loss and acc of predictor for tensorboard plotting
        self.current_loss = None
        self.current_acc = None

        # keep track of current round, is needed for reporting purposes
        self.current_round = 0

        self._build_new_model()
        self.model_path = model_path + 'model'
        if os.path.exists(model_path):
            self.model.load_weights(self.model_path)

        # stores the predictions made by the predictor (statistics)
        # 0 stores all the predictions, the other keys correspond to the number of cards
        self.predictions = {"overall": {i: [] for i in range(0, 16)},
                            "correct_prediction": {i: [] for i in range(0, 16)},
                            "incorrect_prediction": {i: [] for i in range(0, 16)}}

        # stores the absolute difference to the predictions (statistics)
        self.prediction_differences = []

    def _build_prediction_to_expected_num_points_matrix(self):
        # We can describe the calculation from the output of the NN
        # (array of probabilities) to the array of expected points as a
        # matrix-vector-multiplication where the matrix describes for each
        # possible prediction we could make and each game outcome the
        # points we would get in this case. This function computes this matrix
        # once and stores it in self.prediction_to_points

        self.prediction_to_points = np.zeros((self.y_dim, self.y_dim))
        for actual_num_tricks in range(self.y_dim):
            for predicted_num_tricks in range(self.y_dim):
                difference = np.abs(predicted_num_tricks - actual_num_tricks)
                if difference == 0:
                    num_points = 20 + predicted_num_tricks * 10
                else:
                    num_points = -10 * difference
                self.prediction_to_points[actual_num_tricks] \
                    [predicted_num_tricks] = num_points

    def _build_new_model(self):
        self.model = PredictorNetwork(self.x_dim, self.y_dim)
        self.model.compile(optimizer=K.optimizers.Adam(),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self, model_path=None):
        if model_path is None:
            self.model.save_weights(self.model_path)
        else:
            self.model.save_weights(model_path + 'model')

    def make_prediction(self, initial_cards: List[Card],
                        trump_color_card: Card) -> Tuple[np.ndarray, int]:
        """Predict the number of tricks based on initial cards + trump color.

        Args:
            initial_cards: The current hand of the agent
            trump_color_card: A card which has the trump color

        Returns: A tuple consisting of
            - The input used for the NN. Should be passed to
              add_game_result once the result is available
            - The predicted number of tricks based on
              whichever has the highest expected reward
        """
        self.current_round = len(initial_cards)

        x = np.concatenate((FullFeaturizer.cards_to_arr(initial_cards),
                            FullFeaturizer.color_to_bin_arr(trump_color_card)))

        X = np.tile(x, (self.y_dim, 1))

        trick_values = K.utils.to_categorical(np.arange(self.y_dim), num_classes=self.y_dim)

        X = np.hstack([X, trick_values])

        probability_distributions = self.model.predict(X)

        # dot product between same rows of both matrices
        expected_value = (self.prediction_to_points * probability_distributions).sum(axis=1)

        prediction = int(np.argmax(expected_value))

        prediction_encoded = K.utils.to_categorical(prediction, num_classes=self.y_dim)
        x = np.append(x, prediction_encoded)

        return x, prediction

    def add_game_result(self, x: np.ndarray, num_tricks_achieved: int):
        """Adds the corresponding label to the cards & trump color in x.

        Also trains the NN if train_batch_size rounds have passed
        since the last training.

        Args:
            x: The result from make_prediction which has been called
                when the game started.
            num_tricks_achieved: The number of tricks achieved
                after the round which corresponds to the one
                passed to make_prediction before. Used as a label.
        """
        y = K.utils.to_categorical(num_tricks_achieved, num_classes=self.y_dim)

        prediction_encoded = x[-self.y_dim:]
        prediction = np.argmax(prediction_encoded)

        # store prediction values for reporting
        if prediction == num_tricks_achieved:
            self.predictions["correct_prediction"][0].append(prediction)
            self.predictions["correct_prediction"][self.current_round].append(prediction)
        else:
            self.predictions["incorrect_prediction"][0].append(prediction)
            self.predictions["incorrect_prediction"][self.current_round].append(prediction)

        self.predictions["overall"][0].append(prediction)
        self.predictions["overall"][self.current_round].append(prediction)
        self.prediction_differences.append(abs(prediction - num_tricks_achieved))

        self.x_batch[self.batch_position] = x
        self.y_batch[self.batch_position] = y
        self.batch_position += 1

        # Train when train_step samples were reached
        if self.buffer_filled and self.batch_position % self.train_step == 0:
            if not self.keep_models_fixed:
                history = self.model.fit(self.x_batch, self.y_batch)
                # update predictors values of loss and acc --> used for tensorforce reporting
                self.current_acc = history.history['accuracy'][0]
                self.current_loss = history.history['loss'][0]
            else:
                self.current_loss, self.current_acc = \
                    self.model.evaluate(self.x_batch, self.y_batch)

        if self.batch_position == self.train_batch_size - 1:
            self.buffer_filled = True
            self.batch_position = 0

            # if self.verbose:
            #     print("Mean Prediction: ", np.mean(self.predictions[0]))
            #     print("Std Prediction: ", np.std(self.predictions[0]))
            #     print("Abs Prediction difference: ", np.mean(self.prediction_differences))
            # # self.predictions = {i: [] for i in range(0, 16)}
            # self.prediction_differences = []


class RuleBasedPredictor:
    """Predictor that uses rule based approach to predict amount of tricks"""
    def __init__(self, aggression=0.0):
        """
        Args:
            aggression (float): aggression is a measure of how high the agent naturally tries to predict. High aggression is good
                                with weak opponents and low aggression for quality opponents. Takes values between -1 and 1.
        """
        self.aggression = RuleBasedPredictor.bound(aggression, 1, -1)

        # keep track of current round, is needed for reporting purposes
        self.current_round = 0

        # stores the predictions made by the predictor (statistics)
        # 0 stores all the predictions, the other keys correspond to the number of cards
        self.predictions = {"overall": {i: [] for i in range(0, 16)},
                            "correct_prediction": {i: [] for i in range(0, 16)},
                            "incorrect_prediction": {i: [] for i in range(0, 16)}}

        # stores the absolute difference to the predictions (statistics)
        self.prediction_differences = []

    def make_prediction(self, initial_cards, trump_color_card):
        """predicts the amount of tricks that the player should win. It does this assigning
        an expected return for each card and sums all the expected returns together. It also takes into
        consideration the aggression of the agent.

        Args:
            initial_cards (list[Card]: The current hand of the agent
            trump_color_card (Card): A card which has the trump color

        Returns:
            int: The predicted number of tricks
        """
        self.current_round = len(initial_cards)

        prediction = 0
        for card in initial_cards:
            if card.value == 14:
                prediction += 0.95 + (0.05 * self.aggression)
            elif card.color == trump_color_card.color:
                prediction += (card.value * (0.050 + (0.005 * self.aggression))) + 0.3
            else:
                prediction += (card.value * (0.030 + (0.005 * self.aggression)))
        prediction = round(RuleBasedPredictor.bound(prediction, len(initial_cards), 0), 0)

        # the reason why a tuple is returned is to keep consistency with the NN Predictor ( defined above ). There
        # the first entry of the tuple is an np.array. here it is just an int

        return prediction, prediction

    def add_game_result(self, prediction, num_tricks_achieved):
        """Adds game result for plotting purpusoes

        Note:
            The add_game_result of the NN Predictor takes in an ndarray as first argument. Here it is just an int

        Args:
            prediction (int): predicted number of tricks
            num_tricks_achieved: The number of tricks achieved
        """
        # store prediction values for reporting
        if prediction == num_tricks_achieved:
            self.predictions["correct_prediction"][0].append(prediction)
            self.predictions["correct_prediction"][self.current_round].append(prediction)
        else:
            self.predictions["incorrect_prediction"][0].append(prediction)
            self.predictions["incorrect_prediction"][self.current_round].append(prediction)

        self.predictions["overall"][0].append(prediction)
        self.predictions["overall"][self.current_round].append(prediction)
        self.prediction_differences.append(abs(prediction - num_tricks_achieved))

    @staticmethod
    def bound(value, max, min):
        """
        Bounds the value between a given range.
        :param value:
        :param max:
        :param min:
        :return: a value between the maximum and minimum
        """
        if value > max:
            return max
        elif value < min:
            return min
        else:
            return value