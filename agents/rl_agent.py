import os
from typing import List, Dict

import numpy as np

from agents.predictors import NNPredictor
from agents.predictors import RuleBasedPredictor
from agents.featurizers import OriginalFeaturizer, FullFeaturizer
from game_engine.player import Player, AverageRandomPlayer
from game_engine.card import Card

ACTION_DIMENSIONS = 4 * 13 + 2
MODELS_PATH = 'models/'
PLAYER = 4


class RLAgent(AverageRandomPlayer):
    """Abstract RL Agent. Should be extended specific for each library.

    Attributes:
        name (str): Used for things like determining the folder
            where the model is saved. Defaults to class name.
        predictor (NNPredictor / RuleBasedPredictor): A predictor specific to that agent.
            Doesn't share parameters with any other predictor. Can either be a NNPredictor or RuleBasedPredictor
        keep_models_fixed: If set to true, neither the predictor
            nor the extending agent is trained, so only inference is done.
        featurizer (OriginalFeaturizer): Used for getting the state
            from the arguments to play_card
        not_yet_given_reward (float | None):
            We want data in the following form (for use in libraries):
            self.act(state) -> self.observe(reward, terminal) -> self.act ...
            Because we don't know the reward or whether the game has ended
            directly after we play a card, we give this reward + terminal info
            before each card (starting from the second one) or when the game
            has finished so that there is a one-to-one association
            from act to observe.
            Therefore we have to keep track of whether there is a reward
            which we should give before we are asked for the next card
            which is what this variable is for
    """

    def __init__(self, name=None, predictor=None, keep_models_fixed=False, featurizer=None):
        super().__init__()

        # determine type of predictor. Is either `NN` (Neural Network) or `RuleBased`
        # Only case where it is RuleBased is, when it is passed via the constructor, default is `NN`
        if isinstance(predictor, RuleBasedPredictor):
            self.predictor_type = 'RuleBased'
        else:
            self.predictor_type = 'NN'

        # name of agent, also determines the path where models for agent and predictor are saved
        if name is not None:
            self.name = name
        else:
            self.name = '{}_{}Predictor'.format(self.__class__.__name__, self.predictor_type)

        # initialize predictor
        self.predictor_model_path = os.path.join(MODELS_PATH, self.name, 'Predictor/')
        print('Modelpath', os.path.abspath(self.predictor_model_path))
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = NNPredictor(model_path=self.predictor_model_path,
                                         keep_models_fixed=keep_models_fixed)

        self.keep_models_fixed = keep_models_fixed

        if featurizer is None:
            featurizer = FullFeaturizer()
        self.featurizer = featurizer
        self.not_yet_given_reward = None

        # keep track of which colors other players don't have based on if they follow the suit
        self.color_left_indicator = np.zeros((PLAYER - 1, 4))
        self.trick_index = 0

        self.last_10000_cards_played_valid = []
        self.valid_rate = 0

    def save_models(self, path=None):
        """Saves model for predictor. Model for rl agent needs to be saved in agent class e.g. TFAgentsPPOAgent.
        Only save models if predictor is a NN predictor and keep models are not fixed"""
        if self.keep_models_fixed or self.predictor_type != 'NN':
            return

        if path is None:
            path = self.predictor_model_path
        else:
            path = os.path.join(path, 'Predictor/')

        if not os.path.exists(path):
            os.makedirs(path)
        self.predictor.save_model(path)

    def act(self, state: np.ndarray, valid_action_mask: np.ndarray) -> Card:
        """Returns the action the model takes if it is in the given state

        Args:
            state: The observation (output from featurizer)
            valid_action_mask: for each element in the array, a 1 if
                we can play this card (take this action) and a 0 if we can't

        Returns: The card the model will play (removed from hand)
        """
        raise NotImplementedError  # Has to be implemented by child

    def observe(self, reward: float, terminal: bool):
        """Feeds the reward to the model & resets stuff if terminal is True"""
        raise NotImplementedError  # Has to be implemented by child

    def _valid_action_mask(self, first):
        playable_cards = self.get_playable_cards(first)
        playable_cards = [int(card) for card in playable_cards]
        action_mask = np.full(ACTION_DIMENSIONS, -np.inf, dtype=np.float32)
        action_mask[playable_cards] = 0
        return action_mask

    def play_card(self, trump: Card, first: Card, played: Dict[int, Card], players: List[Player],
                  played_in_round: Dict[int, List[Card]], first_player_index: int):
        """Plays a card based on the agents action"""

        # TODO replace this print with the corresponding plotting once
        # implemented to keep track of how many valid cards the agent plays
        if len(self.last_10000_cards_played_valid) == 10000:
            self.valid_rate = np.mean(self.last_10000_cards_played_valid)
            self.last_10000_cards_played_valid = []

        if self.not_yet_given_reward is not None:
            self.observe(reward=self.not_yet_given_reward, terminal=False)
            self.not_yet_given_reward = None

        # keep track of which colors the other players have
        self._update_color_information(first, players, played_in_round, first_player_index)

        state = self.featurizer.transform(self, trump, first, played, players, played_in_round,
                                          self.color_left_indicator, first_player_index)

        action = self.act(state, self._valid_action_mask(first))

        # find the card which corresponds to that action and return it if valid
        playable_cards = self.get_playable_cards(first)
        for card in self.hand:
            if int(card) == action:
                if card not in playable_cards:
                    break
                self.hand.remove(card)
                self.not_yet_given_reward = 0
                self.last_10000_cards_played_valid.append(1)
                return card

        # the agent is trying to play a card which is invalid
        # we give him a negative reward, play a random card and continue
        self.last_10000_cards_played_valid.append(0)
        self.not_yet_given_reward = -10
        return super().play_card(trump, first, played, players, played_in_round, first_player_index)

    def get_prediction(self, trump: Card, num_players: int):
        """Return the round prediction using the build in NN Predictor

        Args:
            trump (Card):
            num_players (int):

        Returns:
            int: prediction
        """
        self.prediction_x, prediction = self.predictor.make_prediction(self.hand, trump)

        return prediction

    def announce_result(self, num_tricks_achieved: int, reward: float):
        """use the game result as a feedback for the agent and predictor"""
        super().announce_result(num_tricks_achieved, reward)
        reward_to_give = self.not_yet_given_reward + reward
        self.observe(reward=reward_to_give, terminal=True)
        self.not_yet_given_reward = None
        self.predictor.add_game_result(self.prediction_x, num_tricks_achieved)

    def _update_color_information(self, first, players, played_in_round, first_player_index):
        """Updates self.color_left_indicator based on if the other players followed the suit during the last round"""

        # if a player hasn't played cards during the round, it's the first trick in the round
        if min(len(cards) for cards in played_in_round.values()) == 0:
            self.color_left_indicator = np.zeros((PLAYER - 1, 4))
            self.trick_index = 0

        # we only infer the color of other players from last rounds
        if self.trick_index > 0:
            # The first card that was played during the last trick
            last_first = played_in_round[(first_player_index - 1) % len(players)][self.trick_index - 1]
            assert last_first is not None

            # when a jester is played, the next card determines the suit to follow
            player_index = first_player_index
            while played_in_round[player_index][self.trick_index - 1] == Card("White", 0):
                player_index = (player_index + 1) % len(players)
                last_first = played_in_round[player_index][self.trick_index - 1]

                # in case everybody plays a jester
                if player_index == first_player_index:
                    break

            color_indicator_index = 0
            for index, player in enumerate(players):
                if not player == self:
                    if not self._follows_suit(last_first, played_in_round[index][self.trick_index - 1]):
                        self.color_left_indicator[color_indicator_index][Card.colors.index(last_first.color) - 1] = 1
                    color_indicator_index += 1

        self.trick_index += 1

    @staticmethod
    def _follows_suit(first, card):
        return card.color == 'White' or first.color == 'White' or first.color == card.color
