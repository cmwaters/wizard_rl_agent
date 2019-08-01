"""Adaption of DQN RL Agent from Github Repository (https://github.com/mvelax/wizard-python). Is used for evaluating
our agent """

import numpy as np

from agents import featurizers
from agents.original import estimators, policies
from agents.predictors import NNPredictor
from agents.predictors import RuleBasedPredictor
from game_engine import player
import os

MODELS_PATH = 'models/'


class OriginalRLAgent(player.AverageRandomPlayer):
    """A computer player that learns using reinforcement learning."""

    def __init__(self, estimator=None, policy=None, featurizer=None, predictor=None, name=None,
                 keep_models_fixed=False):
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

        # initialize predictor. the default predictor is the Neural Network Predictor
        self.predictor_model_path = os.path.join(MODELS_PATH, self.name, 'Predictor/')
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = NNPredictor(model_path=self.predictor_model_path,
                                         keep_models_fixed=keep_models_fixed)

        self.keep_models_fixed = keep_models_fixed

        if estimator is None:
            self.estimator = estimators.DQNEstimator()
        else:
            self.estimator = estimator

        if policy is None:
            if keep_models_fixed:
                self.policy = policies.EGreedyPolicy(self.estimator, epsilon=0)
            else:
                self.policy = policies.EGreedyPolicy(self.estimator, epsilon=0.1)
        else:
            self.policy = policy

        if featurizer is None:
            self.featurizer = featurizers.OriginalFeaturizer()
        else:
            # remove when support for different featurizers
            if type(featurizer) is not featurizers.OriginalFeaturizer:
                raise Exception("No other featurizer than OriginalFeaturizer currently supported by OriginalRLAgent")
            self.featurizer = featurizer

        self.old_state = None
        self.old_score = 0
        self.old_action = None

        self.clone_counter = 0

        # load agent
        self.agent_path = os.path.join(MODELS_PATH, self.name, 'Agent')
        if os.path.exists(self.agent_path):
            self.load_estimator(os.path.join(self.agent_path, 'model'))

    def play_card(self, trump, first, played, players, played_in_round, first_player_index):
        """Plays a card according to the estimator Q function and learns
        on-line.
        Relies on scores being updated by the environment to calculate reward.
        Args:
            trump: (Card) trump card.
            first: (Card) first card.
            played: (list(Card)) list of cards played in Trick, may be empty.
            players: (list(Player)) list of players in the game, including this
            player.
            played_in_round: (list(Card)) list of cards played so far in the
            game, may be empty.
            first_player_index: Index of the first player in the trick

        Returns:
            card_to_play: (Card) the card object that the player
             decided to play.
        """
        state = self.featurizer.transform(self, trump, first, played, players, played_in_round,
                                          None, first_player_index)
        terminal = False
        if self.old_state is not None and self.old_action is not None:
            r = self.reward
            if r != 0:
                terminal = True
                # If we got a reward, it's a terminal state.
                # We signal this with an s_prime == None

                if not self.keep_models_fixed:
                    self.estimator.update(self.old_state, self.old_action, r, None)
            else:
                if not self.keep_models_fixed:
                    self.estimator.update(self.old_state, self.old_action, r, state)

        probs = self.policy.get_probabilities(state)
        a = np.random.choice(len(probs), p=probs)
        card_to_play = self._remove_card_played(a)
        self.old_state = None if terminal else state
        self.old_action = a
        self.reward = 0  # After playing a card, the reward is 0.
        # Unless it's the last card of the game, then the Round object will
        # call give_reward before the next play_card, setting the correct reward
        return card_to_play

    def save_estimator(self, name="default"):
        self.estimator.save(name)

    def load_estimator(self, name="default"):
        self.estimator.load(name)

    def save_models(self):
        if self.keep_models_fixed:
            return

        # save predictor model
        if self.predictor_type == 'NN':
            if not os.path.exists(self.predictor_model_path):
                os.makedirs(self.predictor_model_path)
            self.predictor.save_model(self.predictor_model_path)

        # save agent model
        if not os.path.exists(self.agent_path):
            os.makedirs(self.agent_path)
        self.save_estimator(os.path.join(self.agent_path, 'model'))

    def _remove_card_played(self, a):
        """
        Given an action (integer) remove a card equivalent to it from the
        player's hand and return it.

        Args:
            a: (int) The action taken. Remove a card with the same code.
            If there is more than one that matches, it does not matter which,
            but just remove one.

        Returns:
            card_to_play: The card corresponding to the action.

        Raises:
            RuntimeError when the action does not correspond to any card.

        """
        assert isinstance(a, int), "action played is not an int as expected"
        card_to_return = None
        for card in self.hand:
            if int(card) == a:
                card_to_return = card
                self.hand.remove(card)
                break
        if card_to_return is None:
            raise RuntimeError("Computer did not find a valid card for this"
                               "action.\nHand: {}\nAction: {}".format(self.hand,
                                                                      a))
        return card_to_return

    def get_prediction(self, trump, num_players):
        self.prediction_x, prediction = \
            self.predictor.make_prediction(self.hand, trump)
        return prediction

    def announce_result(self, num_tricks_achieved, reward):
        super().announce_result(num_tricks_achieved, reward)
        self.predictor.add_game_result(self.prediction_x, num_tricks_achieved)

    def clone(self, name=None):
        """Return a clone of this agent with networks & predictor shared"""

        if name is None:
            self.clone_counter += 1
            name = self.name + 'Clone' + str(self.clone_counter)
        return OriginalRLAgent(name=name, estimator=self.estimator, featurizer=self.featurizer)
