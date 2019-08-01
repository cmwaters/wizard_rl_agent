import os
import random
import collections
import numpy as np

from game_engine import player
from agents.predictors import NNPredictor

MODELS_PATH = 'models/'


class RuleBasedAgent(player.Player):
    """A computer player that makes decision on predefined rules.
    Aims to resemble the performance and behaviour of that of a human.
    Agent functions by calculating the win desirability of the trick and the
    win probability given the cards. It then makes a decision of what card based
    on the probabilities.

    Attributes:
        name (str): name of agent
        aggresion (float): see comment below
        use_predictor (bool): if True uses neural net as predictor, else use rule base predictor
        keep_models_fixed (bool): if True, NN is not trained if agent uses it for predictions

    """

    # Use this to print out a trick by trick breakdown of the state of the trick and
    # the action and inferences of the rule-based agent
    DEBUG = False

    def __init__(self, name=None, aggression=0.0, use_predictor=False, keep_models_fixed=False):
        super().__init__()
        self.round = 1
        self.num_players = 4
        self.error = 0
        if name is not None:
            self.name = name
        elif use_predictor:
            self.name = self.__class__.__name__ + 'Predictor'
        else:
            self.name = self.__class__.__name__

        # aggression is a measure of how high the agent naturally tries to predict. High aggression is good
        # with weak opponents and low aggression for quality opponents. Takes values between -1 and 1.
        self.aggression = RuleBasedAgent.bound(aggression, 1, -1)

        self.use_predictor = use_predictor
        self.keep_models_fixed = keep_models_fixed

        if self.use_predictor:
            self.predictor_model_path = os.path.join(MODELS_PATH, self.name, 'Predictor/')
            self.predictor = NNPredictor(model_path=self.predictor_model_path, keep_models_fixed=self.keep_models_fixed)

    def save_models(self):
        if self.keep_models_fixed:
            return
        if not os.path.exists(self.predictor_model_path):
            os.makedirs(self.predictor_model_path)
        self.predictor.save_model()

    def get_prediction(self, trump, num_players):
        """This algorithm predicts the amount of tricks that the player should win. It does this assigning
        an expected return for each card and sums all the expected returns together. It also takes into
        consideration the aggression of the agent.

        Args:
            trump (Card): trump card
            num_players (int): amount of players

        Returns:
            int: prediction
        """

        if self.use_predictor:
            # Neural Network Prediction
            self.prediction_x, prediction = self.predictor.make_prediction(self.hand, trump)
            return prediction
        else:
            # rule based prediction
            self.num_players = num_players
            self.played = []
            self.error += (self.prediction - self.wins)
            prediction = 0
            for card in self.hand:
                if card.value == 14:
                    prediction += 0.95 + (0.05 * self.aggression)
                elif card.color == trump.color:
                    prediction += (card.value * (0.050 + (0.005 * self.aggression))) + 0.3
                else:
                    prediction += (card.value * (0.030 + (0.005 * self.aggression)))
            prediction = round(RuleBasedAgent.bound(prediction, len(self.hand), 0), 0)
            self.prediction = prediction
            return prediction

    def announce_result(self, num_tricks_achieved, reward):
        """
        Occurs after each round, evaluating the score of the player and resetting round-based variables
        :param num_tricks_achieved:
        :param reward:
        :return:
        """

        if self.use_predictor:
            super().announce_result(num_tricks_achieved, reward)
            self.predictor.add_game_result(self.prediction_x, num_tricks_achieved)

        else:
            self.wins = num_tricks_achieved
            self.reward = reward
            self.score += reward
            self.hand = []

        if 60 / self.num_players == self.round:
            self.round = 1
        else:
            self.round += 1

    def play_card(self, trump, first, played, players, played_in_game, first_player_index):
        """
        Finds the card whose win probability most closely matches that of the win desirability
        :param trump:
        :param first:
        :param played:
        :param players:
        :param played_in_game:
        :param first_player_index
        :return: best_card
        """
        played = list(filter(lambda card: not card is None, played.values()))
        played_in_game = sum(played_in_game.values(), [])

        win_desirability = self.win_desirability(players)
        best_card = self.get_playable_cards(first)[0]
        best_delta = abs(win_desirability - self.win_probability(played, best_card, trump, first, players,
                                                                 played_in_game))
        best_win_likelihood = 0
        round_winning_cards = []
        # calculates the win probability for every playable card
        for card in self.get_playable_cards(first):
            if card != best_card:
                win_likelihood = self.win_probability(played, card, trump, first, players, played_in_game)
                if win_likelihood == 1: round_winning_cards.append(card)
                delta = abs(win_desirability - win_likelihood)
                # tries to find a card that minimizes the delta between the win desirability and the win probability
                if delta < best_delta:
                    best_card = card
                    best_delta = delta
                if win_likelihood > best_win_likelihood: best_win_likelihood = win_likelihood
        # If going to lose anyway then play the worst card (the one with the greatest number of stronger cards)
        played_cards = played_in_game
        if best_win_likelihood == 0:
            if win_desirability > 0:
                counter = 0
                for card in self.get_playable_cards(first):
                    number_of_stronger_cards = self.number_of_stronger_cards_remaining(card, trump, first, played_cards)
                    if number_of_stronger_cards > counter:
                        best_card = card
                        counter = number_of_stronger_cards
            else:  # we want to get rid of our most valuable card
                counter = 60
                for card in self.get_playable_cards(first):
                    number_of_stronger_cards = self.number_of_stronger_cards_remaining(card, trump, first, played_cards)
                    if number_of_stronger_cards < counter:
                        best_card = card
                        counter = number_of_stronger_cards
        elif len(round_winning_cards) > 1:  # Play the weakest possible winning hand
            counter = 0
            for card in round_winning_cards:
                number_of_stronger_cards = self.number_of_stronger_cards_remaining(card, trump, first, played_cards)
                if number_of_stronger_cards > counter:
                    best_card = card
                    counter = number_of_stronger_cards
        # debug output after each trick
        if self.DEBUG:
            print("\033[1;32;40mRound: " + str(self.round) + " | Trick: " + str(self.round - len(self.hand) + 1))
            print("\033[1;37;40mPlayed: " + str(played) + " | Total Cards Played: " + str(len(played_in_game)))
            print("Prediction: " + str(self.prediction) + " | Wins: " + str(self.wins))
            print("Hand: " + str(self.hand) + " | Trump: " + str(trump))
            print("Win Probability: " + str(self.win_probability(played, best_card, trump, first, players,
                                                                 played_in_game)) + " | Win Desirability: " + str(
                win_desirability))
            print("Chosen Card: " + str(best_card) + " | " + str(
                self.number_of_stronger_cards_remaining(best_card, trump, first, played_in_game)))
        self.hand.remove(best_card)
        return best_card

    def get_trump_color(self):
        """
        Determines trump color by choosing the color the agent has the most of in its hand
        Returns:
            str: color of trump
        """
        color_counter = collections.Counter()
        for card in self.hand:
            color = card.color
            if color == "White":
                continue
            color_counter[color] += 1
        if not color_counter.most_common(1):
            return self.hand[random.randint(0, len(self.hand) - 1)].color
        else:
            return color_counter.most_common(1)[0][0]

    def win_probability(self, played, card, trump, first, players, played_in_game):
        """
        Given a card and the current state, this algorithm calculates the probability of winning by seeing if the card is
        stronger than those already played and estimates the chance that a stronger card will be played in the future.
        :param played:
        :param card:
        :param trump:
        :param first:
        :param players:
        :return: probability of winning
        """
        if first is None:  # The probability I win is based only on the possibility that another agent has a
            # stronger card and plays it
            probability = ((61 - len(played_in_game) - len(self.hand)) -
                           self.number_of_stronger_cards_remaining(card, trump, first, played_in_game)) \
                          / (61 - len(played_in_game) - len(self.hand))
            return probability
        else:
            for other_card in played:
                if self.round == (60 / len(players)): trump = first
                # if there is a played card stronger than this card then the probability is instantly 0
                if self.strongest_card(other_card, card, trump, first) == True:
                    return 0
            if len(played) == len(players) - 1:
                # However if there is no stronger card and the agent is the last to play then the probability is 1
                return 1
            else:
                # if not then calculate the probability as before (based on how many stronger cards remain
                probability = ((61 - len(played_in_game) - len(self.hand)) - self.number_of_stronger_cards_remaining(
                    card, trump, first, played_in_game)) \
                              / (61 - len(played_in_game) - len(self.hand))
                return probability

    def win_desirability(self, players):
        """
        Approximates the desire to win based on how many rounds are left to play and how many predictions are left to make
        and also dependent on how many tricks left for the opposition to make
        :param players:
        :return: win desirability
        """
        if (self.prediction - self.wins) >= len(self.hand):
            return 1
        elif self.prediction <= self.wins:
            return 0
        else:
            desirability = 1.3 * (self.prediction - self.wins) / len(self.hand)
            for player in players:
                if player != self:
                    desirability += (1 / (len(players) + 1)) * (player.prediction - player.wins) / len(self.hand)
            desirability += 0.1 * np.cos((self.round - len(self.hand)) * (np.pi / self.round))
            return RuleBasedAgent.bound(desirability, 1, 0)

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

    def cards_left_by_color(self, color, target_card, played_in_game):
        """
        Keeps track of all cards played in that round and deduces how many cards of that color are left
        :param color:
        :param card: removes this card from the counting procedure
        :return: number_of_cards_of_that_color_left
        """
        all_cards_left = []
        for card in played_in_game:
            all_cards_left.append(card)
        for card in self.hand:
            all_cards_left.append(card)
        all_cards_left.remove(target_card)
        cards_by_color_left = []
        for card in all_cards_left:
            if card.color == color:
                cards_by_color_left.append(card)
        return cards_by_color_left

    def number_of_stronger_cards_remaining(self, card, trump, first, played_in_game):
        """
        Estimates the number of stronger cards remaining in the deck
        :param card:
        :param trump:
        :param first:
        :param played_in_game:
        :return: number of stronger cards
        """
        if card.value == 14: return 0
        if card.value == 0: return 61 - len(played_in_game) - len(self.hand)
        played_wizard_counter = 0
        played_trump_higher_counter = 0
        played_trump_lower_counter = 0
        played_second_trump_higher_counter = 0
        played_second_trump_lower_counter = 0
        played_remainder_counter = 0
        for played_card in played_in_game:
            if played_card.value == 14: played_wizard_counter += 1
        for played_card in self.cards_left_by_color(trump.color, card, played_in_game):
            if played_card.value > card.value:
                played_trump_higher_counter += 1
            else:
                played_trump_lower_counter += 1
        if first is None:
            color = card.color
        else:
            color = first.color
        for played_card in self.cards_left_by_color(color, card, played_in_game):
            if played_card.value > card.value:
                played_second_trump_higher_counter += 1
            else:
                played_second_trump_lower_counter += 1
        for other_color in ["Red", "Green", "Yellow", "Blue"]:
            if other_color != trump and other_color != color:
                for played_card in self.cards_left_by_color(other_color, card, played_in_game):
                    if played_card.value > card.value: played_remainder_counter += 1

        # If it is the last round then ignore trumps
        if self.round == (60 / self.num_players):
            stronger_cards = (13 - played_second_trump_higher_counter - card.value) + (4 - played_wizard_counter)
        # The amount of stronger cards is dependent on the amount of higher cards minus those already played
        elif card.color == trump.color:  # for trump colored card
            stronger_cards = (13 - played_trump_higher_counter - card.value) + (4 - played_wizard_counter)
        elif first is not None and card.color == first.color:  # for a card that has the same colour as the first
            stronger_cards = (13 - played_second_trump_higher_counter - card.value) + \
                             (12 - played_trump_higher_counter - played_trump_lower_counter) + (
                                         4 - played_wizard_counter)
        else:  # if the color of the card is none of them
            stronger_cards = (12 - played_trump_higher_counter - played_trump_lower_counter) + \
                             (12 - played_second_trump_higher_counter - played_second_trump_lower_counter) + \
                             (24 - played_remainder_counter - card.value * 2) + (4 - played_wizard_counter)
        # checks to see that the amount of stronger cards is not greater than the amount of cards already left
        if stronger_cards > (61 - len(played_in_game) - len(self.hand)):
            return (61 - len(played_in_game) - len(self.hand))
        else:
            return stronger_cards

    def strongest_card(self, new_card, old_card, trump, first_card):
        """Determines whether the new played card wins the trick

        :param new_card: card that contests with current winning card
        :param old_card: card currently winning the trick
        :param trump:
        :param first_card: first card played. Determines the suit of the trick. May be None
        :return: bool: True if the new_card wins, taking into account trump colors, first_card color and order.
        """

        # If a Z was played first, it wins.
        if old_card.value == 14:
            return False
        # If not and the new card is a Z, the new card wins.
        if new_card.value == 14:
            return True
        # First N wins, so if the second card is N, it always wins.
        if new_card.value == 0:
            return False
        # Second N wins only if new_card is NOT N.
        elif old_card.value == 0:
            return True
        # If they are both colored cards, the trump color wins.
        if old_card.color == trump.color:
            if new_card.color != trump.color:
                return False
            else:  # If both are trump color, the higher value wins.
                return old_card.value < new_card.value
        else:
            # old_card is not trump color, then if new_card is, new_card wins
            if new_card.color == trump.color:
                return True
            else:
                # Neither are trump color, so check for first color.
                if old_card.color == first_card.color:
                    if new_card.color != first_card.color:
                        # old card is first_card color but new card is not, old wins.
                        return False
                    else:
                        # Both are first_card color, bigger value wins.
                        return old_card.value < new_card.value
