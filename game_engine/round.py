import random

from game_engine.player import AverageRandomPlayer
from game_engine.card import Card
from game_engine.deck import Deck
from game_engine.trick import Trick


class Round:
    """Round object, plays a number of tricks and awards points depending
    on the outcome of the tricks and the predictions.

    Attributes:
        round_num (int): round number --> determines how many cards are dealt to each player
        players (:obj: `list` of :obj: `int`): list of player objects
        deck (:obj: `Deck`): deck containing all cards
        predictions (:obj: `list` of `int): list containing predictions of each player
        trump_card (:obj: `Card`, None): trump color of the round. Is initialized with `Card` in play method.
        first_player (int): index of player that is coming out in this round
        played_cards (:obj: `list` of :obj: `Card`): list of cards that have already been played in the round
    """

    def __init__(self, round_num, players):
        """
        Args:
            round_num (int): round number --> determines how many cards are dealt to each player
            players (:obj: `list` of :obj: `int`): list of player objects
        """
        self.round_num = round_num
        self.players = players
        self.deck = Deck()
        self.predictions = [-1]*len(players)
        self.trump_card = None
        # -1 adjusts for 1-index in game numbers and 0-index in players
        self.first_player = (round_num - 1) % len(players)
        self.played_cards = dict()

    def play_round(self):
        """Plays on round. Determines Trump Color. Asks for Predictions. Play all Tricks. Determine Scores for each Player

        Returns:
            list of int: scores for each player
        """
        # Holds the played cards for each player (index of player in players are the keys)
        self.played_cards = dict()
        for index, player in enumerate(self.players):
            self.played_cards[index] = []

        # Determining the Trump Color
        self.trump_card = self.distribute_cards()[0]

        if self.trump_card is None:
            # We distributed all cards, the trump is N. (No trump)
            self.trump_card = Card("White", 0)

        if self.trump_card.value == 14:
            # Trump card is a Z, ask the dealer for a trump color.
            self.trump_card.color =\
                self.players[self.first_player].get_trump_color()

        # Now that each player has a hand, ask for predictions.
        self.ask_for_predictions()
        # print("Final predictions {}".format(self.predictions))

        # Reset and initialize all wins.
        wins = [0] * len(self.players)
        for i, player in enumerate(self.players):
            player.wins = wins[i]

        for trick_num in range(self.round_num):
            # Play a trick for each card in the hand (or round number).
            trick = Trick(self.trump_card, self.players, self.first_player, self.played_cards)
            winner, trick_cards = trick.play_trick()

            # Trick winner gets a win and starts the next trick.
            wins[winner] += 1
            self.first_player = winner

            # Update wins
            for i, player in enumerate(self.players):
                player.wins = wins[i]
            # print("Player {} won the trick!".format(winner))
        return self.get_scores(wins)

    def distribute_cards(self):
        # Draw as many cards as game num.
        for player in self.players:
            player.hand += self.deck.draw(self.round_num)
        # Flip the next card, that is the trump card.
        if self.deck.is_empty():
            return [None]
        else:
            return self.deck.draw()

    def ask_for_predictions(self):
        num_players = len(self.players)
        for i in range(num_players):
            # Start with the first player and ascend, then reset at 0.
            current_player_index = (self.first_player + i) % num_players
            player = self.players[current_player_index]
            prediction = player.get_prediction(trump=self.trump_card, num_players=len(self.players))
            self.predictions[current_player_index] = prediction
            # print("Player {} predicted {}".format(current_player_index, prediction))

    def get_scores(self, wins):
        scores = [0]*len(self.players)
        for i, player in enumerate(self.players):
            difference = self.predictions[i] - wins[i]
            if difference == 0:
                scores[i] = 20 + wins[i]*10
            else:
                scores[i] = -10*abs(difference)
            # print("Player score for this round {}: {}".format(i, scores[i]))
            player.announce_result(wins[i], scores[i])
        return scores


if __name__ == "__main__":
    print("Playing a random round No. 9 with 4 players.")
    random.seed(4)

    # generate four players
    players = []
    for player in range(4):
        players.append(AverageRandomPlayer())
    round = Round(round_num=4, players=players)
    round.play_round()
