import random

from game_engine.round import Round
from game_engine.player import AverageRandomPlayer


class Game:
    """Wrapper for the whole game

    Attributes
        players (:obj: `list` of :obj: `Player`): list containing objects of players
        num_players (int): amount of players in the game
        rounds_to_play (int): rounds to play in the whole game
        scores (:obj: `list` of :obj: `int`): list containing scores for each player for the whole game

    """
    NUM_CARDS = 60

    def __init__(self, num_players=4, players=None):
        """        
        Args:
            num_players (int): amount of default players in the game, if `players` argument is None. 
                               Has to be between 2 and 6.
            players (:obj: `list` of :obj: `Player`): list containg objects of players. If not specified, game is played
                                                      with AverageRandomPlayer objects only. Number of players
        """
        self.players = []
        if players is None:
            assert 2 <= num_players <= 6, "Not enough players!" \
                                     "Give an array of players or a" \
                                     "number of players between [2-6]"
            for player in range(num_players):
                # Initialize all players
                # print("Creating players.")
                self.players.append(AverageRandomPlayer())
        else:
            self.players = players
            
        self.num_players = len(self.players)
        self.rounds_to_play = Game.NUM_CARDS // self.num_players
        self.scores = [0] * self.num_players

    def play_game(self):
        """ Starts a game with the generated players.

        Returns:
            list of int: The scores for each player.
        """
        # print("Playing a random game!")
        for round_num in range(1, self.rounds_to_play + 1):
            # print("Play Round No. {}".format(round_num))
            round = Round(round_num, self.players)
            score = round.play_round()
            # print(len(round.played_cards))
            for i in range(self.num_players):
                self.scores[i] += score[i]
            # print("Scores: {}".format(self.scores))
        # print("Final scores: {}".format(self.scores))
        for player in self.players:
            player.reset_score()
        return self.scores


if __name__ == "__main__":
    print("Playing a random game of 4 players.")
    random.seed(4)
    game = Game(num_players=4)
    # print(random.getstate())
    game.play_game()
