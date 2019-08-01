from game_engine.card import Card


class Trick(object):
    """Trick object, plays one trick and determines the winner

    Attributes:
        trump_card (Card): specifies trump_color
        players (:obj: `list` of :obj: `int`): list of player objects
        first_player (int): index of player that is coming out in this round
        played_cards_in_round(:obj: `list` of :obj: `Card`): list of cards already played in the round
                                                             (a round consists of several tricks)
    """

    def __init__(self, trump_card, players, first_player, played_cards_in_round):
        """
        Args:
            trump_card (Card): specifies trump_color
            players (:obj: `list` of :obj: `int`): list of player objects
            first_player (int): index of player that is coming out in this round
            played_cards_in_round(:obj: `list` of :obj: `Card`): list of cards already played in the round
                                                                 (a round consists of several tricks)
        """
        self.trump_card = trump_card
        self.players = players
        self.played_cards_in_round = played_cards_in_round
        self.first_player = first_player

    def play_trick(self):
        """every player plays a card. winner of the trick is determined

        Returns:
            (int, list of :obj: `Card`): index of winning player, list of cards played in the trick
        """
        # define trick variables
        first_card = None
        winning_card = None
        winning_player = None
        num_players = len(self.players)

        # dictionary (player index, card) which determines which player played which card during the trick
        trick_cards = dict()
        for index, player in enumerate(self.players):
            trick_cards[index] = None

        for i in range(len(self.players)):
            # Start with the first player and ascend, then reset at 0.
            player_index = (self.first_player + i) % num_players
            player = self.players[player_index]
            played_card = player.play_card(self.trump_card, first_card,
                                           trick_cards, self.players,
                                           self.played_cards_in_round, self.first_player)

            trick_cards[player_index] = played_card
            self.played_cards_in_round[player_index].append(played_card)

            # if it is the first card played in the trick
            # set first card played to determine suit to follow, if first card is N there is no suit in the trick
            if first_card is None and played_card.value != 0:
                first_card = played_card

            # determine if current player is new winner of the trick
            if winning_player is None or Trick.is_new_winner(new_card=played_card, old_card=winning_card,
                                                             trump=self.trump_card, first_card=first_card):
                winning_card = played_card
                winning_player = player_index

        return winning_player, trick_cards

    @staticmethod
    def is_new_winner(new_card, old_card, trump, first_card):
        """Determines wether the new played card wins the trick

        Args:
            new_card (Card): card that contests with current winning card
            old_card (Card): card currently winning the trick
            trump (Card): trump card
            first_card (Card): first card played. Determines the suit of the trick. May be None

        Returns:
            bool: True if the new_card wins, taking into account trump colors, first_card color and order.
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
