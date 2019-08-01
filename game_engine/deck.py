import random

from game_engine.card import Card


class Deck(object):
    """All cards the game is played with

    Note:
        Four Colors with numbers 1- 13
        Four Wizards (Z)
        Four Jesters (N)
        Wizards and Jesters have artificial color `White`
    """
    def __init__(self):
        self.cards = []
        # Add four colors with 1-13 cards.
        for val in range(1, 14):
            for color in Card.colors[1:]:
                self.cards.append(Card(color, val))
        # Add four Zs (white, 14) and four Ns (white, 0)
        for _ in range(4):
            self.cards.append(Card("White", 0))
            self.cards.append(Card("White", 14))
        random.shuffle(self.cards)

    def draw(self, num=1):
        """Draw specified number of cards from the deck. Default is to draw 1 card.

        Args:
            num (int): number of cards to draw from the deck

        Returns:
            :obj: `list` of :obj: `Card`: list of Cards drawn from the top of the deck
        """
        drawn = self.cards[-num:]
        del self.cards[-num:]
        return drawn

    def is_empty(self):
        """Checks if deck is empty, meaning no cards are left in the deck

        Returns:
            bool: True if deck is empty, False else
        """
        return len(self.cards) <= 0