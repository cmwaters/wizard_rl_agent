class Card(object):
    """Card object containing information about color and value of the card

    Note:
        value 0 specifies jester (N)
        value 14 specifies wizard (Z)
        wizards and jesters have `White` color

    Attributes:
         color (str): color of the card
         value (int): value of the card
    """
    colors = ("White", "Green", "Red", "Blue", "Yellow")
    DIFFERENT_CARDS = 54

    def __init__(self, color, value):
        """
        Args:
            color (str): color of the card
            value (int): value of the card
        """
        if color not in Card.colors or value > 14 or value < 0:
            raise ValueError
        if color == "White" and value not in (0, 14):
            raise ValueError
        if color in Card.colors[1:] and value in (0, 14):
            raise ValueError
        self.color = color
        self.value = value
        self.int = self.__int__()

    def __str__(self):
        return "{} {}".format(self.color, self.value)

    def __repr__(self):
        return str(self)

    def __int__(self):
        # Used for feature vector translation.
        if self.color == "White":
            if self.value == 0:
                # N is 52
                return 52
            else:
                # Z is 53
                return 53
        # The rest is between 0-51 inclusive.
        return (Card.colors.index(self.color)-1)*13 + (self.value - 1)

    @staticmethod
    def int_to_card(x):
        if x == 52:
            return Card("White", 0)
        elif x == 53:
            return Card("White", 14)
        else:
            color = Card.colors[x//13 + 1]
            value = x % 13 + 1
            return Card(color, value)
