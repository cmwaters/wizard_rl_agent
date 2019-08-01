#!/usr/bin/env python3

"""This test is specifically used to examine the behaviour of some of the rule_based_agent functions"""

from random import seed
import numpy as np
import sys
sys.path.append('../')

from game_engine.card import Card
from game_engine.deck import Deck
from agents.rule_based_agent import RuleBasedAgent

joker = Card('White', 14)
deck = Deck()
card = Deck.draw(deck)[0]
played = Deck.draw(deck, 3)
trump = Deck.draw(deck)[0]
print(played)
player = RuleBasedAgent()
print(card)
print(trump)
print(player.win_probability(played, card, trump, played[0], range(4)))
print(player.number_of_stronger_cards_remaining(card, trump, played[0], played))

