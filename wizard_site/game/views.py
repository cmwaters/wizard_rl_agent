from django.shortcuts import render, redirect
import sys
sys.path.append('../')
from game_engine.card import Card
from game_engine.trick import Trick
# from agents.rule_based_agent import RuleBasedAgent
from game_engine.round import Round
from game_engine.player import Player
from agents.tf_agents.tf_agents_ppo_agent import TFAgentsPPOAgent
from agents.featurizers import OriginalFeaturizer
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()


class TrickManager(Trick):

    def __init__(self, trump_card, players, first_player, played_cards_in_round):
        super().__init__(trump_card, players, first_player, played_cards_in_round)
        self.current_winner = self.first_player
        self.old_card = Card("White", 0)
        self.new_card = Card("White", 0)
        self.trick_cards = dict()
        self.first_card = Card("White", 0)


players = [TFAgentsPPOAgent(featurizer=OriginalFeaturizer()) for _ in range(3)] + [Player()]
# players = [RuleBasedAgent(), RuleBasedAgent(), RuleBasedAgent(), Player()]
game_round = Round(round_num=1, players=players)
trick = TrickManager(Card('White', 0), None, 0, [None])
blind = True


def home(request):

    return render(request, 'start.html')


def play_game(request):
    for player in players:
        if player.__class__.__name__ == "TFAgentsPPOAgent":
            player.__init__(keep_models_fixed=True, featurizer=OriginalFeaturizer())
        else:
            player.__init__()
    return redirect('play_round', game_round_no=1)


def play_round(request, game_round_no):
    game_round.__init__(round_num=game_round_no, players=players)
    for player in players:
        player.wins = 0
        player.prediction = -1
    game_round.played_cards = dict()
    for index, player in enumerate(game_round.players):
        game_round.played_cards[index] = []
    return redirect('get_prediction', game_round_no=game_round_no)


def get_prediction(request, game_round_no):
    print("Start of Round: " + str(game_round_no))
    for player in players:
        player.hand = game_round.deck.draw(game_round_no)
    if game_round.deck.is_empty():
        game_round.trump_card = Card("White", 0)
        print("Deck is empty. No trump color")
    else:
        game_round.trump_card = game_round.deck.draw()[0]
        while game_round.trump_card.color == "White" and not game_round.deck.is_empty():
            game_round.trump_card = game_round.deck.draw()[0]
            # game_round.trump_card.value = 0
            # if game_round.first_player != 3:
            #     print("Robot choosing suit")
            #     game_round.trump_card.color = players[game_round.first_player].get_trump_color()
            # else:
            #     print("User choosing suit")
            #     #TODO integrate the user choosing the trump color
            #     game_round.trump_card.color = players[3].hand[0].color
        # else:
            # game_round.played_cards.update({5: game_round.trump_card})
    for player in players:
        print(str(player.hand))
    if game_round_no > 10:
        width = len(players[1].hand) * 55
        height = len(players[2].hand) * 33
    else:
        width = len(players[1].hand) * 66
        height = len(players[2].hand) * 50
    print("Trump Color: " + str(game_round.trump_card.color))
    return render(request, 'game.html', {'left_agent': players[0], 'top_agent': players[1],
                                         'right_agent': players[2], 'human_player': players[3],
                                         'prediction_phase': True, 'round': game_round_no, 'width': width,
                                         'height': height, 'trump_card': game_round.trump_card,
                                         'prediction_range': range(0, game_round_no+1), 'blind': blind,
                                         'first_player': game_round.first_player + 1})


def receive_prediction(request, game_round_no, prediction):
    for player in players:
        if player.__class__.__name__ != "Player":
            player.prediction = int(player.get_prediction(game_round.trump_card, len(players)))
        else:
            player.prediction = prediction
    return redirect('get_play', game_round_no=game_round_no)


def get_play(request, game_round_no):
    print("Start of Trick: " + str(game_round_no - len(players[0].hand)))
    if game_round_no - len(players[0].hand) == 0:
        last_winner = game_round.first_player
    else:
        last_winner = trick.current_winner
    trick.__init__(game_round.trump_card, players, last_winner, game_round.played_cards)
    trick.trick_cards = dict()
    # To convert for the viewer
    trick_cards = []
    for index, player in enumerate(game_round.players):
        trick.trick_cards[index] = None
    print("First Player: " + str(trick.first_player))
    if len(players[0].hand) == game_round_no:  # Starting a new round
        trick.first_player = game_round.first_player
    player_index = trick.first_player
    while player_index != 3:
        print("Playable Cards for " + str(player_index) + ":")
        if player_index == trick.first_player:  # First player
            print(players[player_index].get_playable_cards(Card('White', 0)))
            trick.first_card = (players[player_index].play_card(game_round.trump_card, None, trick.trick_cards, players,
                                                                game_round.played_cards, game_round.first_player))
            trick.old_card = trick.first_card
            game_round.played_cards[player_index].append(trick.old_card)
            trick.trick_cards[player_index] = trick.old_card
        else:
            print(players[player_index].get_playable_cards(trick.first_card))
            trick.new_card = (players[player_index].play_card(game_round.trump_card, trick.first_card, trick.trick_cards, players,
                                                              game_round.played_cards, game_round.first_player))
            if trick.first_card.color == "White":
                trick.first_card = trick.new_card
            trick.trick_cards[player_index] = trick.new_card
            if trick.is_new_winner(trick.new_card, trick.old_card, game_round.trump_card, trick.first_card):
                trick.current_winner = player_index
                trick.old_card = trick.new_card
            game_round.played_cards[player_index].append(trick.old_card)
        trick_cards.append(trick.trick_cards[player_index])
        player_index = (player_index + 1) % len(players)
        print("Current Winner: " + str(trick.current_winner))
    if game_round_no > 10:
        width = len(players[1].hand) * 55
        height = len(players[2].hand) * 33
    else:
        width = len(players[1].hand) * 66
        height = len(players[2].hand) * 50
    return render(request, 'game.html', {'left_agent': players[0], 'top_agent': players[1],
                                         'right_agent': players[2], 'human_player': players[3],
                                         'prediction_phase': False, 'round': game_round_no, 'width': width,
                                         'height': height, 'trump_card': game_round.trump_card,
                                         'trick_cards': trick_cards, 'blind': blind})


def receive_play(request, game_round_no, trick_card):
    player_index = 3
    player_card = Card.int_to_card(trick_card)
    print(player_card.__str__())
    valid = False
    print("Playable Cards:")
    for card in players[player_index].get_playable_cards(trick.first_card):
        print(card)
        if player_card.__str__() == card.__str__():
            valid = True
            break
    if not valid:
        print("Incorrect Action")
        print(players[player_index].get_playable_cards(trick.first_card))
        if game_round_no > 10:
            width = len(players[2].hand) * 55
            height = len(players[2].hand) * 33
        else:
            width = len(players[2].hand) * 66
            height = len(players[2].hand) * 50
        trick_cards = []
        for index in range(len(players)):
            if index >= trick.first_player:
                if trick.trick_cards[index] is not None:
                    trick_cards.append(trick.trick_cards[index])
        return render(request, 'game.html', {'left_agent': players[0], 'top_agent': players[1],
                                      'right_agent': players[2], 'human_player': players[3],
                                      'prediction_phase': False, 'round': game_round_no, 'width': width,
                                      'height': height, 'trump_card': game_round.trump_card,
                                      'trick_cards': trick_cards, 'blind': blind})
    if trick.first_player == 3:
        trick.first_card = player_card
    elif trick.first_card.color == "White":
        trick.first_card = trick.new_card
    trick.new_card = player_card
    trick.trick_cards[player_index] = trick.new_card
    game_round.played_cards[player_index].append(trick.old_card)
    if trick.is_new_winner(trick.new_card, trick.old_card, game_round.trump_card, trick.first_card):
        trick.current_winner = player_index
        trick.old_card = trick.new_card
    card_index = 0
    for card in players[player_index].hand:
        if card.__str__() == trick.new_card.__str__():
            break
        else:
            card_index += 1
    players[player_index].hand.pop(card_index)
    # Move to the next player after the human player
    player_index = 0
    print("First Player Index: " + str(trick.first_player))
    # loop through the rest of the players that are after the human player but haven't played yet
    while player_index != trick.first_player:
        # print(players[player_index].hand)
        print("Playable Cards for " + str(player_index) + ":")
        print(players[player_index].get_playable_cards(trick.first_card))
        trick.new_card = (players[player_index].play_card(game_round.trump_card, trick.first_card, trick.trick_cards,
                                                          players, game_round.played_cards, game_round.first_player))
        print("Chosen Card: " + str(trick.new_card))
        if trick.first_card.color == "White":
            trick.first_card = trick.new_card
        trick.trick_cards[player_index] = trick.new_card
        if trick.is_new_winner(trick.new_card, trick.old_card, game_round.trump_card, trick.first_card):
            trick.current_winner = player_index
            trick.old_card = trick.new_card
        game_round.played_cards[player_index].append(trick.old_card)
        player_index += 1
        print("Winning Player: " + str(trick.current_winner))
    players[trick.current_winner].wins += 1
    return redirect('show_result', game_round_no)


def show_result(request, game_round_no):
    if len(players[0].hand) == 0:
        for player in players:
            if player.prediction == player.wins:
                player.score += player.prediction*10 + 20
                player.accuracy += 1
            else:
                player.score -= abs(player.prediction - player.wins)*10
        next = 'round'
        if game_round_no == 15:
            return redirect('end')
    else:
        game_round.first_player = (game_round.first_player + 1) % len(players)
        next = 'trick'
    trick_cards = []
    for index in range(len(players)):
        if trick.trick_cards[(trick.first_player + index) % 4] is not None:
            trick_cards.append(trick.trick_cards[(trick.first_player + index) % 4])
    if game_round_no > 10:
        width = len(players[1].hand) * 55
        height = len(players[2].hand) * 33
    else:
        width = len(players[1].hand) * 66
        height = len(players[2].hand) * 50
    return render(request, 'game.html', {'left_agent': players[0], 'top_agent': players[1],
                                         'right_agent': players[2], 'human_player': players[3],
                                         'prediction_phase': False, 'round': game_round_no, 'width': width,
                                         'height': height, 'trump_card': game_round.trump_card,
                                         'trick_cards': trick_cards, 'blind': blind,
                                         'winner': trick.current_winner + 1, 'next': next, 'nr': game_round_no + 1})


def end(request):
    if game_round.round_num < 15:
        return redirect('play_game')
    for player in players:
        player.accuracy = round((player.accuracy * 100 / 15),2)
    return render(request, 'end.html', {'player_1_score': players[0].score, 'player_2_score': players[1].score,
                                        'player_3_score': players[2].score, 'player_4_score': players[3].score,
                                        'player_1_acc': players[0].accuracy, 'player_2_acc': players[1].accuracy,
                                        'player_3_acc': players[2].accuracy, 'player_4_acc': players[3].accuracy})
