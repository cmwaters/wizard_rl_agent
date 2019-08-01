from game_engine.game import Game
from game_engine.player import RandomPlayer
import time
import sys
sys.path.append('../')


games = 5000

players = [RandomPlayer() for _ in range(4)]

initial = time.perf_counter()
last = initial

for i in range(games):
    if i % 100 == 0:
        print("{}/{}  time: {}".format(i, games, time.perf_counter() - last))
        last = time.perf_counter()
    wiz = Game(players=players)
    wiz.play_game()

print(time.perf_counter() - initial)