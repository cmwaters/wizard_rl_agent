#!/usr/bin/env python3

# This allows the file to be run in a test folder
# as opposed to having to be in the root directory
import sys

sys.path.append('../')

import os
import datetime
import json
import random
import subprocess
import itertools
import atexit
import psutil

import numpy as np
import tensorflow as tf

from game_engine.game import Game
from game_engine.player import Player, AverageRandomPlayer
from agents.rl_agent import RLAgent
from agents.tf_agents.tf_agents_ppo_agent import TFAgentsPPOAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.predictors import NNPredictor
from agents.predictors import RuleBasedPredictor
from agents.featurizers import OriginalFeaturizer
from agents.original.rl_agents import OriginalRLAgent


class TensorboardWrapper:
    def __init__(self):
        self.create_logdir()
        self.launch_tensorboard()

    def create_logdir(self):
        """
        Create logdir in /logs/$id where $id is counting up.
        Save the path to self.logdir.
        """

        path = 'logs/'

        test_count = 1
        while os.path.exists(path + str(test_count)):
            test_count += 1

        self.logdir = path + str(test_count) + '/'
        os.makedirs(self.logdir)

    def launch_tensorboard(self):
        tensorboard = subprocess.Popen(['tensorboard',
                                        '--logdir', self.logdir,
                                        '--reload_interval', str(10)],
                                       stdout=open(self.logdir + 'stdout', 'w'),
                                       stderr=open(self.logdir + 'stderr', 'w'))
        atexit.register(tensorboard.terminate)

    def set_game_num(self, game_num):
        self.game_num = game_num

    def view_as(self, agent, name=None):
        return TensorboardAgentView(self, agent, name)


class TensorboardAgentView:
    """
    Giving an instance of this class as an argument to some function makes it easier than
    working with filewriters directly because the function can just call .scalar(name, value)
    and doesn't have to keep track of the filewriter, giving game_num as argument every time
    and using `with self.filewriter.as_default(), ...` around everything.
    """

    def __init__(self, tensorboard_wrapper, agent, name):
        self.tb = tensorboard_wrapper

        if name is None:
            name = agent.name
        self.filewriter = tf.contrib.summary.create_file_writer(self.tb.logdir + name)

    def scalar(self, name, value):
        with self.filewriter.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, value, step=self.tb.game_num)

    def histogram(self, name, value):
        with self.filewriter.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.histogram(name, value, step=self.tb.game_num)


class AgentPool:
    """
    Keeps track of an (learning) agent and its past versions.
    Enables easy selection of players for one game so that the agent doesn't overfit
    against itself. This is done by sampling from the past versions.

    The exact behaviour how sampling happens will probably change, e.g. we might only
    sample from past versions in N% of the cases or only have a few past versions
    and the rest being the current version etc.

    Because we use an on-policy method we don't gain experience from the past versions
    and gain more experience when playing against the current version (but might overfit).
    """

    def __init__(self, main_agent, max_size, path=None):
        self.pool = []
        self.agent = main_agent
        self.max_size = max_size

        if path is None:
            path = 'pools/MainPool'
        self.path = path

        if os.path.exists(path):
            self._load()
        else:
            os.makedirs(os.path.dirname(path))
            self.save()

        self.precomputed_clones = [self.agent.clone() for p in range(3)]

    def select_players(self):
        """Get an array of 4 players which can be used for a new game"""

        if len(self.pool) < 3:
            return [self.agent] + self.precomputed_clones

        # playing only against past versions makes training 4x slower
        # because we don't get any experience from those past versions (on policy)
        # therefore we make some tradeoff here by varying the number
        # of past versions each game so that we can still get a bit more
        # experience while not overfitting against the current version

        # Another solution would of course be to use some off policy method
        # Maybe we should do that but I'm not sure about the number of changes
        # it would require

        # with p = 0.1 we play against only current versions (-> 3 current versions),
        # with p = 0.4 we play against 2 current versions, 1 past version
        # etc.
        num_past_versions = np.random.choice(4, p=[0.1, 0.4, 0.3, 0.2])
        num_additional_current_versions = 3 - num_past_versions

        return ([self.agent]
                + self.precomputed_clones[:num_additional_current_versions]
                + random.sample(self.pool, num_past_versions))

    def add_current_version(self):
        """
        Use the current agent given in __init__(main_agent, ...), clone a fixed version
        of it and add it to the pool for future random selection by self.select_players.
        """

        if len(self.pool) >= self.max_size:
            self.pool.pop(random.randrange(len(self.pool)))

        clone_name = self.agent.name + '@' + datetime.datetime.now().isoformat()

        clone = self.agent.clone(clone_name)
        clone.save_models()

        # we discard that clone and load it again because it currently shares models
        # with our original agent, i.e. once the original agent is updated,
        # this clone is also updated which we don't want

        clone = self.agent.__class__(name=clone_name,
                                     keep_models_fixed=True, featurizer=self.agent.featurizer)

        self.pool.append(clone)
        self.save()

    def save(self):
        """
        Save a list of names of agents in the pool to self.path (pools/MainPool) by default
        """

        with open(self.path, 'w') as f:
            json.dump([agent.name for agent in self.pool], f, indent=4)

    def _load(self):
        """
        Use the list of agent names saved previously to create corresponding agents
        with the same name (assumes that the agent loads its past version by itself
        when initialized with the same name) and adds all of them to the pool.
        """

        with open(self.path) as f:
            pool_data = json.load(f)
            for agent_name in pool_data:
                agent = self.agent.__class__(name=agent_name,
                                             keep_models_fixed=True, featurizer=self.agent.featurizer)
                self.pool.append(agent)


def tensorboard_plot(agent: Player, tb: TensorboardAgentView,
                     avg_score: float, win_percentage: float):
    """
    Use this function after a number of games have been played by some player
    and you got the results of the games (avg_score, win_percentage).
    These results will be plotted and additional data saved in the agent
    will also be plotted, i.e. predictor stats if the agent has one
    or valid_rate if tracked by the player (only for RLAgent).
    """

    tb.scalar('1_win_percentage', win_percentage)
    tb.scalar('2_score', avg_score)

    if hasattr(agent, 'valid_rate'):
        tb.scalar('7_valid_rate', agent.valid_rate)

    if not hasattr(agent, 'predictor'):
        return

    # this value is only available after the predictor has been trained
    # the frequency of this may be different than plotting frequency
    if isinstance(agent.predictor, NNPredictor) and agent.predictor.current_loss is not None:
        tb.scalar('5_predictor_loss', agent.predictor.current_loss)
        tb.scalar('3_predictor_acc', agent.predictor.current_acc)

        agent.predictor.current_loss = None
        agent.predictor.current_acc = None

    prediction_differences = agent.predictor.prediction_differences
    if len(prediction_differences) > 0:
        tb.scalar('6_prediction_differences', np.mean(prediction_differences))
        tb.histogram('1_prediction_differences', prediction_differences)

        # real prediction accuracy
        prediction_accuracy = (len(prediction_differences)
                               - np.count_nonzero(prediction_differences)) / len(prediction_differences)
        tb.scalar('4_predictor_acc_real', prediction_accuracy)
    agent.predictor.prediction_differences = []

    for amount_cards in range(0, 16):
        if len(agent.predictor.predictions['overall'][amount_cards]) == 0:
            continue

        # mean predictions
        for plt_name, datapoint_name in [
            ('7_overall_mean_predictions', 'overall'),
            ('8_correct_mean_predictions', 'correct_prediction'),
            ('9_incorrect_mean_predictions', 'incorrect_prediction')]:
            data = agent.predictor.predictions[datapoint_name][amount_cards]
            if len(data) == 0:
                continue
            tb.scalar(f'{plt_name}_{amount_cards}', np.mean(data))

        # prediction distributions
        for plt_name, datapoint_name in [
            ('2_overall_predictions', 'overall'),
            ('3_correct_predictions', 'correct_prediction'),
            ('4_incorrect_predictions', 'incorrect_prediction')]:
            tb.histogram(f'{plt_name}_{amount_cards}',
                         agent.predictor.predictions[datapoint_name][amount_cards])

        # reset predictions variable
        for e in ['overall', 'correct_prediction', 'incorrect_prediction']:
            agent.predictor.predictions[e][amount_cards] = []


def calculate_win_percentage(scores):
    """Given the scores for all players, calculate the win percentage for each player

    Args:
        scores (float array of dimensionality [num_games_played][num_players]):
            For each game and player, the number of points achieved.

    Returns: an np.ndarray(shape=(num_players,), dtype=float32)
        containing the win percentage for each player
    """

    scores = np.array(scores)
    player_indices, win_counts = np.unique(np.argmax(scores, axis=1), return_counts=True)
    win_percentages = np.zeros(scores.shape[1])
    win_percentages[player_indices] = win_counts / len(scores)
    return win_percentages


def plot_agents(tb, scores, agents, agents_to_plot):
    """
    Calculates the win percentages and calls plot_agent
    for each of the agents in agents_to_plot

    Args:
        tb (TensorboardWrapper): The tensorboard where to plot the data
        scores (float array of dimensionality [num_games_played][num_players]):
            For each game and player, the number of points achieved.
        agents ([Player]): All players participating in the game
        agents_to_plot ([int]): The positions of those agents in `agents`
            where data should be plotted
    """

    agents_to_plot = [(p, agents[p], tb.view_as(agents[p])) for p in agents_to_plot]
    mean_scores = np.mean(scores, axis=0)
    win_percentages = calculate_win_percentage(scores)

    for agent_position, agent, agent_tb_view in agents_to_plot:
        tensorboard_plot(agent, agent_tb_view,
                         mean_scores[agent_position], win_percentages[agent_position])


def play_games(player_selector, tb, agents_to_plot, flags, shuffle_positions=True):
    """Play games infinitly, plot results and yield the current game number every game

    Args:
        player_selector (function () => [Player]): Called every game to determine
            the players in this game.
        tb (TensorboardWrapper): The tensorboard to plot to
        agents_to_plot ([int]): The positions of those agents in the array returned by
            `player_selector` where data should be plotted
        flags ({ flag_name: flag_value }): Constants used throughout the program,
            usually things like how often stuff should be plotted / saved etc.
        shuffle_positions (bool): Whether to randomly shuffle the positions of players
            each game
    """

    def play_game_shuffled(agents):
        agents_with_positions = list(zip(agents, range(4)))
        random.shuffle(agents_with_positions)

        shuffled_agents = [agent for agent, old_position in agents_with_positions]
        shuffled_agents_to_plot = [position for position, (agent, old_position)
                                   in enumerate(agents_with_positions) if old_position in agents_to_plot]

        shuffled_scores = Game(players=shuffled_agents).play_game()

        unshuffled_scores = [None] * 4
        for shuffled_position, (agent, unshuffled_position) in \
                enumerate(agents_with_positions):
            unshuffled_scores[unshuffled_position] = shuffled_scores[shuffled_position]

        return unshuffled_scores

    def play_game_unshuffled(agents):
        return Game(players=agents).play_game()

    def play_game(agents):
        if shuffle_positions:
            return play_game_shuffled(agents)
        return play_game_unshuffled(agents)

    scores = []
    for game_num in itertools.count():
        process = psutil.Process(os.getpid())
        memory_usage = int(process.memory_info().rss / 2 ** 20)
        print(game_num, f'[memory usage: {memory_usage} MiB]')

        agents = player_selector()
        scores.append(play_game(agents))

        if game_num == 0:
            continue

        if game_num % flags['tensorboard_plot_frequency'] == 0:
            tb.set_game_num(game_num)
            plot_agents(tb, scores, agents, agents_to_plot)
            scores = []

        yield game_num


"""
The functions below can be seen as seperate tests. All of them play some number
of games and plot the results. The train_... functions also train some kind of agent
and save the model once in a while. The evaluate_... functions keep the main model
which we want to evaluate fixed (though the opponent might still be learning)
and only plot the game results.

All of the functions take at least these two arguments:
    tb (TensorboardWrapper): The tensorboard where to plot the data
    flags ({ flag_name: flag_value }): Constants used throughout the program,
        usually things like how often stuff should be plotted / saved etc.

The functions are called from main() quite directly based on the command line arguments.
"""


def train_with_self_play_against_newest_version(tb, flags):
    """
    Do self play with where all of the 3 opponents are of the same type
    and share models / paramenters with the agent itself.

    This means that the agent is getting the experience of 4 games after one game
    but it also means that it might overfit against itself.
    """

    agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer())
    agents = [agent, agent.clone(), agent.clone(), agent.clone()]

    for game_num in play_games(lambda: agents, tb, range(4), flags):
        if game_num % flags['agent_save_frequency'] == 0:
            agent.save_models()


def train_with_self_play_against_old_versions(tb, flags):
    """
    Do self play the 3 opponents are past fixed versions of the agent itself.
    Every flags['pool_save_frequency'] steps the agent is added to the pool
    as a cloned fixed version. The exact behaviour on how opponents are selected
    depends on the behaviour of the AgentPool class.
    """

    agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer())
    agent_pool = AgentPool(agent, max_size=flags['max_pool_size'])

    for game_num in play_games(agent_pool.select_players, tb, [0], flags):
        if game_num % flags['agent_save_frequency'] == 0:
            agent.save_models()

        if game_num % flags['pool_save_frequency'] == 0:
            agent_pool.add_current_version()


def train_original_agent(tb, flags):
    agent = OriginalRLAgent()
    agents = [agent]
    for i in range(3):
        agents.append(agent.clone())

    for game_num in play_games(lambda: agents, tb, range(4), flags):
        if game_num % flags['agent_save_frequency'] == 0:
            agent.save_models()


def train_rule_based_agent_with_predictor(tb, flags):
    """Trains Predictor of Rulebased agent against trained RL PPO Agents with fixed models"""
    agent = RuleBasedAgent(use_predictor=True)
    rl_agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer(), keep_models_fixed=True)
    agents = [agent, rl_agent]

    for i in range(2):
        agents.append(rl_agent.clone())

    for game_num in play_games(lambda: agents, tb, range(4), flags):
        if game_num % flags['agent_save_frequency'] == 0:
            agent.save_models()


def evaluate(tb, flags, other_agents):
    """Evaluate the TFAgentsPPOAgent against `other_agents`"""

    agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer(), keep_models_fixed=True)
    agents = [agent] + other_agents(agent)

    for game_num in play_games(lambda: agents, tb, range(4), flags):
        pass


def evaluate_rule_based(tb, flags):
    """
    Evaluate the RuleBasedAgent against 3 AverageRandomPlayers.
    This doesn't use the TFAgentsPPOAgent and only corresponds to the previous
    test_rule_based_agent.py which has been removed and replaced with this shorter version.
    """

    agents = [RuleBasedAgent()] + [AverageRandomPlayer(
        name='AverageRandomPlayer' + str(i)) for i in range(3)]
    for game_num in play_games(lambda: agents, tb, range(4), flags):
        pass


def main():
    default_flags = ({
        'tensorboard_plot_frequency': 20,
        'agent_save_frequency': 50,
        'pool_save_frequency': 100,
        'max_pool_size': 500,
    })

    # TODO maybe also make it possible to specify these flags as command line options
    flags = default_flags

    subcmds = ({
        'train_vs_old_self': (train_with_self_play_against_old_versions, []),
        'train_vs_current_self': (train_with_self_play_against_newest_version, []),
        'train_original': (train_original_agent, []),
        'train_rule_based': (train_rule_based_agent_with_predictor, []),
        'evaluate': (evaluate, [lambda agent:
                                [AverageRandomPlayer(), RuleBasedAgent(use_predictor=True), RuleBasedAgent()]]),
        # TODO some other evaluate_something could also be added here
        # which uses other opponents
        # TODO we allow the rule based agent predictor to learn while evaluating against it
        # maybe it should learn stuff before and then be fixed ?
        # But on the other hand if we can, while we are fixed, beat an agent which is still
        # learning against us, that's also not bad
        'evaluate_rule_based': (evaluate_rule_based, []),
        'evaluate_original': (
            evaluate, [lambda agent: [OriginalRLAgent(keep_models_fixed=True), RuleBasedAgent(), RuleBasedAgent()]])
    })

    if len(sys.argv) > 1:
        selected_subcmd = sys.argv[1]
    else:
        selected_subcmd = 'train_vs_old_self'

    selected_fn, args = subcmds[selected_subcmd]
    selected_fn(TensorboardWrapper(), flags, *args)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()

    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
