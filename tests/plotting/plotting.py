import tensorflow as tf
import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from operator import itemgetter
from scipy.signal import savgol_filter

### Plot Settings ###

PLOT_SIZE = (16, 4)
PLOT_STYLE = 'ggplot'

LEGEND = {
    'TFAgentsPPOAgent': {'name': 'RL Agent', 'color': '#38739B'},
    'RuleBasedAgent': {'name':'Rule based play & prediction', 'color': 'C0'},
    'AverageRandomPlayer': {'name': 'Random play & avg. prediction', 'color': 'C2'},
    'RuleBasedAgentPredictor': {'name': 'Rule based play & NN prediction', 'color': 'C3'}
}

plt.rcParams["figure.figsize"] = PLOT_SIZE
plt.style.use(PLOT_STYLE)


### Loading Data Functionalities ###

def load_single_file(file_path):
    """Loads all summary from one tensorboard logfile and structures it in two dictionaries
    One for scalar values
    One for histograms

    The Dictionaries have following structure:

    {'summary_name': [np.array([step, value]), np.array([step, value])], 'summary_name': ....}

    Note:
        in our case we only have on step, value pair per summary per file

    Args:
        file_path (str): path to summary file

    Returns:
        dict: {'scalars': scalars_dict, 'histos': histos_dict}
    """
    scalars = dict()
    histos = dict()
    for index, summary in enumerate(tf.train.summary_iterator(file_path)):
        if summary.file_version:
            # first entry has info about file
            pass
        else:
            tag = summary.summary.value[0].tag
            histo_value = summary.summary.value[0].histo
            scalar_value = summary.summary.value[0].simple_value
            step = summary.step

            if histo_value.ByteSize() > 0:
                if tag not in histos.keys():
                    histos[tag] = [np.array([step, histo_value])]
                else:
                    histos[tag].append(np.array([step, histo_value]))
            else:
                if tag not in scalars.keys():
                    scalars[tag] = [np.array([step, scalar_value])]
                else:
                    scalars[tag].append(np.array([step, scalar_value]))

    return {'scalars': scalars, 'histos': histos}


def load_data(experiment_number, load_histos=False, log_path=None, time_range=[0, sys.maxsize]):
    """loads all data for given experiment number

    Args:
        experiment_number (int): number of experiment folder
        load_histos (bool): if True data for histos are also loaded
        path (str): path where tensorboard logs are saved. If None default path is Wizard/tests/logs
        time_range (list): time range for which data should be loaded e.g. (0, 20000)

    Returns:
        dict: dictionary of all data for each player
    """

    # by default only load data for scalar values
    summary_types = ['scalars']
    if load_histos:
        summary_types.append('histos')

    if not log_path:
        log_path = '../logs/{}/*A*'.format(experiment_number)
    else:
        log_path = os.path.join(log_path, '*A*')

    print(f'files are loaded from paths {log_path}')

    # get file paths for each player
    player_paths = glob.glob(log_path)
    player_files = {player_path.split('/')[-1]: glob.glob('{}/events*'.format(player_path)) for player_path in
                    player_paths}

    # for each player load and combine all log files
    player_data = {}
    for player_name, player_files in player_files.items():
        player_dict = {f'{summary_type}': {} for summary_type in summary_types}
        for path in player_files:
            file_dict = load_single_file(path)
            for summary_type in summary_types:
                for tag, value in file_dict[summary_type].items():
                    # print(value)
                    if tag not in player_dict[summary_type].keys():
                        player_dict[summary_type][tag] = value
                    else:
                        player_dict[summary_type][tag] += value

        player_data[player_name] = player_dict

        # sort summaries by time step
        for summary_type in summary_types:
            for tag, value in player_data[player_name][summary_type].items():
                time_range_filtered_value = [x for x in value if time_range[0] <= x[0] <= time_range[1]]
                player_data[player_name][summary_type][tag] = sorted(time_range_filtered_value, key=itemgetter(0))

    return player_data


def print_infos(data_set):
    """prints info about available agents and their summaries"""
    print('##### Agents #####')
    for agent in data_set.keys():
        print(agent)

    print('\n')
    print('##### Summaries #####')
    print('\n')
    for agent, agent_dict in data_set.items():
        print(f'### {agent} ###')
        print('\n')
        for summary_type, summaries in agent_dict.items():
            print(f'## {summary_type} ##')
            print('\n')
            for summary, data in summaries.items():
                print(summary, f'({len(data)} time steps)')
            print('\n')


### Plotting Functionalities ###

def calculate_smoothing_window_length(time_steps, smoothing_factor, smoothing_polyorder):
    """returns the window length for savitzky golay filter

    smoothing_window_length = smoothing_factor * time_steps

    Note:
        he length of the filter window (i.e. the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `x`.

    Args:
        time_steps (int): amount of time steps in the data
        smoothing_factor (float): has to be between 0 and 1
        smoothing_polyorder (int): The order of the polynomial used to fit the samples.
    `                              polyorder` must be less than `window_length`.

    Returns:
        int: window length for the savaitzky golay filter
    """

    assert 0 <= smoothing_factor <= 1, 'Smoothing Factor needs to be in [0,1]'

    window_length = int(np.floor(smoothing_factor * time_steps))
    # polyorder needs to be less tahn window_length
    if smoothing_polyorder >= window_length:
        window_length = smoothing_polyorder + 1
    # window_length needs to be an odd number
    if window_length % 2 == 0:
        window_length -= 1

    print(window_length, time_steps)

    return window_length


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed



def plot_scalar(data, summary_name, agents=None, plot_original=True, smoothing=False, save_plot=True, plot_title=None,
                plot_name=None, smoothing_factor=0.5, smoothing_polyorder=4, ylim=None, ylabel=None):
    """plotting scalar metrics for tensorboard summaries and save plot to /plots

    Args:
        data (dict): data containing all summary data (return of load_data())
        summary_name (str): name of summary to plot. run print_info(data) to see which summaries are available
        agents (list): list of agents to plot. If None all agents in data will be plotted
        plot_original (bool): If True, plot the raw data
        smoothing (bool): If True, plot the smoothed data
        save_plot (bool): If True, save plot to file
        plot_title (str): title of the plot. If None, summary_name is the title
        plot_name (str): file name when plot is saved. If None, use plot_title for naming the file
        smoothing_factor (float): has to be in [0,1]
        smoothing_polyorder (int): The order of the polynomial used to fit the samples.
    `                              polyorder` must be less than `window_length`.
        ylim (tuple): lower and upper limit for y axes. e.g. (0,200)
    """
    if not plot_title:
        plot_title = summary_name

    if not plot_name:
        plot_name = plot_title

    if not ylabel:
        ylabel = summary_name

        # agents for which data should be plotted
    if not agents:
        agents = data.keys()
    else:
        # check if provided agents exist
        for agent in agents:
            assert agent in data.keys(), f'Agent `{agent}` does not exist'

    agents_with_data = [agent for agent in agents if summary_name in data[agent]['scalars'].keys()]
    print(f'Agents that have data for metric `{summary_name}`: ', agents_with_data)

    # plotting
    fig, ax = plt.subplots()

    labels = []
    for index, agent in enumerate(agents_with_data):
        agent_data = data[agent]['scalars'][summary_name]
        x = [x[0] for x in agent_data]
        y = [x[1] for x in agent_data]

        if smoothing:
            # smooth data with savitzky_golay filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
            # window size and polyorder can be adjusted or could be extracted as paramters
            smoothing_window_length = calculate_smoothing_window_length(time_steps=len(y),
                                                                        smoothing_factor=smoothing_factor,
                                                                        smoothing_polyorder=smoothing_polyorder)
            # smoothed_data = savgol_filter(y, window_length=smoothing_window_length, polyorder=3)
            smoothed_data = smooth(y, weight=smoothing_factor)
            ax.set_xlabel('games', labelpad=5)
            ax.set_ylabel(ylabel)
            # ax.plot(x, smoothed_data, linewidth=3.5, color=f'C{index}')
            ax.plot(x, smoothed_data, linewidth=3.5, color=LEGEND[agent]["color"])
            ax.tick_params(axis='both', labelsize=15)
            labels.append(f'{LEGEND[agent]["name"]} (smoothed)')

        if plot_original:
            if smoothing:
                linestyle = '--'
                opacity = 0.5
            else:
                linestyle = '-'
                opacity = 1

            ax.plot(x, y, color=LEGEND[agent]["color"], linestyle=linestyle, alpha=opacity)
            labels.append(LEGEND[agent]["name"])

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(labels,loc=4, prop={'size':15})

    # ax.set_title(plot_title)
    # tikzplotlib.save(f'plots/{plot_name}.tikz')
    # plt.show()

    if save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')

        plt.draw()
        fig.savefig(f'plots/{plot_name}.svg', bbox_inches='tight')
