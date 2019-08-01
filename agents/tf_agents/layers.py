from typing import Tuple

from agents.rl_agent import ACTION_DIMENSIONS

output_dimensions = ACTION_DIMENSIONS


def equal_spacing_fc(num_hidden_layers: int, input_dimensions: int) -> Tuple[int]:
    """Return a list of hidden layer sizes which have equal spacing"""
    diff = input_dimensions - output_dimensions
    return tuple(int(input_dimensions - layer * diff / (num_hidden_layers + 1))
                 for layer in range(1, num_hidden_layers + 1))
