"""
Utility functions, including the function for calculation of the path loss from node positions using
the TGax channel model.
"""

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray, ArrayLike

from mapc_optimal.constants import BREAKING_POINT, CENTRAL_FREQUENCY, REFERENCE_DISTANCE, WALL_LOSS


class OptimizationType(Enum):
    SUM = auto()
    MAX_MIN = auto()
    MAX_MIN_BASELINE = auto()
    PROPORTIONAL = auto()


def dbm_to_lin(x: ArrayLike) -> NDArray:
    """
    Converts dBm to a linear scale.

    Parameters
    ----------
    x : array_like
        Input in dBm.

    Returns
    -------
    NDArray
        Output in a linear scale.
    """

    return np.power(10., x / 10.)


def lin_to_dbm(x: ArrayLike) -> NDArray:
    """
    Converts linear scale to dBm.

    Parameters
    ----------
    x : array_like
        Input in a linear scale.

    Returns
    -------
    NDArray
        Output in dBm.
    """

    return 10. * np.log10(x)


def tgax_path_loss(distance: ArrayLike, walls: ArrayLike) -> NDArray:
    r"""
    Calculates the path loss according to the TGax channel model [1]_.

    Parameters
    ----------
    distance: array_like
        Distance between nodes.
    walls: array_like
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).

    Returns
    -------
    array_like
        Two dimensional array of path losses (dB) between all nodes.

    References
    ----------
    .. [1] https://www.ieee802.org/11/Reports/tgax_update.htm#:~:text=TGax%20Selection%20Procedure-,11%2D14%2D0980,-TGax%20Simulation%20Scenarios
    """

    distance = np.clip(distance, REFERENCE_DISTANCE, None)
    return (40.05 + 20 * np.log10((np.minimum(distance, BREAKING_POINT) * CENTRAL_FREQUENCY) / 2.4) +
            (distance > BREAKING_POINT) * 35 * np.log10(distance / BREAKING_POINT) + WALL_LOSS * walls)


def positions_to_path_loss(pos: ArrayLike, walls: ArrayLike) -> NDArray:
    """
    Calculates the path loss for all nodes based on their positions and the wall positions.
    Channel is modeled using the TGax path loss model.

    Parameters
    ----------
    pos : array_like
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    walls : array_like
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).

    Returns
    -------
    NDArray
        Two-dimensional array of path losses (dB) between all nodes.
    """

    distance = np.sqrt(np.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    return tgax_path_loss(distance, walls)
