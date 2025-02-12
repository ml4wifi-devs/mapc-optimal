"""
Utility functions, including the function for calculation of the path loss from node positions using
the TGax channel model.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike


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


r"""
TGax channel model parameters 
https://mentor.ieee.org/802.11/dcn/14/11-14-0980-16-00ax-simulation-scenarios.docx (p. 19)
https://en.wikipedia.org/wiki/List_of_WLAN_channels#5_GHz_(802.11a/h/n/ac/ax)
"""
CENTRAL_FREQUENCY = 5.160
WALL_LOSS = 7.
BREAKING_POINT = 10.
REFERENCE_DISTANCE = 1.


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


def positions_to_path_loss(pos: ArrayLike, walls: ArrayLike, path_loss_fn: Callable = tgax_path_loss) -> NDArray:
    """
    Calculates the path loss for all nodes based on their positions and the wall positions.
    Channel is modeled using the TGax path loss model.

    Parameters
    ----------
    pos : array_like
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    walls : array_like
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).
    path_loss_fn: Callable
        A function that calculates the path loss between two nodes. The function signature should be
        `path_loss_fn(distance: Array, walls: Array) -> Array`, where `distance` is the matrix of distances
        between nodes and `walls` is the adjacency matrix of walls. By default, the simulator uses the
        residential TGax path loss model.

    Returns
    -------
    NDArray
        Two-dimensional array of path losses (dB) between all nodes.
    """

    distance = np.sqrt(np.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    return path_loss_fn(distance, walls)
