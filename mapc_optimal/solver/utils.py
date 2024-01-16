import numpy as np
from mapc_sim.constants import REFERENCE_DISTANCE
from mapc_sim.utils import tgax_path_loss as path_loss
from numpy.typing import NDArray


def dbm_to_lin(x: NDArray) -> NDArray:
    """
    Converts dBm to linear scale.

    Parameters
    ----------
    x : array_like
        Input in dBm.

    Returns
    -------
    array_like
        Output in linear scale.
    """

    return np.power(10., x / 10.)


def lin_to_dbm(x: NDArray) -> NDArray:
    """
    Converts linear scale to dBm.

    Parameters
    ----------
    x : array_like
        Input in linear scale.

    Returns
    -------
    array_like
        Output in dBm.
    """

    return 10. * np.log10(x)


def positions_to_path_loss(pos: NDArray, walls: NDArray) -> NDArray:
    """
    Calculates the path loss for all nodes based on their positions and the wall positions.
    Channel is modeled using the TGax path loss model.

    Parameters
    ----------
    pos : array_like
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    walls : array_like
        Adjacency matrix of walls. Each entry corresponds to a node.

    Returns
    -------
    array_like
        Two dimensional array of path losses.
    """

    distance = np.sqrt(np.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = np.clip(distance, REFERENCE_DISTANCE, None)
    return path_loss(distance, walls)
