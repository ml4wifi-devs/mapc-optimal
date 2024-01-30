import numpy as np

from mapc_optimal.constants import BREAKING_POINT, CENTRAL_FREQUENCY, REFERENCE_DISTANCE, WALL_LOSS


def dbm_to_lin(x: np.ndarray) -> np.ndarray:
    """
    Converts dBm to a linear scale.

    Parameters
    ----------
    x : array_like
        Input in dBm.

    Returns
    -------
    array_like
        Output in a linear scale.
    """

    return np.power(10., x / 10.).astype(float)


def lin_to_dbm(x: np.ndarray) -> np.ndarray:
    """
    Converts linear scale to dBm.

    Parameters
    ----------
    x : array_like
        Input in a linear scale.

    Returns
    -------
    array_like
        Output in dBm.
    """

    return 10. * np.log10(x).astype(float)


def tgax_path_loss(distance: np.ndarray, walls: np.ndarray) -> np.ndarray:
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


def positions_to_path_loss(pos: np.ndarray, walls: np.ndarray) -> np.ndarray:
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
    array_like
        Two dimensional array of path losses (dB) between all nodes.
    """

    distance = np.sqrt(np.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    return tgax_path_loss(distance, walls)
