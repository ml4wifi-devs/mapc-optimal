import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from chex import Array, Numeric

from mapc_sim.constants import REFERENCE_DISTANCE
from mapc_sim.utils import tgax_path_loss as path_loss

tfd = tfp.distributions


def dbm_to_lin(x: Numeric) -> Numeric:
    """
    Converts dBm to linear scale.

    Parameters
    ----------
    x : Numeric
        Input in dBm.

    Returns
    -------
    Numeric
        Output in linear scale.
    """

    return jnp.power(10., x / 10.)


def lin_to_dbm(x: Numeric) -> Numeric:
    """
    Converts linear scale to dBm.

    Parameters
    ----------
    x : Numeric
        Input in linear scale.

    Returns
    -------
    Numeric
        Output in dBm.
    """

    return 10. * jnp.log10(x)


@jax.jit
def positions_to_path_loss(pos: Array, walls: Array) -> Array:
    """
    Calculates the path loss for all nodes based on their positions and the wall positions.
    Channel is modeled using the TGax path loss model.

    Parameters
    ----------
    pos : Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    walls : Array
        Adjacency matrix of walls. Each entry corresponds to a node.

    Returns
    -------
    Array
        Two dimensional array of path losses.
    """

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)
    return path_loss(distance, walls)
