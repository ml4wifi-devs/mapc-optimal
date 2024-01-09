from functools import partial
from typing import Dict, Optional

import jax
from chex import Array, Scalar, PRNGKey
from mapc_sim.sim import network_data_rate

from mapc_optimal.envs.scenario import Scenario


class DynamicScenario(Scenario):
    """
    Dynamic scenario with fixed noise standard deviation. The configuration of node positions, MCS,
    and tx power is variable.

    Parameters
    ----------
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.
    associations: Dict
        Dictionary of associations between access points and stations.
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    """

    def __init__(
            self,
            sigma: Scalar,
            associations: Dict,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None
    ) -> None:
        super().__init__(associations, walls, walls_pos)
        self.data_rate_fn = jax.jit(partial(network_data_rate, sigma=sigma, walls=self.walls))

    def __call__(self, key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array) -> Scalar:
        return self.data_rate_fn(key, tx, pos, mcs, tx_power)

    def plot(self, pos: Array, filename: str = None) -> None:
        super().plot(pos, filename)
