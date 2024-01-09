import unittest

from mapc_optimal.envs.static_scenarios import simple_scenario_5
from mapc_optimal.solver import Solver
from mapc_optimal.solver.utils import positions_to_path_loss


class SolverTestCase(unittest.TestCase):
    def test_simple_network(self):
        s = simple_scenario_5()

        associations = s.get_associations()
        ap = list(associations.keys())
        sta = [s for stations in associations.values() for s in stations]
        path_loss = positions_to_path_loss(s.pos, s.walls)

        solver = Solver(sta, ap)
        result, rate = solver(path_loss)

        ...
