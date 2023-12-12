import unittest

from ml4wifi.envs.scenarios.static import simple_scenario_5
from ml4wifi.solver import Solver
from ml4wifi.solver.utils import positions_to_path_loss


class SolverTestCase(unittest.TestCase):
    def test_simple_network(self):
        s = simple_scenario_5()

        associations = s.get_associations()
        ap = list(associations.keys())
        sta = [s for stations in associations.values() for s in stations]
        path_loss = positions_to_path_loss(s.pos, s.walls)

        solver = Solver(sta, ap)
        solver(path_loss)
