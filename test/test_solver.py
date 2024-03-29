import unittest

import numpy as np

from mapc_optimal import Solver, positions_to_path_loss


class SolverTestCase(unittest.TestCase):
    def test_simple_network(self):
        d_ap = 100.
        d_sta = 2.

        ap_pos = [
            [0 * d_ap, 0 * d_ap],  # AP A
            [1 * d_ap, 0 * d_ap],  # AP B
            [1 * d_ap, 1 * d_ap],  # AP C
            [0 * d_ap, 1 * d_ap],  # AP D
        ]

        dx = np.array([-1, 1, 1, -1]) * d_sta / np.sqrt(2)
        dy = np.array([-1, -1, 1, 1]) * d_sta / np.sqrt(2)

        sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
        pos = np.array(ap_pos + sta_pos)
        walls = np.zeros((pos.shape[0], pos.shape[0]))

        path_loss = positions_to_path_loss(pos, walls)
        sta = list(range(4, 20))
        ap = list(range(4))

        for opt_sum in [True, False]:
            solver = Solver(sta, ap, opt_sum=opt_sum)
            result, rate, obj = solver(path_loss, return_objectives=True)

            assert obj[-1] < 1e-5
            assert 550 < rate < 600
