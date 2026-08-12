import unittest

import numpy as np

from mapc_optimal import Solver, OptimizationType, positions_to_path_loss


def station_rates(result):
    rates = {}

    for c, share in result['shares'].items():
        for l, rate in result['link_rates'][c].items():
            rates[l[1]] = rates.get(l[1], 0.) + share * rate

    return sorted(round(r, 6) for r in rates.values())


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
        baselines = {f'STA_{i}': 0. for i in sta}

        for opt_type in OptimizationType:
            if opt_type == OptimizationType.LEXICOGRAPHIC:  # covered by test_lexicographic, too slow for this network
                continue

            solver = Solver(sta, ap, opt_type=opt_type)
            result, rate, obj = solver(path_loss, baseline=baselines, return_objectives=True)

            assert obj[-1] < 1e-5
            print(rate)
            assert 650 < rate < 700

    def test_lexicographic(self):
        # AP 0 and AP 1 interfere with each other, AP 2 is far away and can always transmit
        ap_pos = [[0., 0.], [15., 0.], [400., 0.]]
        sta_pos = [[x + s * 2., 2.] for x, _ in ap_pos for s in (-1, 1)]
        pos = np.array(ap_pos + sta_pos)
        walls = np.zeros((pos.shape[0], pos.shape[0]))

        path_loss = positions_to_path_loss(pos, walls)
        sta = list(range(3, 9))
        ap = list(range(3))
        associations = {a: [3 + 2 * a, 4 + 2 * a] for a in ap}

        max_min, _ = Solver(sta, ap, opt_type=OptimizationType.MAX_MIN)(path_loss, associations)
        lexicographic, _ = Solver(sta, ap, opt_type=OptimizationType.LEXICOGRAPHIC)(path_loss, associations)

        max_min_rates = station_rates(max_min)
        lexicographic_rates = station_rates(lexicographic)

        # the worst station is the same as in the max-min optimization, but the following ones are better
        assert lexicographic_rates[0] == max_min_rates[0]
        assert lexicographic_rates > max_min_rates
        assert abs(sum(lexicographic['shares'].values()) - 1.) < 1e-6
