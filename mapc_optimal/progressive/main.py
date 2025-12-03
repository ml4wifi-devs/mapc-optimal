import pulp as plp


class Main:
    def __init__(self, solver: plp.LpSolver, M: float) -> None:
        self.solver = solver
        self.M = M

    def __call__(
            self,
            stations: list,
            selected_stations: list,
            link_node_b: dict,
            conf_links: dict,
            conf_link_rates: dict,
            confs: list,
            sigma: dict,
            rho: dict,
            l: float
    ) -> tuple[dict, float]:
        main = plp.LpProblem('main', plp.LpMaximize)

        conf_weight = plp.LpVariable.dicts('conf_weight', confs, lowBound=0, upBound=1, cat=plp.LpContinuous)
        node_throughput = plp.LpVariable.dicts('node_throughput', stations, lowBound=0, cat=plp.LpContinuous)
        min_throughput = plp.LpVariable('min_throughput', lowBound=0, cat=plp.LpContinuous)
        y = plp.LpVariable('y', lowBound=0, cat=plp.LpContinuous)

        # fraction of time when we use the configuration, the sum of all the weights is 1
        main += plp.lpSum(conf_weight[c] for c in confs) == 1 - l, 'conf_weight_c'  # dual: alpha

        for s in stations:
            # calculation of the summarized throughput of each station (over all the used configurations)
            main += node_throughput[s] == plp.lpSum(
                conf_link_rates[c][l] * conf_weight[c] for c in confs for l in conf_links[c] if link_node_b[l] == s
            ), f'node_throughput_{s}_c'  # dual: beta

            # calculation of the worst throughput (this constraint is especially necessary for max-min optimization)
            main += node_throughput[s] >= sigma[s] - y, f'sigma_{s}_c'  # dual: gamma
            main += node_throughput[s] >= rho[s] - y, f'rho_{s}_c'  # dual: delta
        
        for s in selected_stations:
            main += node_throughput[s] >= min_throughput, f'worst_throughput_{s}_c'

        main += min_throughput - self.M * y, 'min_throughput_baseline_g'
        
        main.conf_weight = conf_weight

        main.solve(self.solver)

        if main.status != plp.LpStatusOptimal:
            raise Exception('Main problem not solved optimally')

        result = {
            'alpha': main.constraints['conf_weight_c'].pi,
            'beta': {s: main.constraints[f'node_throughput_{s}_c'].pi for s in stations},
            'gamma': {s: main.constraints[f'sigma_{s}_c'].pi for s in stations},
            'delta': {s: main.constraints[f'rho_{s}_c'].pi for s in stations},
            'shares': {c: main.conf_weight[c].varValue for c in confs}
        }

        return result, plp.value(main.objective)
