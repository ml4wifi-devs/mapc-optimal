import random
from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from mapc_optimal.pricing import Pricing
from mapc_optimal.utils import OptimizationType


class TabuPricing(Pricing):
    r"""
    The pricing problem solved with heuristic tabu search instead of the MILP model. The configurations
    are evaluated either with the model of the pricing problem, i.e., with the same channel model and
    SINR thresholds as in :class:`mapc_optimal.pricing.Pricing`, or with an evaluator given by the user,
    e.g., a network simulator.
    """

    def __init__(
            self,
            n_steps: int = 500,
            n_candidates: int = 25,
            tabu_size: int = 80,
            n_restarts: int = 4,
            tx_power_levels: int = 6,
            seed: int = 42,
            evaluator: Callable[[list], list] = None,
            n_draws: int = 8,
            **kwargs
    ) -> None:
        r"""
        Parameters
        ----------
        n_steps : int, default=500
            The number of tabu search steps in a single run.
        n_candidates : int, default=25
            The number of neighbors generated in each step.
        tabu_size : int, default=80
            The number of the recently visited configurations which cannot be revisited.
        n_restarts : int, default=4
            The number of runs of the search, each starting from a random configuration. The best
            configuration found in all the runs is returned.
        tx_power_levels : int, default=6
            The number of transmission power levels, equally spaced in the logarithmic
            scale between the minimum and the maximum transmission power.
        seed : int, default=42
            The seed of the random number generator.
        evaluator : Callable, default=None
            Function ``evaluator(confs) -> rates`` evaluating all the candidates of a search step at
            once, where ``confs`` is a list of dictionaries mapping a transmitting link to a tuple with
            its transmission power (in the linear scale) and its MCS, and ``rates`` is a list of
            dictionaries mapping the links to their rates (in Mb/s), with the links which cannot
            transmit omitted or given a zero rate. The whole batch is passed in one call, so the function
            can be vectorized. If it is not set, the rates are calculated with the model of the pricing problem.
        n_draws : int, default=8
            The number of calls to the evaluator whose rates are averaged. For a deterministic evaluator,
            it should be set to 1.
        **kwargs
            The arguments of :class:`mapc_optimal.pricing.Pricing`.
        """

        super().__init__(**kwargs)

        self.n_steps = n_steps
        self.n_candidates = n_candidates
        self.tabu_size = tabu_size
        self.n_restarts = n_restarts
        self.mcs_data_rates = np.asarray(self.mcs_data_rates)
        self.tx_power_set = tuple(np.geomspace(self.min_tx_power, self.max_tx_power, tx_power_levels).tolist())
        self.evaluator = evaluator
        self.n_draws = n_draws
        self.rng = random.Random(seed)

    def initial_configuration(self, links: list, link_path_loss: dict, configurations: list = None) -> dict:
        """
        Generates the initial configurations as :class:`mapc_optimal.pricing.Pricing` does, but, if the
        evaluator is set, draws the MCS of the links at random and takes their rates from the evaluator,
        so that all the configurations passed to the main problem are evaluated in the same way.
        """

        configuration = super().initial_configuration(links, link_path_loss, configurations)

        if self.evaluator is None:
            return configuration

        confs = [
{
                l: (configuration['conf_link_tx_power'][c][l], self.rng.randrange(len(self.mcs_values)))
                for l in configuration['conf_links'][c]
            }
            for c in configuration['confs']
        ]

        for c, conf, rates in zip(configuration['confs'], confs, self._mean_rates(confs)):
            rates = {l: float(r) for l, r in rates.items() if r > 0.}
            configuration['conf_links'][c] = list(rates)
            configuration['conf_link_rates'][c] = rates
            configuration['conf_link_mcs'][c] = {l: conf[l][1] for l in rates}
            configuration['conf_link_tx_power'][c] = {l: conf[l][0] for l in rates}
            configuration['conf_total_rates'][c] = sum(rates.values())

        return configuration

    def _mean_rates(self, confs: list) -> list:
        """
        Calls the evaluator `n_draws` times and returns the mean rates of the links of each configuration.
        """

        rates = [{} for _ in confs]

        for _ in range(self.n_draws):
            for c, conf_rates in enumerate(self.evaluator(confs)):
                for l, r in conf_rates.items():
                    rates[c][l] = rates[c].get(l, 0.) + r / self.n_draws

        return rates

    def _setup(self, links: list, link_node_a: dict, link_node_b: dict, link_path_loss: dict, dual_beta: dict) -> None:
        """
        Precomputes the arrays used to evaluate all the candidates of a step at once: the gains of the
        interfering links, the gains of the transmitting links, and the weight of each link in the
        pricing objective function.
        """

        self.link_index = {l: i for i, l in enumerate(links)}
        self.ap_links = {a: [l for l in links if link_node_a[l] == a] for a in sorted(set(link_node_a.values()))}
        self.gains = np.array([
            [0. if link_node_a[j] == link_node_a[i] else 1. / link_path_loss[link_node_a[j], link_node_b[i]] for j in links]
            for i in links
        ])
        self.direct_gains = np.array([1. / link_path_loss[l] for l in links])

        if self.opt_type == OptimizationType.SUM:
            weights = [1. + dual_beta[link_node_b[l]] for l in links]
        elif self.opt_type in (OptimizationType.MAX_MIN, OptimizationType.MAX_MIN_BASELINE, OptimizationType.LEXICOGRAPHIC):
            weights = [dual_beta[link_node_b[l]] for l in links]
        elif self.opt_type == OptimizationType.PROPORTIONAL:
            weights = [sum(a * dual_beta[link_node_b[l], k] for k, a in enumerate(self.log_approx[0])) for l in links]
        else:
            raise ValueError('Invalid optimization type')

        self.weights = np.array(weights)

    def _random_settings(self) -> tuple:
        """
        Draws a random transmission power and a random MCS of a link.
        """

        return self.rng.choice(self.tx_power_set), self.rng.randrange(len(self.mcs_values))

    def _random_conf(self) -> dict:
        """
        Draws a random configuration: a random subset of the APs transmits, each to a random station,
        with random settings. At least one link is always active.
        """

        aps = self.rng.sample(list(self.ap_links), self.rng.randint(1, len(self.ap_links)))
        return {self.rng.choice(self.ap_links[ap]): self._random_settings() for ap in aps}

    def _change_link(self, conf: dict) -> dict:
        """
        Changes the link of a randomly selected AP: the AP starts transmitting if it was silent,
        and otherwise it stops transmitting or switches to another station.
        """

        ap_links = self.ap_links[self.rng.choice(list(self.ap_links))]
        active = next((l for l in conf if l in ap_links), None)
        other = [l for l in ap_links if l != active]

        if active is None:
            conf[self.rng.choice(ap_links)] = self._random_settings()
        elif self.rng.random() < 0.5:
            if len(conf) > 1:                   # at least one link is always active
                del conf[active]
        elif other:
            conf[self.rng.choice(other)] = conf.pop(active)

        return conf

    def _change_tx_power(self, conf: dict) -> dict:
        """
        Changes the transmission power of a randomly selected transmitting link to another level.
        """

        link = self.rng.choice(list(conf))
        tx_power, mcs = conf[link]
        conf[link] = (self.rng.choice([p for p in self.tx_power_set if p != tx_power]), mcs)

        return conf

    def _change_mcs(self, conf: dict) -> dict:
        """
        Changes the MCS of a randomly selected transmitting link to another value.
        """

        link = self.rng.choice(list(conf))
        tx_power, mcs = conf[link]
        conf[link] = (tx_power, self.rng.choice([m for m in range(len(self.mcs_values)) if m != mcs]))

        return conf

    def _neighbor(self, conf: dict) -> dict:
        """
        Generates a neighbor of a configuration by applying one of the three changes, drawn at random.
        Each AP transmits on at most one link and at least one link is always active, so a change which
        would break these rules leaves the configuration as it is; such a neighbor is tabu, because the
        current configuration is always in the tabu list.
        """

        change = self.rng.choice([self._change_link, self._change_tx_power, self._change_mcs])
        return change(dict(conf))

    def _rates(self, confs: list) -> NDArray:
        """
        Returns the rates of the links of each configuration in a batch, as a matrix with one row per
        configuration and one column per link. The rates come from the evaluator if it is set, and from
        the model of the pricing problem otherwise.
        """

        rates = np.zeros((len(confs), len(self.link_index)))

        if self.evaluator:
            for c, conf_rates in enumerate(self._mean_rates(confs)):
                for l, r in conf_rates.items():
                    rates[c, self.link_index[l]] = r

            return rates

        tx_power = np.zeros_like(rates)
        mcs = np.zeros_like(rates, dtype=int)

        for c, conf in enumerate(confs):
            for l, (p, m) in conf.items():
                tx_power[c, self.link_index[l]] = p
                mcs[c, self.link_index[l]] = m

        interference = tx_power @ self.gains.T + self.noise_floor
        feasible = (tx_power > 0.) & (tx_power * self.direct_gains >= self.min_sinr[mcs] * interference)

        return np.where(feasible, self.mcs_data_rates[mcs], 0.)

    def _evaluate(self, confs: list, dual_alpha: float) -> NDArray:
        """
        Evaluates a batch of configurations at once and returns the values of the pricing objective
        function, i.e., their reduced costs.
        """

        return self._rates(confs) @ self.weights - dual_alpha

    def _search(self, dual_alpha: float) -> dict:
        """
        Runs tabu search and returns the best configuration found. Every run of the search starts from
        a random configuration, and the best configuration found in all the runs is returned.
        """

        best_conf, best_objective = {}, -np.inf

        for _ in range(self.n_restarts):
            conf = self._random_conf()
            objectives = self._evaluate([conf], dual_alpha)
            tabu = deque([frozenset(conf.items())], maxlen=self.tabu_size)

            if objectives[0] > best_objective:
                best_conf, best_objective = conf, objectives[0]

            for _ in range(self.n_steps):
                neighbors = [self._neighbor(conf) for _ in range(self.n_candidates)]
                objectives = self._evaluate(neighbors, dual_alpha)

                # a tabu neighbor is selected only if it is better than the best configuration found
                # so far; the search moves in every step, even when all the neighbors are worse
                scores = [
                    o if frozenset(c.items()) not in tabu or o > best_objective else -np.inf
                    for o, c in zip(objectives, neighbors)
                ]
                best_idx = max(range(len(neighbors)), key=lambda i: scores[i])
                conf = neighbors[best_idx]
                tabu.append(frozenset(conf.items()))

                if objectives[best_idx] > best_objective:
                    best_conf, best_objective = conf, objectives[best_idx]

        return best_conf

    def __call__(
            self,
            dual_alpha: float,
            dual_beta: dict,
            stations: list,
            access_points: list,
            links: list,
            link_node_a: dict,
            link_node_b: dict,
            link_path_loss: dict,
            max_interference: dict,
            configuration: dict
    ) -> tuple[dict, float]:
        """
        Searches for a configuration which improves the solution of the main problem. The signature is
        the same as in :class:`mapc_optimal.pricing.Pricing`.
        """

        self._setup(links, link_node_a, link_node_b, link_path_loss, dual_beta)
        best_conf = self._search(dual_alpha)

        # a stochastic evaluator overestimates the best configuration, because it is selected as the
        # maximum of many noisy evaluations, so it is evaluated once more before it is returned
        rates = self._rates([best_conf])
        best_rates = {l: float(rates[0, i]) for l, i in self.link_index.items() if rates[0, i] > 0.}

        conf_num = configuration['conf_num']
        configuration['confs'] = range(1, conf_num + 1)
        configuration['conf_links'][conf_num] = list(best_rates)
        configuration['conf_link_rates'][conf_num] = best_rates
        configuration['conf_link_mcs'][conf_num] = {l: best_conf[l][1] for l in best_rates}
        configuration['conf_link_tx_power'][conf_num] = {l: best_conf[l][0] for l in best_rates}
        configuration['conf_total_rates'][conf_num] = sum(best_rates.values())
        configuration['conf_num'] += 1

        return configuration, float(rates[0] @ self.weights - dual_alpha)
