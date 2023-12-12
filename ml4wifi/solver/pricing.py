import pulp as plp


class Pricing:
    def __init__(
            self,
            mcs_values: list,
            mcs_data_rates: list,
            min_snr: list,
            max_tx_power: float,
            noise_floor: float,
            opt_sum: bool
    ) -> None:
        self.mcs_values = mcs_values
        self.min_snr_diff = {m: min_snr[0] if m == 0 else (min_snr[m] - min_snr[m - 1]) for m in mcs_values}
        self.mcs_rate_diff = {m: mcs_data_rates[0] if m == 0 else (mcs_data_rates[m] - mcs_data_rates[m - 1]) for m in mcs_values}
        self.max_tx_power = max_tx_power
        self.noise_floor = noise_floor
        self.opt_sum = opt_sum

    def __call__(
            self,
            dual_alpha: float,
            dual_beta: dict,
            dual_gamma: dict,
            stations: list,
            access_points: list,
            links: list,
            link_node_a: dict,
            link_node_b: dict,
            link_path_loss: dict,
            configuration: dict
    ) -> tuple:

        # Definition of the pricing problem
        pricing = plp.LpProblem('pricing', plp.LpMaximize)

        # Variables:
        # which station is on (binary)
        station_on = plp.LpVariable.dicts('station_on', stations, cat=plp.LpBinary)

        # which link is on (binary)
        link_on = plp.LpVariable.dicts('link_on', links, cat=plp.LpBinary)

        # MCS mode used in link (binary)
        link_mcs = plp.LpVariable.dicts('link_mcs', [(e, m) for e in links for m in self.mcs_values], cat=plp.LpBinary)

        # data rate obtained in link
        link_data_rate = plp.LpVariable.dicts('link_data_rate', links, lowBound=0, cat=plp.LpContinuous)

        # transmission power in link
        link_tx_power = plp.LpVariable.dicts('link_tx_power', links, lowBound=0, cat=plp.LpContinuous)

        # signal to interference plus noise ratio (SINR) in link
        link_sinr = plp.LpVariable.dicts('link_sinr', links, lowBound=0, cat=plp.LpContinuous)

        # noise influence of one transmitting station to another
        interference = plp.LpVariable.dicts('interference', [(e, s) for e in links for s in stations], lowBound=0, cat=plp.LpContinuous)

        # Constraints:
        for s in stations:
            # station receives transmission (on one of its links) if this station is on
            pricing += plp.lpSum(link_on[l] for l in links if link_node_b[l] == s) == station_on[s], f'station_on_{s}_c'

        for a in access_points:
            # AP can simultaneously transmit to at most one station on all of its links
            pricing += plp.lpSum(link_on[l] for l in links if link_node_a[l] == a) <= 1, f'ap_on_{a}_c'

        for l in links:
            # if link is on, then node can transmit with its maximum power (but, less is also possible)
            pricing += self.max_tx_power * link_on[l] >= link_tx_power[l], f'link_tx_power_{l}_c'

            # the way transmission modes are switched on in link (incremental switching-on)
            for m in self.mcs_values:
                if m == 0:
                    pricing += link_mcs[l, 0] <= link_on[l], f'link_mcs_{l}_{m}_c'
                else:
                    pricing += link_mcs[l, m] <= link_mcs[l, m - 1], f'link_mcs_{l}_{m}_c'

            # interference from other concurrent transmissions
            for s in stations:
                if s != link_node_b[l]:
                    pricing += interference[l, s] >= link_tx_power[l] - link_path_loss[l], f'interference_{l}_{s}_c'

            # SINR in link
            pricing += link_sinr[l] >= (
                    link_tx_power[l]
                    - link_path_loss[l]
                    - plp.lpSum(interference[l, s] for s in stations if s != link_node_b[l])
                    - self.noise_floor
            ), f'link_sinr_{l}_c'

            # MCS selection in link (on the basis of the SINR)
            pricing += link_sinr[l] >= plp.lpSum(self.min_snr_diff[m] * link_mcs[l, m] for m in self.mcs_values), f'link_mcs_{l}_c'

            # data rate obtained in link (on the basis of the switched-on MCS modes)
            pricing += link_data_rate[l] == plp.lpSum(self.mcs_rate_diff[m] * link_mcs[l, m] for m in self.mcs_values), f'link_data_rate_{l}_c'

        # Goal for the pricing problem
        if self.opt_sum:
            pricing += (
                plp.lpSum(link_data_rate[e] for e in links)
                - dual_alpha
                + plp.lpSum(dual_beta[v] * link_data_rate[e] for v in stations for e in links if link_node_b[e] == v)
            ), 'tx_set_throughput_g'
        else:
            pricing += (
                - dual_alpha
                + plp.lpSum(dual_beta[v] * link_data_rate[e] for v in stations for e in links if link_node_b[e] == v)
            ), 'tx_set_throughput_g'

        # To access the variables from outside:
        pricing.link_on = link_on
        pricing.link_data_rate = link_data_rate
        pricing.link_tx_power = link_tx_power

        # Solve the pricing problem
        pricing.solve()

        # Check the status of the optimization problem
        if pricing.status != plp.LpStatusOptimal:
            raise Exception('Pricing problem not solved optimally')

        # Generation of a new compatible set
        conf_num = configuration['conf_num']
        configuration['confs'] = range(1, conf_num + 1)

        # links used in the new compatible set
        configuration['conf_links'][conf_num] = [e for e in links if pricing.link_on[e].varValue == 1]

        # link rates for the new compatible set
        for l in links:
            if pricing.link_on[l].varValue == 1:
                configuration['conf_link_rates'][conf_num, l] = pricing.link_data_rate[l].varValue
            else:
                configuration['conf_link_rates'][conf_num, l] = 0

            configuration['conf_link_tx_power'][conf_num, l] = pricing.link_tx_power[l].varValue

        # station rates for compatible sets
        for s in stations:
            configuration['conf_sta_rates'][conf_num, s] = sum(
                configuration['conf_link_rates'][conf_num, e] for e in configuration['conf_links'][conf_num] if link_node_b[e] == s
            )

        # total rates for compatible sets
        configuration['conf_total_rates'][conf_num] = sum(configuration['conf_sta_rates'][conf_num, v] for v in stations)

        # and for the next interation...
        configuration['conf_num'] += 1

        return configuration, plp.value(pricing.objective)
