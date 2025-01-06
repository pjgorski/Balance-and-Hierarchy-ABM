"""LTD experiment specific for complete networks"""
import csv
import sys
from datetime import datetime
from functools import reduce
import random
from itertools import combinations, permutations

import git
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from experiments.decorator import register_experiment
from experiments.experiment import Experiment


@register_experiment
class LtdComplete(Experiment):
    def __init__(self, args):
        # Basic settlements
        super().__init__(args)
        stamp = str(datetime.now()).replace(' ', '_').replace('.', '_').replace(":", "-")
        self.output_path = Path('./outputs/') / self.__class__.__name__ / self.args.exp_name / stamp
        self.output_fname = 'outputs.tsv'
        self.positive_k_fname = 'pkdensities.csv'
        self.connections = np.zeros((self.args.n_agents, self.args.n_agents))
        self.is_directed = self.args.is_directed

        # Dynamics type
        if self.args.agent_based and self.args.is_directed:
            self.step = self.step_agent_oriented
        else:
            if self.args.agent_based:
                raise Warning("Building triad oriented dynamics as it is not")
            self.step = self.step_triad_oriented

        # Masks setting
        if self.is_directed:
            # All elements except from diagonal
            self.mask = np.where(~np.eye(self.args.n_agents, dtype=bool))
        else:
            # All elements from upper triangle
            self.mask = np.mask_indices(self.args.n_agents, np.triu, k=1)

        # Remaining utilities
        self.links_num = len(self.mask[0])
        self.initialize_output_file()

    @classmethod
    def add_args(cls, subparsers):
        """Add simulation arguments"""
        exp_parser = subparsers.add_parser(cls.name(), help='Optimized LTD simulation for complete network')
        exp_parser.add_argument('exp_name', help="Name identifying the experiment run")
        exp_parser.add_argument('-n', '--n-agents', type=int, default=100, help='Number of agents')
        exp_parser.add_argument('-p', '--probability', type=float, nargs='*', default=None, required=False,
                                help='Probability p - parameter of LTD dynamics')
        exp_parser.add_argument('-s', '--steps', type=int, default=1000,
                                help='Number of steps of simulation, each step'
                                     ' is number of updates equal to n-agents')
        exp_parser.add_argument('-r', '--repetitions', type=int, default=1)
        exp_parser.add_argument('--is-directed', action="store_true")
        exp_parser.add_argument('--agent-based', action="store_true")
        exp_parser.add_argument('--rho-init', type=float, nargs = '*', default=[0.5], required=False,
                                help='Initial density of positive links')
        exp_parser.add_argument('--no-triad-stats', action="store_true")
        

    def __call__(self, *args, **kwargs):
        assert self.args.probability is not None, '--probability (-p) argument is required'
        self.randomize_connections(0.5)
        # probs_arr = [prob for prob in self.args.probability for ii in range(self.args.repetitions)]
        # results = pd.DataFrame(columns=["n", "p", "n0" "n1", "n2", "n3", "positive"])

        for p in self.args.probability:
            for rho_init in self.args.rho_init:
                for _ in range(self.args.repetitions):
                    """Initialize output data frame"""
                    # output = pd.DataFrame({'n0': [], 'n1': [], 'n2': [], 'n3': [], 'rho': []})
                    output_columns = []
                    if not self.args.no_triad_stats:
                        if self.is_directed:
                            output_columns = [f'tr{i}' for i in range(8)]
                        else:
                            output_columns = [f'n{i}' for i in range(3)]
                    output_columns.append('rho')
                    output = {}
                    for column in output_columns:
                        output[column] = []
                    output = pd.DataFrame(output)

                    """Initialize simulation"""
                    self.randomize_connections(rho_init)
                    updates_num = self.links_num * self.args.steps

                    """Run simulation"""
                    for i in tqdm(range(updates_num)):
                        self.step(p)
                        if not i % self.links_num:
                            stats = self.get_antal_statistics()
                            output = output.append(stats, ignore_index=True)
                            with open(self.output_path / self.positive_k_fname, 'a') as pkfile:
                                csv_writer = csv.writer(pkfile, delimiter=',')
                                csv_writer.writerow(self.get_positive_in_degree_dist())
                            if self.links_num == stats['rho']:
                                """Paradise check"""
                                print("Paradise reached...")
                                for _ in range(int(i / self.links_num) + 1, self.args.steps):
                                    output = output.append(stats, ignore_index=True)
                                break

                    """Save results"""
                    with open(self.output_path / self.output_fname, 'a') as file:
                        results = []
                        for column in output.columns:
                            results.append(reduce((lambda x, y: str(x) + ',' + str(y)), output[column].tolist()))
                        results_string = "\t".join([f"{result}" for result in results])
                        file.write(
                            f"{p}\t{rho_init}\t{results_string}\t"
                            f"{self.get_balanced_pairs_count()}\n"
                        )

    def get_antal_statistics(self):
        """
        If flag --no-triad-stats was set, no triad statistics is returned, only number of positive links. Otherwise:
        If is not directed it returns dict with positive links number and number of triads n_k where k stands for
        number of negative links in a triad, otherwise it returns number of each triad considered as following:\
        The triad consists of connections between agents A, B and C, connected in a specific way:\
        AB: A->B
        AC: A->C
        BC: B->C,
        then the naming of each triad goes:

        name        AB      AC    BC
        -----------------------------------
        tr_0        -1      -1      -1
        tr_1        -1      -1      +1
        tr_2        -1      +1      -1
        tr_3        -1      +1      +1
        tr_4        +1      -1      -1
        tr_5        +1      -1      +1
        tr_6        +1      +1      -1
        tr_7        +1      +1      +1

        returns columns in order: \n
        i)  n0, n1, n2, n3, rho \n
        ii) tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, rho
        """
        positive_links = np.sum(self.connections[self.mask] > 0)
        if self.args.no_triad_stats:
            if not self.is_directed:
                return {'rho': int(positive_links/2)}
            else:
                return {'rho': int(positive_links)}
        if not self.is_directed:
            n_arr = [0] * 4
            for triad in self.get_all_triads():
                # n_arr[len(np.where(self.connections[triad] < 0)[0])] += 1
                n_arr[np.sum(self.connections[triad] < 0)] += 1
            return {'n0': int(n_arr[0]), 'n1': int(n_arr[1]), 'n2': int(n_arr[2]), 'n3': int(n_arr[3]),
                    'rho': int(positive_links/2)}
        else:
            n_arr = [0] * 8
            for triad in self.get_all_triads():
                """
                Variable *triad* relates to: L1: AB, L2: BC, L3: AC, 
                so to achieve specific index consider -1 as 0 and compute the index of triad type
                as it was binary: i = AB*2^2 + AC*2^1 + BC*2^0
                """
                connections_values = [0 if cv < 0 else 1 for cv in self.connections[triad]]
                n_arr[connections_values[0]*4 + connections_values[2]*2 + connections_values[1]] += 1
            """Build return dict"""
            return_dict = {}
            for i in range(8):
                return_dict[f'tr{i}'] = n_arr[i]
            return_dict['rho'] = positive_links
            return return_dict

    def step_triad_oriented(self, p):
        x_triad, y_triad = self.random_triad()
        values = self.connections[x_triad, y_triad]
        balanced = True if reduce(lambda x, y: x * y, values) > 0 else False
        if balanced:
            return
        else:
            num_of_negatives = len([v for v in values if v < 0])
            if num_of_negatives != 1:
                to_change = np.random.choice([0, 1, 2])
            else:
                negatives = np.where(values < 0)[0]
                if random.random() < p:
                    # Change negative link to positive
                    to_change = negatives[0]
                else:
                    # Change positive link to negative
                    positives = np.where(values > 0)[0]
                    to_change = np.random.choice(positives)
            self.connections[x_triad[to_change],
                             y_triad[to_change]] *= -1

    def step_agent_oriented(self, p):
        # Returns "coordinates" of links in connection matrix
        x_triad, y_triad = self.random_triad()
        # Extracts values of the links
        values = self.connections[x_triad, y_triad]
        balanced = True if reduce(lambda x, y: x * y, values) > 0 else False
        if balanced:
            return
        else:
            if values[0] * values[2] > 0:
                # a1->a2 and a1->a3 are both positive or negative
                to_change = np.random.choice([0, 2])
            else:
                probs = [p, 1 - p] if values[0] < values[2] else [1 - p, p]
                to_change = np.random.choice([0, 2], p=probs)
        self.connections[x_triad[to_change],
                         y_triad[to_change]] *= -1

    def randomize_connections(self, rho_init):
        if rho_init == 0.5:
            self.connections[self.mask] = np.random.choice([-1, 1], self.links_num)
        else:
            neg_dens = 1. -rho_init
            self.connections[self.mask] = np.random.choice([-1, 1], self.links_num, p = [neg_dens, rho_init])

    def random_element(self):
        x, y = self.mask
        return np.random.choice(x), np.random.choice(y)

    def get_all_triads(self):
        """Generates all triads in the network"""
        if self.is_directed:
            comb_generator = (x for c in combinations(range(self.args.n_agents), 3) for x in permutations(c))
        else:
            comb_generator = combinations(range(self.args.n_agents), 3)
        for a in comb_generator:
            a1, a2, a3 = a
            yield self.get_specific_triad(a1, a2, a3)

    def random_triad(self):
        a1, a2, a3 = np.random.choice(self.args.n_agents, 3, replace=False)
        return self.get_specific_triad(a1, a2, a3)

    def get_specific_triad(self, a1, a2, a3):
        """
        L1: a1->a2 ( x = a1, y = a2)
        L2: a2->a3 ( x = a2, y = a3)
        L3: a1->a3 ( x = a1, y = a3)
        """
        if not self.is_directed:
            a1, a2, a3 = sorted([a1, a2, a3])
        return np.array([a1, a2, a1]), np.array([a2, a3, a3])

    def get_balanced_pairs_count(self):
        if self.is_directed:
            balance_m = self.connections * self.connections.T
            return np.where(np.triu(balance_m, 1) > 0)[0].size
        else:
            return self.links_num

    def initialize_output_file(self):
        self.output_path.mkdir(parents=True)
        with open(self.output_path / self.output_fname, 'w') as file:
            if not self.args.no_triad_stats:
                file.write(f"# Arguments: {str(sys.argv)}\n"
                        "# p - p probability\n"
                        "# nk - number of nk triads\n"
                        "# rho - number of positive triads\n"
                        "# bp - number of balanced pairs\n")
            else:
                file.write(f"# Arguments: {str(sys.argv)}\n"
                        "# p - p probability\n"
                        "# rho - number of positive triads\n"
                        "# bp - number of balanced pairs\n")
            if not self.is_directed:
                file.write("p\trho_init\tn0\tn1\tn2\tn3\trho\tbp\n")
            else:
                triad_header = "\t".join(["tr"+str(i) for i in range(8)])
                file.write("p\trho_init\t"+triad_header+"\trho\tbp\n")

    def get_positive_in_degree_dist(self):
        w = np.argwhere(self.connections > 0)
        return [len(np.argwhere(w[:, 1] == i)) for i in range(self.args.n_agents)]


@register_experiment
class LtdStatus(LtdComplete):
    def __init__(self, args):
        Experiment.__init__(self, args)
        stamp = str(datetime.now()).replace(' ', '_').replace('.', '_').replace(":", "-")
        self.output_path = Path('./outputs/') / self.__class__.__name__ / self.args.exp_name / stamp
        self.output_fname = 'outputs.tsv'
        self.positive_k_fname = 'pkdensities.csv'
        self.connections = np.zeros((self.args.n_agents, self.args.n_agents))
        self.mask = np.where(~np.eye(self.args.n_agents, dtype=bool))
        self.is_directed = True
        self.links_num = len(self.mask[0])

        if self.args.ltd_agent_based:
            self.ltd_step = self.step_agent_oriented
        else:
            self.ltd_step = self.step_triad_oriented

        if self.args.on_triad_status:
            self.status_imbalanced_triads = np.array([[-1, -1, 1], [1, 1, -1]])
            self.status_step = self.status_step_triad_balance
        else:
            self.status_step = self.status_step_pair_balance

        self.initialize_output_file()

    def __call__(self, *args, **kwargs):
        assert self.args.probability is not None, '--probability (-p) argument is required'
        self.randomize_connections(0.5)
        # probs_arr = [prob for prob in self.args.probability for ii in range(self.args.repetitions)]

        for q in self.args.q:
            for p in self.args.probability:
                for ps in self.args.psprob:
                    for rho_init in self.args.rho_init:
                        for _ in range(self.args.repetitions):
                            """Initialize output data frame"""
                            output_columns = []
                            if not self.args.no_triad_stats:
                                output_columns = [f'tr{i}' for i in range(8)]
                            output_columns.append('rho')
                            output = {}
                            for column in output_columns:
                                output[column] = []
                            output = pd.DataFrame(output)

                            """Initialize connections and parameters"""
                            self.randomize_connections(rho_init)
                            updates_num = self.links_num * self.args.steps
                            _ps = ps if ps >= 0 else p  # ps equals to heider prob

                            """Run simulation"""
                            for i in tqdm(range(updates_num)):
                                self.step(p, q, _ps)
                                if not i % self.links_num:
                                    stats = self.get_antal_statistics()
                                    output = output.append(stats, ignore_index=True)
                                    if self.args.save_pk_distribution:
                                        with open(self.output_path / (f'p{p}q{q}ps{ps}'.replace('.', '-')
                                                                    + self.positive_k_fname), 'a') as pkfile:
                                            csv_writer = csv.writer(pkfile, delimiter=',')
                                            csv_writer.writerow(self.get_positive_in_degree_dist())
                                    if self.links_num == stats['rho']:
                                        """Paradise check"""
                                        print("Paradise reached...")
                                        for _ in range(int(i / self.links_num) + 1, self.args.steps):
                                            output = output.append(stats, ignore_index=True)
                                        break

                            """Save results"""
                            with open(self.output_path / self.output_fname, 'a') as file:
                                results = []
                                for column in output.columns:
                                    # Join array by ',' as string
                                    results.append(reduce(lambda x, y: str(x) + ',' + str(y), output[column].tolist()))
                                results_string = "\t".join([f"{result}" for result in results])
                                file.write(
                                    f"{q}\t{p}\t{ps}\t{rho_init}\t{results_string}\t"
                                    f"{self.get_balanced_pairs_count()}\n"
                                )

    @classmethod
    def add_args(cls, subparsers):
        """Add simulation arguments"""
        exp_parser = subparsers.add_parser(cls.name(), help='Optimized LTD simulation for complete network')
        exp_parser.add_argument('exp_name', help="Name identifying the experiment run")
        exp_parser.add_argument('-n', '--n-agents', type=int, default=100, help='Number of agents')
        exp_parser.add_argument('-p', '--probability', type=float, nargs='*', default=None, required=False,
                                help='Probability p - parameter of LTD dynamics')
        exp_parser.add_argument('-q', type=float, nargs='*', default=[0.5], help='Probability of status dynamics'
                                                                                 ' against LTD dynamics')
        exp_parser.add_argument('-ps', '--psprob', type=float, nargs='*', default=[0.5],
                                help='Only if on-triad-status is set; probability of choosing positive link rather than'
                                     ' negative')
        exp_parser.add_argument('-s', '--steps', type=int, default=1000,
                                help='Number of steps of simulation, each step'
                                     ' is number of updates equal to n-agents')
        exp_parser.add_argument('-r', '--repetitions', type=int, default=1)
        exp_parser.add_argument('--ltd-agent-based', action='store_true')
        exp_parser.add_argument('--on-triad-status', action='store_true')
        exp_parser.add_argument('--save-pk-distribution', action='store_true')
        exp_parser.add_argument('--rho-init', type=float, nargs = '*', default=[0.5], required=False,
                                help='Initial density of positive links')
        exp_parser.add_argument('--no-triad-stats', action="store_true")

    def step(self, p, q, ps):
        if random.random() > q:
            self.ltd_step(p)
        else:
            self.status_step(ps)

    def status_step_pair_balance(self, ps):
        a1, a2 = np.random.choice(range(self.args.n_agents), 2, replace=False)
        if self.connections[a1, a2] * self.connections[a2, a1] < 0:
            return
        probs = [ps, 1 - ps] if self.connections[a1, a2] < self.connections[a2, a1] else [1 - ps, ps]
        choices = [(a1, a2), (a2, a1)]
        choice = np.random.choice([0,1], p=probs)
        a1, a2 = choices[choice]
        self.connections[a1, a2] *= -1

    def status_step_triad_balance(self, ps):
        """
        Imbalanced triads:\
        a)  a1->a2 = -1\
            a2->a3 = -1\
            a1->a3 = +1\

        b)  a1->a2 = +1\
            a2->a3 = +1\
            a1->a3 = -1\
        Any other triad is balanced according to statuses in it
        """
        # Returns "coordinates" of links in connection matrix
        x_triad, y_triad = self.random_triad()
        # Extracts values of the links
        values = self.connections[x_triad, y_triad]
        balanced = not any([all(values == imbalanced_triad) for imbalanced_triad in self.status_imbalanced_triads])
        if balanced:
            return
        else:
            probs = [ps, 1 - ps] if values[0] < values[2] else [1 - ps, ps]
            to_change = np.random.choice([0, 2], p=probs)
            self.connections[x_triad[to_change], y_triad[to_change]] *= -1

    def initialize_output_file(self):
        self.output_path.mkdir(parents=True)
        commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
        with open(self.output_path / self.output_fname, 'w') as file:
            if not self.args.no_triad_stats:
                triad_header = "\t".join(["tr" + str(i) for i in range(8)])
                file.write(f"# Commit: {commit_hash}"
                        f"# Arguments: {str(self.args)}\n"
                        "# q - q probability\n"
                        "# p - p probability\n"
                        "# ps - probability of positive link in status update\n"
                        "# nk - number of nk triads\n"
                        "# rho - number of positive triads\n"
                        "# bp - number of balanced pairs\n")
                file.write(f"q\tp\tps\trho_init\t" + triad_header + "\trho\tbp\n")
            else:
                file.write(f"# Commit: {commit_hash}"
                        f"# Arguments: {str(self.args)}\n"
                        "# q - q probability\n"
                        "# p - p probability\n"
                        "# ps - probability of positive link in status update\n"
                        "# rho - number of positive triads\n"
                        "# bp - number of balanced pairs\n")
                file.write(f"q\tp\tps\trho_init\t" + "rho\tbp\n")
