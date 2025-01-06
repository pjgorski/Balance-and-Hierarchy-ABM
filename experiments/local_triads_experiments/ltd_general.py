import logging
import sys
import warnings
from datetime import datetime
from functools import reduce

from pathlib import Path
import random

import git
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.decorator import register_experiment
from experiments.experiment import Experiment
from networktools.network_generators import generator

ADM_SUBDIR_NAME = 'adjacency_matrices'


def build_triad(a, b, c):
    """
    L1: a->b
    L2: b->c
    L3: a->c
    """
    return [a, b, a], [b, c, c]


def find_common_neighbours(adjacency_matrix: np.matrix, a: int, b: int):
    """
    Take adjacency matrix as an argument and finds common neighbours of two agents _a_ and _b_;
    adjacency_matrix[i, j] stands for connection i->j. If no common neighbours returns None.
    """
    common_neighbours = np.intersect1d(np.where(adjacency_matrix[a, :] != 0)[0],
                                       np.where(adjacency_matrix[b, :] != 0)[0])
    if not common_neighbours.size:
        return None
    return common_neighbours


def find_available_triads(is_directed, adjacency_matrix):
    """Searches for available triads in adjacency matrix. Found triads are returned as a list."""
    triads = []
    n_agents = adjacency_matrix.shape[0]
    for base_agent in range(n_agents):
        for neighbour_agent in range(n_agents):
            if neighbour_agent != base_agent and adjacency_matrix[base_agent, neighbour_agent]:
                common_neighbours = find_common_neighbours(adjacency_matrix, base_agent, neighbour_agent)
                if common_neighbours is not None:
                    for cn in common_neighbours:
                        triad = [base_agent, neighbour_agent, cn]
                        if not is_directed:
                            triad = sorted(triad)
                        triads.append(triad)
    # drop duplicates
    triads = np.unique(triads, axis=0)
    if not triads.size:
        warnings.warn('No triad found in the given adjacency matrix!')
    return [build_triad(*triad) for triad in triads]


def classify_triad(adjacency_matrix, triad):
    """
    Classification as in the get_antal_statistics() desc.
    Returns string \"tr_i\"
    """
    ab, bc, ac = [0 if v < 0 else 1 for v in adjacency_matrix[triad]]
    return "tr"+str(range(10)[ab*4 + ac*2 + bc])


@register_experiment
class LtdGeneral(Experiment):
    """
    Generalised experiment to run LTD, DLTD and StatusLTD on any network's topology.
    """
    def __init__(self, args):
        # Basic settlements
        super().__init__(args)
        self.adm = None
        self.triads = None
        self.links_num = None
        self.is_directed = True if self.args.dynamics in ['DLTD', 'SLTD'] else False
        if self.args.dynamics == 'SLTD':
            self.status_imbalanced_triads = np.array([[-1, -1, 1], [1, 1, -1]])
        if self.args.dynamics in ['DLTD', 'SLTD']:
            self.ltd_step = self.ltd_step_agent_based
        else:
            self.ltd_step = self.ltd_step_triad_based

        stamp = str(datetime.now()).replace(' ', '_').replace('.', '_').replace(":", "-")
        self.output_path = Path('./outputs/') / self.__class__.__name__ / self.args.exp_name / stamp
        self.output_fname = 'outputs.tsv'
        self.initialize_output_file()

        # Setup logger
        logging.basicConfig(filename=str(self.output_path / 'logs.log'),
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s- %(message)s')
        logging.info('Experiment is set up successfully')

    def __call__(self, *args, **kwargs):
        assert self.args.hprobability is not None, '--hprobability (-ph) argument is required'
        if self.args.track_changes:
            _performance_message = "This simulation runs with tracking changes of triads what may " \
                                   "significantly reduce computational performance."
            print(f"!!WARNING!!\n{_performance_message}\n!! !!")
            logging.warning(_performance_message)
        logging.info(f'Dynamics: {self.args.dynamics}')
        ph_arr = [prob for prob in self.args.hprobability for ii in range(self.args.repetitions)]
        if self.args.dynamics in ['LTD', 'DLTD']:
            q_arr = [0]
            ps_arr = [0]
        else:
            q_arr = self.args.q
            ps_arr = self.args.sprobability

        all_simulations_count = len(q_arr) * len(ph_arr) * len(ps_arr)
        done_simulations_count = 0

        for q in q_arr:
            for ph in ph_arr:
                for ps in ps_arr:
                    done_simulations_count += 1
                    logging.info(f"Parameters on this step: q={q} ph={ph} ps={ps}: "
                                 f"sim {done_simulations_count}/{all_simulations_count}")
                    self.initialize_connections()
                    self.triads = find_available_triads(
                        is_directed=self.is_directed,
                        adjacency_matrix=self.adm
                    )
                    """Initialize output data frame"""
                    output_columns = [f'tr{i}' for i in range(8)]
                    output_columns.append('rho')
                    output = {}
                    reciprocity = []
                    changes = []
                    for column in output_columns:
                        output[column] = []
                    output = pd.DataFrame(output)
                    updates_num = self.links_num * self.args.steps
                    _ps = ps if ps >= 0 else ph  # ps equals to heider prob if ps parameter below 0

                    """Get initial links-to-triads"""
                    if self.args.initial_link_to_triads:
                        logging.info('Saving initial links-to-triads')
                        with open(self.output_path / 'initial_link_to_triads.csv', 'a') as file:
                            for _, row in self.get_link_in_triad().iterrows():
                                file.write(f"{q}\t{ph}\t{ps}\t" + "\t".join([str(el) for el in row]) + '\n')

                    """Run simulation"""
                    for i in tqdm(range(updates_num), desc=f'ph={ph}, ps={ps}, q={q}'):
                        picked_triad, updated_link = self.step(ph, q, _ps)
                        if self.args.track_changes and updated_link:
                            changes.append(dict(
                                self.get_acc_inc_change(picked_triad, updated_link)
                            ))

                        if not i % self.links_num:
                            logging.info(f'Saving steps on {(i/updates_num*100):.2f}%: ph={ph}, ps={ps}, q={q}:'
                                         f'sim {done_simulations_count}/{all_simulations_count}')
                            stats = self.get_antal_statistics()
                            output = output.append(stats, ignore_index=True)
                            if self.args.reciprocity:
                                reciprocity.append(self.get_reciprocity())

                    """Save results"""
                    logging.info('Saving results...')
                    with open(self.output_path / self.output_fname, 'a') as file:
                        results = []
                        for column in output.columns:
                            # Join array by ',' as string
                            results.append(reduce(lambda x, y: str(x) + ',' + str(y), output[column].tolist()))
                        results_string = "\t".join([f"{result}" for result in results])
                        file.write(f"{q}\t{ph}\t{ps}\t{results_string}\n")
                    if self.args.reciprocity:
                        with open(self.output_path / 'reciprocity.csv', 'a') as recip_file:
                            recip_file.write(f"{q}\t{ph}\t{ps}\t{reciprocity}\n")
                    if self.args.track_changes:
                        with open(self.output_path / 'changes.csv', 'a') as file:
                            for timestep, change in enumerate(changes):
                                file.write(f"{q}\t{ph}\t{ps}\t{timestep}\t"
                                           f"{change['incidental'][0]}\t"
                                           f"{change['incidental'][1]}\t"
                                           f"{change['accidental'][0]}\t"
                                           f"{change['accidental'][1]}\n")
                    if self.args.link_to_triads:
                        logging.info('Saving links-to-triads')
                        with open(self.output_path / 'link_to_triads.csv', 'a') as file:
                            for _, row in self.get_link_in_triad().iterrows():
                                file.write(f"{q}\t{ph}\t{ps}\t" + "\t".join([str(el) for el in row]) + '\n')
                    logging.info('Results saved')
                    if self.args.save_adjacency_matrix:
                        adm_filename = f'adm_q{q}_ph{ph}_ps{ps}.npy'
                        np.save(self.output_path / ADM_SUBDIR_NAME / adm_filename, self.adm)
                        logging.info(f'Adjacency matrix saved to: {self.output_path / ADM_SUBDIR_NAME / adm_filename}')
        logging.info('Simulation has ended after finishing all tasks')

    @classmethod
    def add_args(cls, subparsers):
        """Add simulation arguments"""
        exp_parser = subparsers.add_parser(cls.name(), help='LTD simulation for any network')
        exp_parser.add_argument('exp_name', help="Name identifying the experiment run")
        exp_parser.add_argument('-n', '--n-agents', type=int, default=100, help='Number of agents')
        exp_parser.add_argument('-ph', '--hprobability', type=float, nargs='*', default=None, required=False,
                                help='Probability p_H')
        exp_parser.add_argument('-ps', '--sprobability', type=float, nargs='*', default=None, required=False,
                                help='Probability p_S')
        exp_parser.add_argument('-q', type=float, nargs='*', default=None, required=False,
                                help='q parameter of SLTD dynamics')
        exp_parser.add_argument('-s', '--steps', type=int, default=1000,
                                help='Number of steps of simulation, each step'
                                     ' is number of updates equal to n-agents')
        exp_parser.add_argument('-r', '--repetitions', type=int, default=1)
        exp_parser.add_argument('-t', '--topology', type=str, choices=generator.keys(), required=True)
        exp_parser.add_argument('--dynamics', type=str, choices=['LTD', 'DLTD', 'SLTD'], required=True,
                                help="LTD stands for triad based dynamics, DLTD for agent based dynamics"
                                     " and SLTD for LTD with status")
        exp_parser.add_argument('--generator-params', type=float, nargs='*',
                                help='Give required arguments for network generating except from the n_agents'
                                     ' argument. Pass it in the proper order', default=[])
        exp_parser.add_argument('--reciprocity', action='store_true',
                                help='Use this argument if reciprocity level computations are required. '
                                     'Results will be saved in outputdir/reciprocity.csv')
        exp_parser.add_argument('--track-changes', action='store_true',
                                help="Keep track of incidental and accidental triad changes. Warning! This is very "
                                     "computational costly!")
        exp_parser.add_argument('--link-to-triads', action='store_true',
                                help="Return a file with information of all adjacent triads' types to each link")
        exp_parser.add_argument('--initial-link-to-triads', action='store_true',
                                help='As link-to-triads but before the evolution begins')
        exp_parser.add_argument('--save-adjacency-matrix', action='store_true',
                                help="Save adjacency matrix with signs after the network evolution. Saving will be "
                                     "performed after each simulation.")

    def initialize_connections(self):
        self.adm = generator[self.args.topology](
            self.args.n_agents,
            *self.args.generator_params).get_adjacency_matrix()
        connections = np.where(self.adm != 0)
        self.links_num = len(connections[0])
        self.adm[connections] = np.random.choice([-1, 1], self.links_num)

    def pick_random_triad(self):
        return self.triads[random.randint(0, len(self.triads)-1)]

    def step(self, ph, q, ps):
        # Returns "coordinates" of links in connection matrix
        triad = self.pick_random_triad()
        if random.random() > q:
            updated_link = self.ltd_step(ph, triad)
        else:
            updated_link = self.status_step(ps, triad)
        return triad, updated_link

    def ltd_step_agent_based(self, p, triad):
        values = self.adm[triad]
        x_triad, y_triad = triad
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
        self.adm[x_triad[to_change], y_triad[to_change]] *= -1
        return x_triad[to_change], y_triad[to_change]

    def ltd_step_triad_based(self, p, triad):
        values = self.adm[triad]
        x_triad, y_triad = triad
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
            self.adm[x_triad[to_change], y_triad[to_change]] *= -1
            return x_triad[to_change], y_triad[to_change]

    def status_step(self, ps, triad):
        """
        Imbalanced triads:\
        a)  a1->a2 = -1\
            a2->a3 = -1\
            a1->a3 = +1\

        b)  a1->a2 = +1\
            a2->a3 = +1\
            a1->a3 = -1\
        Any other triad is balanced according to statuses
        """
        x_triad, y_triad = triad
        # Extracts values of the links
        values = self.adm[x_triad, y_triad]
        balanced = not any([all(values == imbalanced_triad) for imbalanced_triad in self.status_imbalanced_triads])
        if balanced:
            return
        else:
            probs = [ps, 1 - ps] if values[0] < values[2] else [1 - ps, ps]
            to_change = np.random.choice([0, 2], p=probs)
            self.adm[x_triad[to_change], y_triad[to_change]] *= -1
            return x_triad[to_change], y_triad[to_change]

    def initialize_output_file(self):
        self.output_path.mkdir(parents=True)
        commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
        with open(self.output_path / self.output_fname, 'w') as file:
            triad_header = "\t".join(["tr" + str(i) for i in range(8)])
            if not self.is_directed:
                file.write("# Notice that this simulation was run for non directed network and triads should be joined"
                           " first before analysing\n")
            file.write(f"# Commit: {commit_hash}"
                       f"# Arguments: {str(sys.argv)}\n"
                       "# q - q probability\n"
                       "# p - p probability\n"
                       "# ps - probability of positive link in status update\n"
                       "# nk - number of nk triads\n"
                       "# rho - number of positive triads\n")
            file.write(f"q\tp\tps\t" + triad_header + "\trho\n")
        if self.args.reciprocity:
            with open(self.output_path / 'reciprocity.csv', 'w') as recip_file:
                recip_file.write(f'# Commit: {commit_hash}\n')
                recip_file.write(f'# Arguments: {str(sys.argv)}\n')
                recip_file.write(f"q\tph\tps\treciprocity\n")
        if self.args.track_changes:
            with open(self.output_path / 'changes.csv', 'w') as file:
                file.write(f'# Commit: {commit_hash}')
                file.write(f'# Arguments: {str(sys.argv)}\n')
                file.write(f"q\tph\tps\ttimestep\tincidental_before\t"
                           f"incidental_after\taccidental_before\taccidental_after\n")
        if self.args.link_to_triads:
            with open(self.output_path / 'link_to_triads.csv', 'w') as file:
                file.write(f'# Commit: {commit_hash}\n')
                file.write(f'# Arguments: {str(sys.argv)}\n')
                file.write('q\tph\tps\ttriad_type\tlink_AB\tlink_BC\tlink_AC\n')

        if self.args.initial_link_to_triads:
            with open(self.output_path / 'initial_link_to_triads.csv', 'w') as file:
                file.write(f'# Commit: {commit_hash}\n')
                file.write(f'# Arguments: {str(sys.argv)}\n')
                file.write('q\tph\tps\ttriad_type\tlink_AB\tlink_BC\tlink_AC\n')

        if self.args.save_adjacency_matrix:
            (self.output_path / ADM_SUBDIR_NAME).mkdir(parents=True)

    def get_reciprocity(self):
        """
        In general reciprocity is equal to:\n
        r = \sum_{i,j}A_{ij}A_{ji}/E + 1/2\n
        for complete graph:\n
        r = 2\sum_{i,j}A_{ij}A_{ji}/(n(n-1)) + 1/2
        """
        n = self.adm.shape[0]
        e = self.links_num
        recip = 0
        for i in range(n):
            for j in range(i+1, n):
                recip += self.adm[i, j]*self.adm[j, i]
        recip /= e
        return recip + 1/2

    def get_acc_inc_change(self, triad, link):
        """
        triad: picked triad to be changed, coordinates should be given as follows: [a, b, a], [b, c, c]
        link: link that was change in a format: [start, end]

        Given triad and links will be evaluated as if, the update was already done.
        Returns as dict in a given format:
        {
        incidental = (*before*, *after*),
        accidental = ([*before_1*, *before_2*, ... , *before_k*], [*after_1*, *after_2*, ... , *after_k*])
        }
        """
        triad_index = None
        adm = np.copy(self.adm)

        # Get index of incidentally changed triad:
        for i, t in enumerate(self.triads):
            if t == triad:
                triad_index = i
                break

        # Compute incidental and accidental changes after update
        incidental_triad_after = classify_triad(adm, triad)
        accidentally_changed_triads_after = [classify_triad(adm, t)
                                             for t in self.triads[:triad_index] + self.triads[triad_index+1:]]

        # Compute incidental and accidental changes before update
        adm[link[0], link[1]] *= -1
        incidental_triad_before = classify_triad(adm, triad)
        accidentally_changed_triads_before = [classify_triad(adm, t)
                                              for t in self.triads[:triad_index] + self.triads[triad_index + 1:]]

        return dict(
            incidental=(incidental_triad_before, incidental_triad_after),
            accidental=(accidentally_changed_triads_before, accidentally_changed_triads_after)
        )

    def get_link_in_triad(self):
        """
        Returns a data frame with columns about triads in which are specific links
        """
        _COLNAMES = ['triad_type', 'linkA', 'linkB', 'linkC']
        link_in_triad_df = pd.DataFrame(columns=_COLNAMES)
        for triad in self.triads:
            x_triad, y_triad = triad
            new_row = [classify_triad(self.adm, triad,)]
            new_row.extend([f"{source}-{target}" for source, target in zip(x_triad, y_triad)])
            link_in_triad_df.loc[len(link_in_triad_df)] = new_row
        return link_in_triad_df

    def get_antal_statistics(self):
        """
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

        returns columns in order:
        tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, rho
        """
        if not self.is_directed:
            positive_links = np.sum(self.adm[np.triu_indices(self.adm.shape[0], k=1)] > 0)
        else:
            positive_links = np.sum(self.adm > 0)

        n_arr = [0] * 8
        for triad in self.triads:
            """
            Variable *triad* relates to: L1: AB, L2: BC, L3: AC, 
            so to achieve specific index consider -1 as 0 and compute the index of triad type
            as it was binary: i = AB*2^2 + AC*2^1 + BC*2^0
            """
            connections_values = [0 if cv < 0 else 1 for cv in self.adm[triad]]
            n_arr[connections_values[0]*4 + connections_values[2]*2 + connections_values[1]] += 1
        """Build return dict"""
        return_dict = {}
        for i in range(8):
            return_dict[f'tr{i}'] = n_arr[i]
        return_dict['rho'] = positive_links
        return return_dict
