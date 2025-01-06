"""The goal of this file is to run simulations on real networks. 

It copies some elements of `ltd_general' and some of `ltd_complete`. 
"""

import csv
import logging
import sys
import warnings
from datetime import datetime
from functools import reduce

from pathlib import Path
# import random

import git
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from experiments.decorator import register_experiment
from experiments.experiment import Experiment
from networktools.network_generators import generator

ADM_SUBDIR_NAME = 'adjacency_matrices'

def build_triad(a, b, c, dim = 2):
    """
    L1: a->b
    L2: b->c
    L3: a->c
    """
    if dim == 2:
        return [a, b, a], [b, c, c]
    else:
        return [[a,b], [b,c], [a,c]]


#??? TO DELETE???
def classify_triad(adjacency_matrix, triad):
    """
    Classification as in the get_antal_statistics() desc.
    Returns string \"tr_i\"
    """
    ab, bc, ac = [0 if v < 0 else 1 for v in adjacency_matrix[triad]]
    return "tr"+str(range(10)[ab*4 + ac*2 + bc])


@register_experiment
class LtdReal(Experiment):
    """
    Generalised experiment to run LTD, DLTD and StatusLTD on real networks.
    """
    def __init__(self, args):
        # Basic settlements
        super().__init__(args)
        
        stamp = str(datetime.now()).replace(' ', '_').replace('.', '_').replace(":", "-")
        self.output_path = Path('./outputs/') / self.__class__.__name__ / self.args.exp_name / stamp
        self.output_fname = 'outputs.tsv'
        self.positive_k_fname = 'pkdensities.csv'
        
        self.initialize_connections()
        
        # self.adm = None
        # # self.triads = None
        # self.links_num = None
        self.is_directed = True
        
        # Dynamics type
        if hasattr(self.args, "agent_based"):
            if self.args.agent_based:
                self.ltd_step = self.ltd_step_agent_based
                self.status_step_agent_based = True
            else:
                self.ltd_step = self.ltd_step_triad_based
                self.status_step_agent_based = False
        else:    
            if self.args.ltd_agent_based:
                self.ltd_step = self.ltd_step_agent_based
            else:
                self.ltd_step = self.ltd_step_triad_based
            
            self.status_step_agent_based = True

        if self.args.on_triad_status:
            self.status_imbalanced_triads = np.array([[-1, -1, 1], [1, 1, -1]])
            self.status_step = self.status_step
        else:
            raise ValueError('Pair status not implemented here')
            # self.status_step = self.status_step_pair_balance



        if len(self.args.keep_rho_at) == 4:
            self.trigger_start_l = self.args.keep_rho_at[0]
            self.trigger_start_r = self.args.keep_rho_at[1]
            self.finish_sim_l = self.args.keep_rho_at[2]
            self.finish_sim_r = self.args.keep_rho_at[3]
        elif len(self.args.keep_rho_at) == 2:
            self.trigger_start_l = self.args.keep_rho_at[0]
            self.trigger_start_r = self.args.keep_rho_at[1]
            self.finish_sim_l = self.args.keep_rho_at[0]
            self.finish_sim_r = self.args.keep_rho_at[1]
        elif len(self.args.keep_rho_at) == 0:
            self.trigger_start_l = 0
            self.trigger_start_r = 1
            self.finish_sim_l = 0
            self.finish_sim_r = 1
        else:
            raise TypeError("keep_rho_at must have 2 or 4 input variables.")
        
        if not hasattr(self.args, "keep_average_rho"):
            self.args.keep_average_rho = False
        
        if self.args.control_initial_rho:
            if isinstance(self.args.rho_init, np.ndarray):
                self.args.rho_init[::-1].sort()
            elif isinstance(self.args.rho_init, list):
                self.args.rho_init.sort(reverse=True)
    
        self.initialize_output_file()
        
        if not hasattr(self.args, 'silent'):
            self.args.silent = False
        # if self.args

        # Setup logger
        logging.basicConfig(filename=str(self.output_path / 'logs.log'),
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s- %(message)s')
        logging.info('Experiment is set up successfully')

    @classmethod
    def add_args(cls, subparsers):
        """Add simulation arguments"""
        exp_parser = subparsers.add_parser(cls.name(), help='Optimized SLTD simulation for real networks')
        exp_parser.add_argument('exp_name', help="Name identifying the experiment run")
        exp_parser.add_argument('-d', '--dataset', type = str, default="WikiElections", help='Which dataset to choose: WikiElections or Slashdot? Alternatively path to file.')
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
        exp_parser.add_argument('-m', '--saving-multiplier', type=int, default=1, help = 'Every how many large steps net stats should be gathered.')
        exp_parser.add_argument('-t', '--build-triad', type = str, default="choose_agents", help='How to construct a triad? Possible values: "sample_triad", "choose_agents", "choose_and_sample"')
        exp_parser.add_argument('--ltd-agent-based', action='store_true')
        exp_parser.add_argument('--on-triad-status', action='store_true')
        exp_parser.add_argument('--save-pk-distribution', action='store_true')
        exp_parser.add_argument('--rho-init', type=float, nargs = '*', default=-1, required=False,
                                help='Initial density of positive links. If negative, true density is chosen.')
        exp_parser.add_argument('--no-triad-stats', action="store_true")
        
        exp_parser.add_argument('--keep-rho-at', type = float, nargs = '*', default= [], 
                                help="""If two values are given, when rho enters this range, geting out of this range will stop the simulation.
                                If 4 values are given, the first pair is the range that triggers the 2nd pair, and when the simulation's
                                rho is out of the 2nd pair range, it is stopped.
                                The stopping is also triggered when the simulation does not approach first range. """)
        exp_parser.add_argument('--keep-average-rho', action='store_false', 
                                help = """When it is true, then not the current rho is checked whether it is in `keep_rho_at` windows,
                                but the average (calculated every Large step, that is number of edges) rho is checked whether
                                it reached the window (trigger) and whether it left the window (this stops simulations). """)
        exp_parser.add_argument('--control-initial-rho', action='store_false', 
                                help = """When this is True, then if many rho_inits are given, they sorted in reverse order, 
                                and when one rho_init triggers simulation stop due to keep_rho_at argument, 
                                then the simulations for the next rho_inits (the smaller ones) may not be run at all. 
                                This happens according to the rules, seen in this example: keep_rho_at=[0.75, 0.8, 0.7, 0.85].
                                When for rho_init=0.5 one obtained rho equal to 0.7, 
                                then smaller rho_inits will give even smaller QS rho, so there is no sense to simulate it. """)
        
        exp_parser.add_argument('--network', action="store_true", help = "Not to be used in terminal. One can start simulation from a script by assiging a defined network. ")
        exp_parser.add_argument('--silent', action="store_true", help = "Set to make `tqdm` silent its output.")

    def clear(self):
        self.network = []
        self.adm = []
        self.connections = []

    def __call__(self, *args, **kwargs):
        assert self.args.probability is not None, '--probability (-p) argument is required'
        
        # logging.info(f'Dynamics: {self.args.dynamics}')
        
        simulation_history = []

        for q in self.args.q:
            for p in self.args.probability:
                for ps in self.args.psprob:
                    triggered_sim_stop_below_allowed_range = 0
                    triggered_sim_stop_above_allowed_range = 0
                    
                    rho_init_change_order = 0
                    
                    # The purpose of following code with `while` loop etc. is to stop more rho inits from both side. 
                    # Imagine parameter sets with only 0 (q=ps=ph=0), then we would like the simulation to stop after performing a run for largest rho_init.
                    # Imagine parameter sets with only 1 (q=ps=ph=0), then we would like the simulation to stop after performing a run for smallest rho_init.
                    # So the order of rho_inits should be changed in the runtime. 
                    # The order should be changes if for the large rho_init we get a result above range. 
                    # Or for the low rho_init we get the result below range
                    rho_inits = np.copy(self.args.rho_init)
                    
                    if self.args.control_initial_rho:
                        rho_inits[::-1].sort()
                        largest_first = True
                    
                    continue_ = False
                    while len(rho_inits) > 0:
                        
                        if triggered_sim_stop_below_allowed_range == self.args.repetitions:
                            continue_ = True
                        elif triggered_sim_stop_above_allowed_range == self.args.repetitions:
                            continue_ = True
                        else:
                            if self.args.control_initial_rho:
                                if rho_init_change_order == self.args.repetitions:
                                    rho_inits = rho_inits[::-1]
                                    largest_first = not largest_first
                                    pass
                            
                            triggered_sim_stop_below_allowed_range = 0
                            triggered_sim_stop_above_allowed_range = 0
                            rho_init_change_order = 0
                        
                        rho_init = rho_inits[0]
                        rho_inits = rho_inits[1:]
                        
                        if continue_:
                            simulation_history.append((q, p, ps, rho_init, triggered_sim_stop_below_allowed_range, triggered_sim_stop_above_allowed_range, self.args.repetitions))
                            continue
                        for rep_ctr in range(self.args.repetitions):
                            
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
                            self.initialize_signs(rho_init)
                            updates_num = self.links_num * self.args.steps
                            if updates_num == 0:
                                raise ValueError("updates_num is 0")
                            
                            _ps = ps if ps >= 0 else p  # ps equals to heider prob

                            current_rho = self.get_pos_links() / self.links_num
                            old_rho = current_rho 
                            old_rho_decreases_count = 0
                            triggered_finish_out_of_range = False
                            
                            if self.args.keep_average_rho:
                                average_rho_cumulative = 0.
                            
                            """Run simulation"""
                            for i in tqdm(range(updates_num), disable=self.args.silent):
                                current_rho += self.step(p, q, _ps) / self.links_num
                                
                                if not self.args.keep_average_rho:
                                    if triggered_finish_out_of_range:
                                        if (current_rho > self.finish_sim_r) or (current_rho < self.finish_sim_l):
                                            # print(current_rho)
                                            # print(self.links_num)
                                            # stats = self.get_antal_statistics()
                                            # print(stats)
                                            if self.args.control_initial_rho:
                                                if largest_first:
                                                    if (current_rho < self.finish_sim_l):
                                                        triggered_sim_stop_below_allowed_range += 1
                                                    else:
                                                        rho_init_change_order += 1
                                                else: 
                                                    if (current_rho > self.finish_sim_r):
                                                        triggered_sim_stop_above_allowed_range += 1
                                                    else:
                                                        rho_init_change_order += 1
                                            break  
                                    elif (current_rho < self.trigger_start_r) and (current_rho > self.trigger_start_l):
                                        triggered_finish_out_of_range = True
                                else:
                                    average_rho_cumulative += current_rho
                                
                                if not i % (self.links_num * self.args.saving_multiplier):
                                    
                                    if self.args.keep_average_rho:
                                        average_rho_cumulative /= self.links_num * self.args.saving_multiplier
                                        if triggered_finish_out_of_range:
                                            if (average_rho_cumulative > self.finish_sim_r) or (average_rho_cumulative < self.finish_sim_l):
                                                # print(current_rho)
                                                # print(self.links_num)
                                                # stats = self.get_antal_statistics()
                                                # print(stats)
                                                if self.args.control_initial_rho:
                                                    if largest_first:
                                                        if (average_rho_cumulative < self.finish_sim_l):
                                                            triggered_sim_stop_below_allowed_range += 1
                                                        else:
                                                            rho_init_change_order += 1
                                                    else: 
                                                        if (average_rho_cumulative > self.finish_sim_r):
                                                            triggered_sim_stop_above_allowed_range += 1
                                                        else:
                                                            rho_init_change_order += 1
                                                break  
                                        elif (average_rho_cumulative < self.trigger_start_r) and (average_rho_cumulative > self.trigger_start_l):
                                            triggered_finish_out_of_range = True
                                        average_rho_cumulative = 0.
                                
                                    if not triggered_finish_out_of_range:
                                        """Checking if we approach first range"""
                                        dif_to_left_range_old = self.trigger_start_l - old_rho 
                                        dif_to_right_range_old = old_rho - self.trigger_start_r 
                                        if dif_to_left_range_old > 0:
                                            dif_to_left_range_new = self.trigger_start_l - current_rho
                                            if dif_to_left_range_old < dif_to_left_range_new:
                                                old_rho_decreases_count += 1
                                            else:
                                                old_rho_decreases_count = 0
                                        elif dif_to_right_range_old > 0:
                                            dif_to_right_range_new = current_rho - self.trigger_start_r 
                                            if dif_to_right_range_old < dif_to_right_range_new:
                                                old_rho_decreases_count += 1
                                            else:
                                                old_rho_decreases_count = 0
                                        
                                        if old_rho_decreases_count == 3:
                                            if self.args.control_initial_rho:
                                                if largest_first:
                                                    if (current_rho < self.finish_sim_l):
                                                        triggered_sim_stop_below_allowed_range += 1
                                                    else:
                                                        rho_init_change_order += 1
                                                else: 
                                                    if (current_rho > self.finish_sim_r):
                                                        triggered_sim_stop_above_allowed_range += 1
                                                    else:
                                                        rho_init_change_order += 1
                                            # if self.args.control_initial_rho:
                                            #     if largest_first and (current_rho < self.finish_sim_l):
                                            #         triggered_sim_stop_below_allowed_range += 1
                                            #     elif (not largest_first) and (current_rho > self.finish_sim_r):
                                            #         triggered_sim_stop_above_allowed_range += 1
                                            break
                                        
                                        old_rho = current_rho
                                    
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
                                        for _ in range(int(i / (self.links_num* self.args.saving_multiplier)) + 1, int(self.args.steps / self.args.saving_multiplier)):
                                            output = output.append(stats, ignore_index=True)
                                        break
                                
                            
                            stats = self.get_antal_statistics()
                            output = output.append(stats, ignore_index=True)
                            if self.args.save_pk_distribution:
                                with open(self.output_path / (f'p{p}q{q}ps{ps}'.replace('.', '-')
                                                            + self.positive_k_fname), 'a') as pkfile:
                                    csv_writer = csv.writer(pkfile, delimiter=',')
                                    csv_writer.writerow(self.get_positive_in_degree_dist())
                            
                            """Save results"""
                            with open(self.output_path / self.output_fname, 'a') as file:
                                results = []
                                for column in output.columns:
                                    # Join array by ',' as string
                                    results.append(reduce(lambda x, y: str(x) + ',' + str(y), output[column].tolist()))
                                results_string = "\t".join([f"{result}" for result in results])
                                file.write(
                                    f"{q}\t{p}\t{ps}\t{rho_init}\t{results_string}\n"
                                    # f"{self.get_balanced_pairs_count()}\n"
                                )
                            simulation_history.append((q, p, ps, rho_init, triggered_sim_stop_below_allowed_range, triggered_sim_stop_above_allowed_range, rep_ctr + 1))
                                
        logging.info('Simulation has ended after finishing all tasks')
        return simulation_history

    def initialize_connections(self):
        if not self.args.network:
            logging.info('Started reading dataset.')
            if self.args.dataset not in generator:
                self.network = generator['GeneralRealNetwork'](self.args.dataset)
            else:
                self.network = generator[self.args.dataset]()
            logging.info('Finished reading dataset.')
        else:
            self.network = copy.deepcopy(self.args.network)

        
        self.adm = self.network.get_edges_data()
        if self.network.n_agents <= 1000:
            self.connections = np.where(self.adm != 0)
            self.links_num = len(self.connections[0])
        else:
            # connections = (np.array([agents[0] for agents in self.adm.keys()]), 
            #                np.array([agents[1] for agents in self.adm.keys()]))
            self.connections = self.network.elist.keys()
            self.links_num = len(self.connections)
    
    def initialize_signs(self, rho_init):
        if rho_init < 0: #we take real values as starting points
            self.network.refresh()
            self.adm = self.network.get_edges_data()
        else:
            if self.network.n_agents <= 1000:
                if rho_init == 0.5:
                    self.adm[self.connections] = np.random.choice([-1, 1], self.links_num)
                else:
                    neg_dens = 1. -rho_init
                    self.adm[self.connections] = np.random.choice([-1, 1], self.links_num, p = [neg_dens, rho_init])
            else:
                neg_dens = 1. -rho_init
                rand_vals = np.random.choice([-1, 1], self.links_num, p = [neg_dens, rho_init])
                # print(self.connections)
                # print(rand_vals)
                # print(self.links_num)
                for key, val in zip(self.connections, rand_vals):
                    self.adm[key] = val

    # def pick_random_triad(self):
    #     return self.triads[random.randint(0, len(self.triads)-1)]

    def sample_triad(self):
        if self.args.build_triad == "choose_agents":
            focal = np.random.choice(self.network.focal_agents)
            meet = np.random.choice(self.network.b_agents_of_focal_agents[focal])
            another = np.random.choice(self.network.c_agents_of_ab_pairs[(focal, meet)])
        elif self.args.build_triad == "sample_triad":
            T = len(self.network.all_triads)
            focal, meet, another = self.network.all_triads[np.random.choice(np.arange(T))]
        elif self.args.build_triad == "choose_and_sample":
            focal = np.random.choice(self.network.focal_agents)            
            T1 = len(self.network.triads_of_agents[focal])
            focal, meet, another = self.network.triads_of_agents[
                np.random.choice(np.arange(T1))]
        else:
            raise ValueError("Wrong method for constructing triads.")
        
        if self.network.n_agents <= 1000:
            return build_triad(focal, meet, another, dim = 2)
        else:
            return build_triad(focal, meet, another, dim = 1)

    def step(self, p, q, ps):
        """

        Args:
            p (_type_): _description_
            q (_type_): _description_
            ps (_type_): _description_

        Returns:
            int: 1/-1 if negative/positive link became positive/negative, 0 - otherwise 
        """
        triad = self.sample_triad()
        if np.random.random() > q:
            return self.ltd_step(p, triad)
        else:
            return self.status_step(ps, triad)

    def ltd_step_agent_based(self, p, triad):
        """_summary_

        Args:
            p (_type_): _description_
            triad (_type_): _description_

        Returns:
            int: 1/-1 if negative/positive link became positive/negative, 0 - otherwise 
        """
        if self.network.n_agents <= 1000:
            x_triad, y_triad = triad
            # Extracts values of the links
            values = self.adm[x_triad, y_triad]
        else:
            values = [self.adm[pair[0], pair[1]] for pair in triad]
        balanced = True if reduce(lambda x, y: x * y, values) > 0 else False
        if balanced:
            return 0
        else:
            if values[0] * values[2] > 0:
                # a1->a2 and a1->a3 are both positive or negative
                to_change = np.random.choice([0, 2])
            else:
                probs = [p, 1 - p] if values[0] < values[2] else [1 - p, p]
                to_change = np.random.choice([0, 2], p=probs)
            if self.network.n_agents <= 1000:
                self.adm[x_triad[to_change], y_triad[to_change]] *= -1
                # return x_triad[to_change], y_triad[to_change]
                return self.adm[x_triad[to_change], y_triad[to_change]]
            else:
                x = triad[to_change][0]
                y = triad[to_change][1]
                self.adm[x, y] *= -1
                # return triad[to_change]
                return self.adm[x, y]

    def ltd_step_triad_based(self, p, triad):
        """_summary_

        Args:
            p (_type_): _description_
            triad (_type_): _description_

        Returns:
            int: 1/-1 if negative/positive link became positive/negative, 0 - otherwise 
        """
        if self.network.n_agents <= 1000:
            x_triad, y_triad = triad
            # Extracts values of the links
            values = self.adm[x_triad, y_triad]
        else:
            values = [self.adm[pair[0], pair[1]] for pair in triad]
        
        balanced = True if reduce(lambda x, y: x * y, values) > 0 else False
        if balanced:
            return 0
        else:
            num_of_negatives = len([v for v in values if v < 0])
            if num_of_negatives != 1:
                to_change = np.random.choice([0, 1, 2])
            else:
                negatives = np.where(values < 0)[0]
                if np.random.random() < p:
                    # Change negative link to positive
                    to_change = negatives[0]
                else:
                    # Change positive link to negative
                    positives = np.where(values > 0)[0]
                    to_change = np.random.choice(positives)
            if self.network.n_agents <= 1000:
                self.adm[x_triad[to_change], y_triad[to_change]] *= -1
                # return x_triad[to_change], y_triad[to_change]
                return self.adm[x_triad[to_change],y_triad[to_change]]
            else:
                x = triad[to_change][0]
                y = triad[to_change][1]
                self.adm[x, y] *= -1
                # return triad[to_change]
                return self.adm[x, y]

    def update_link_agent_based(self, p, values, agent_based_flag):
        """Updates a link of a triad. If the triad has 3 links of the same sign, then any of its links. 
        If it has a 2/1 positive/negative, then with prob p, a negative links is changed to become positive. 

        Args:
            p (_type_): _description_
            values (_type_): _description_
            agent_based_flag (bool): _description_
            
        Returns: an index which link should be changed
        """
        
        important_links = [0,2] if agent_based_flag else [0,1,2]
        # important_values = [values[0], values[2]] if agent_based_flag else values
        
        # num_of_negatives = len([v for v in important_values if (v < 0) ])
        num_of_negatives = len([v for i, v in enumerate(values) if (v < 0) and i in important_links])
        if (num_of_negatives == len(important_links)) or (num_of_negatives == 0):
            to_change = np.random.choice(important_links)
        else:
            negatives = np.where([v * (i in important_links) < 0 for i,v in enumerate(values)])[0]
            if np.random.random() < p:
                # Change negative link to positive
                to_change = np.random.choice(negatives)
            else:
                # Change positive link to negative
                positives = np.where([v * (i in important_links) > 0 for i,v in enumerate(values)])[0]
                to_change = np.random.choice(positives)
        
        return to_change

    def status_step(self, ps, triad):
        """
        Unbalanced triads:\
        a)  a1->a2 = -1\
            a2->a3 = -1\
            a1->a3 = +1\

        b)  a1->a2 = +1\
            a2->a3 = +1\
            a1->a3 = -1\
        Any other triad is balanced according to statuses
        
        Returns:
            int: 1/-1 if negative/positive link became positive/negative, 0 - otherwise 
        """
        if self.network.n_agents <= 1000:
            x_triad, y_triad = triad
            # Extracts values of the links
            values = self.adm[x_triad, y_triad]
        else:
            values = [self.adm[pair[0], pair[1]] for pair in triad]
        balanced = not any([all(values == imbalanced_triad) for imbalanced_triad in self.status_imbalanced_triads])
        if balanced:
            return 0
        else:
            # probs = [ps, 1 - ps] if values[0] < values[2] else [1 - ps, ps]
            # to_change = np.random.choice([0, 2], p=probs)
            to_change = self.update_link_agent_based(ps, values, self.status_step_agent_based)
            if self.network.n_agents <= 1000:
                self.adm[x_triad[to_change], y_triad[to_change]] *= -1
                # return x_triad[to_change], y_triad[to_change]
                return self.adm[x_triad[to_change], y_triad[to_change]]
            else:
                # print(values)
                # print(triad)
                # middle = values[1]
                # if not ([triad[0][0], triad[0][1], triad[1][1]] in self.network.all_triads):
                #     raise ValueError("is the triad good?")
                # print(to_change)
                x = triad[to_change][0]
                y = triad[to_change][1]
                self.adm[x, y] *= -1
                # if middle != self.adm[triad[1][0], triad[1][1]]:
                #     raise ValueError("Changed the wrong link!")
                # print(self.adm[x,y])
                # print([self.adm[pair[0], pair[1]] for pair in triad])
                # return triad[to_change]
                return self.adm[x, y]

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
                        "# rho - number of positive triads\n")
                        # "# bp - number of balanced pairs\n")
                file.write(f"q\tp\tps\trho_init\t" + triad_header + "\trho\n")
            else:
                file.write(f"# Commit: {commit_hash}"
                        f"# Arguments: {str(self.args)}\n"
                        "# q - q probability\n"
                        "# p - p probability\n"
                        "# ps - probability of positive link in status update\n"
                        "# rho - number of positive triads\n")
                        # "# bp - number of balanced pairs\n")
                file.write(f"q\tp\tps\trho_init\t" + "rho\n")


    # def get_balanced_pairs_count(self):
    #     if self.is_directed:
    #         balance_m = self.connections * self.connections.T
    #         return np.where(np.triu(balance_m, 1) > 0)[0].size
    #     else:
    #         return self.links_num

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

    def get_pos_links(self):
        if self.network.n_agents <= 1000:
            if not self.is_directed:
                positive_links = np.sum(self.adm[np.triu_indices(self.adm.shape[0], k=1)] > 0)
            else:
                positive_links = np.sum(self.adm > 0)
        else:
            positive_links = sum(np.array(list(self.adm.values())) > 0)
        
        if not self.is_directed:
            return int(positive_links/2)
        else:
            return int(positive_links)
        
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
        positive_links = self.get_pos_links()

        if self.args.no_triad_stats:
            return {'rho': positive_links}
            # else:
            #     return {'rho': int(positive_links)}
        if not self.is_directed:
            n_arr = [0] * 4
            for triad in self.network.all_triads:
                if self.network.n_agents <= 1000:
                    x_triad, y_triad = triad
                    # Extracts values of the links
                    values = self.adm[x_triad, y_triad]
                else:
                    values = [self.adm[pair[0], pair[1]] for pair in triad]
                n_arr[np.sum(values < 0)] += 1
            return {'n0': int(n_arr[0]), 'n1': int(n_arr[1]), 'n2': int(n_arr[2]), 'n3': int(n_arr[3]),
                    'rho': int(positive_links/2)}
        else:
            n_arr = [0] * 8
            for triad in self.network.all_triads:
                """
                Variable *triad* relates to: L1: AB, L2: BC, L3: AC, 
                so to achieve specific index consider -1 as 0 and compute the index of triad type
                as it was binary: i = AB*2^2 + AC*2^1 + BC*2^0
                """
                if self.network.n_agents <= 1000:
                    x_triad, y_triad = build_triad(*triad, dim = 2) #triad
                    # Extracts values of the links
                    values = self.adm[x_triad, y_triad]
                else:
                    pairs = build_triad(*triad, dim=1)
                    values = [self.adm[pair[0], pair[1]] for pair in pairs]
                connections_values = [0 if cv < 0 else 1 for cv in values]
                n_arr[connections_values[0]*4 + connections_values[2]*2 + connections_values[1]] += 1
            """Build return dict"""
            return_dict = {}
            for i in range(8):
                return_dict[f'tr{i}'] = n_arr[i]
            return_dict['rho'] = positive_links
            return return_dict