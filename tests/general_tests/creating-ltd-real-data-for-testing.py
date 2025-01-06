import argparse
import os
import sys
import inspect
import itertools

# import multiprocessing
# NUM_PROC = 1

import numpy as np

# os.chdir("../..") 
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.getcwd())
print(os.getcwd())
print(sys.path)

from experiments import EXPERIMENTS
from networktools.network_generators import generator


foldername = "ltd_real_test"

if os.path.isdir("outputs/LtdReal/" + foldername):
    import shutil
    
    shutil.rmtree("outputs/LtdReal/" + foldername)

fname = "../data/complete/complete_triads.h5"

dataset = fname
network = generator['GeneralRealNetwork'](dataset)

what = "ltd_real_testing"

if what == "ltd_real_testing":
    np.random.seed(0)
    
    sim_sets = [[list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5)), 
                 list(np.arange(0, 1.01, 0.5))], 
                 ]
    
    rho_inits = [0,  0.3, 0.6, 0.9]
    steps = 10
    saving_multiplier = 1
    reps = 2
    control_initial_rho = True
    keep_rho_at = [0.67, 0.9, 0.65, 0.92]


def explode_sims(sim_sets):
    """ Returns simulation sets in the form of single list. 
    The input are sets with nested lists. 
    """
    exp_sets = []
    
    for sim_set in sim_sets:
        if len(sim_set) == 3:
            q, ps, ph = sim_set
            rho_init = []
        elif len(sim_set) == 4:
            q, ps, ph, rho_init = sim_set
        if not isinstance(q, (list,np.ndarray)):
            q = [q]
        if not isinstance(ps, (list,np.ndarray)):
            ps = [ps]
        if not isinstance(ph, (list,np.ndarray)):
            ph = [ph]
        if not isinstance(rho_init, (list,np.ndarray)):
            rho_init = [rho_init]
        
        for q_ in q:
            for ps_ in ps:
                for ph_ in ph:
                    if len(rho_init) > 0:
                        for rho_init_ in rho_init:
                            exp_sets.append([q_, ps_, ph_, rho_init_])
                    else:
                        exp_sets.append([q_, ps_, ph_])
    
    return exp_sets


def prepare_args2(dataset, network, single_sim_set, 
                  rho_init, steps, saving_multiplier, reps, no_triad_stats, 
                  foldername, keep_rho_at = [], control_initial_rho = False):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Experiments')
    args = parser.parse_args()
    
    args.dataset = dataset  
    args.network = network
    
    args.is_directed = True
    args.agent_based = True
    args.ltd_agent_based = True
    args.on_triad_status = True
    args.save_pk_distribution = False
    args.build_triad = 'choose_agents'
    
    args.exp_name = foldername
    
    args.probability = [single_sim_set[2]]
    args.q = [single_sim_set[0]]
    args.psprob = [single_sim_set[1]]
    
    if not isinstance(rho_init, (list,np.ndarray)):
        rho_init = [rho_init]
    args.rho_init = rho_init
    args.steps = steps
    args.saving_multiplier = saving_multiplier
    args.repetitions = reps
    
    args.no_triad_stats = no_triad_stats
    args.keep_rho_at = keep_rho_at
    args.control_initial_rho = control_initial_rho
    
    return args


def create_and_run_one_exp(args):
    experiment = EXPERIMENTS['LtdReal'](args)
    experiment()
    pass
    

if what in ["chosen_grid_short", "chosen_grid_short2", "rho_inits"]:
    exp_sets = explode_sims(sim_sets)
    
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, steps, saving_multiplier, reps, False, foldername, keep_rho_at) for single_sim_set in exp_sets]
elif what in ["rho_inits2", "ltd_real_testing"]:
    exp_sets = explode_sims(sim_sets)
    
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, 
                              steps, saving_multiplier, reps, False, 
                              foldername, keep_rho_at, control_initial_rho) for single_sim_set in exp_sets]

for args in all_args:
    np.random.seed(0)
    create_and_run_one_exp(args)

# with multiprocessing.Pool(processes=NUM_PROC) as pool:
#     pool.map(create_and_run_one_exp, all_args)