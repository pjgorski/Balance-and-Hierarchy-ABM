# This file was used to create simulation of a triad-based model. 
# See the file run-par-2.py for the description.

import argparse
import sys
import itertools

import multiprocessing

import numpy as np

from experiments import EXPERIMENTS
from networktools.network_generators import generator

NUM_PROC = 20

dataset = "CompleteGraph"
fname = "../data/complete/complete_triads.h5"

try:
    network = generator[dataset]()
except (FileNotFoundError, KeyError):
    dataset = fname
    network = generator['GeneralRealNetwork'](dataset)

what = "rho_inits"
# what = "rho_inits5"
# what = "test_run2"

if what == "rho_inits":
    foldername = "outputs/complete-triads-triad-based"
    
    sim_sets = [[[0.1, 0.8, 0.95], list(np.arange(0, 1.01, 0.1)), list(np.arange(0, 1.01, 0.1))],
                 ]
    
    rho_inits = [0.5, ]
    steps = 1000
    saving_multiplier = 1
    reps = 1
    control_initial_rho = False
    keep_rho_at = []
    silent = True

elif what == "test_run":
    foldername = "outputs/epinions-triads-test"
    
    sim_sets = [[list(np.arange(0.1, 0.95, 0.5)), list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5))], 
                # [0.0, list(np.arange(0,1.01, 0.1)), 0.5], [1.0, 0.5, list(np.arange(0,1.01, 0.1))]
                 ]
    
    rho_inits = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
    steps = 1
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    keep_rho_at = [0.67, 0.9, 0.65, 0.92]
elif what == "test_run2":
    foldername = "outputs/epinions-triads-test"
    
    sim_sets = [(0.1, 0.5, 0.8), (0.95, 0.5, 0.8), [list(np.arange(0., 0.95, 0.02)), 0.,0.], (0.7, 0.9, 0.5), (0.9, 0.9, 0.3), (0.9, 0.7, 0.5), 
                [list(np.arange(0., 0.95, 0.02)), 0.,0.], (0.9, 0.9, 0.3), (0.9, 0.7, 0.5), ]
    
    rho_inits = [0, 0.6, 0.9]
    steps = 100
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    keep_rho_at = [0.77, 0.97, 0.75, 0.98]

# rho_init_s = [0.6, 0.8, 0.9]

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
        if not isinstance(ph, (list,np.ndarray)):
            ph = [ph]
        if not isinstance(ps, (list,np.ndarray)):
            ps = [ps]
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
                  foldername, keep_rho_at = [], control_initial_rho = False, 
                  silent = False):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Experiments')
    args = parser.parse_args()
    
    args.dataset = dataset  
    args.network = network
    
    args.is_directed = True
    args.agent_based = False
    # args.ltd_agent_based = True
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
    
    args.silent = silent
    
    return args

def prepare_args(N, steps, reps, psb, ph, q, rho_init, foldername):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Experiments')
    args = parser.parse_args()
    args.n_agents = N
    args.is_directed = True
    args.agent_based = True
    args.exp_name = foldername
    if not isinstance(psb,(list,np.ndarray)):
        args.probability = [psb]
    if not isinstance(q,(list,np.ndarray)):
        args.q = [q]
    if not isinstance(ph,(list,np.ndarray)):
        args.psprob = [ph]
    args.steps = steps
    args.repetitions = reps
    args.ltd_agent_based = True
    if not isinstance(rho_init,(list,np.ndarray)):
        args.rho_init = [rho_init]
    args.no_triad_stats = True
    args.on_triad_status = True
    args.save_pk_distribution = False
    
    return args


def create_and_run_one_exp(args):
    experiment = EXPERIMENTS['LtdReal'](args)
    experiment()
    experiment.clear()
    del experiment
    pass
    

if what in ["chosen_grid_short", "chosen_grid_short2"]:
    exp_sets = explode_sims(sim_sets)
    
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, steps, saving_multiplier, reps, False, foldername, keep_rho_at) for single_sim_set in exp_sets]
elif what in ["rho_inits", ]:
    exp_sets = explode_sims(sim_sets)
    
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, 
                              steps, saving_multiplier, reps, False, 
                              foldername, keep_rho_at, control_initial_rho, 
                              silent) for single_sim_set in exp_sets]
# elif what in ["rho_inits22",]:
#     exp_sets = explode_sims2(sim_sets)
    
#     rho_init = rho_inits
    
#     all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, 
#                               steps, saving_multiplier, reps, False, 
#                               foldername, keep_rho_at, control_initial_rho) for single_sim_set in exp_sets]


with multiprocessing.Pool(processes=NUM_PROC, maxtasksperchild = 1) as pool:
    r = pool.map_async(create_and_run_one_exp, all_args, chunksize = 1)
    r.wait()
    
