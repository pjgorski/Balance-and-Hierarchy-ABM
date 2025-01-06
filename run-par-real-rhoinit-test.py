# This file was used for testing.  
# 

import argparse
import sys
import itertools

import multiprocessing

import numpy as np

from experiments import EXPERIMENTS
from networktools.network_generators import generator

NUM_PROC = 20

dataset = "WikiElections"
wiki_fname = "../data/wikielections/wikielections_triads2.h5"
dataset = "CompleteGraph"
# wiki_fname = "../data/wikielections/wikielections_triads2.h5"
fname = "../data/complete/complete_triads.h5"

try:
    network = generator[dataset]()
except (FileNotFoundError, KeyError):
    dataset = fname
    network = generator['GeneralRealNetwork'](dataset)

what = "rho_inits2"
what = "ltd_real_testing"

if what == "chosen_grid_short":
    
    reps = 1
elif what == "rho_inits":
    foldername = "outputs/complete-triads-test"
    
    sim_sets = [[list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5)), 
                 list(np.arange(0, 1.01, 0.5))], 
                 ]
    
    rho_inits = list(np.arange(0, 0.75, 0.5))
    steps = 10
    saving_multiplier = 1
    reps = 1
    keep_rho_at = [0.67, 0.9, 0.65, 0.92]
elif what == "rho_inits2":
    foldername = "outputs/complete-triads-test"
    
    sim_sets = [[list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5)), 
                 list(np.arange(0, 1.01, 0.5))], 
                 ]
    
    rho_inits = [0,  0.3, 0.7]
    steps = 10
    saving_multiplier = 1
    reps = 2
    control_initial_rho = True
    keep_rho_at = [0.67, 0.9, 0.65, 0.92]
elif what == "ltd_real_testing":
    NUM_PROC = 1
    np.random.seed(0)
    foldername = "ltd_real_test"
    
    sim_sets = [[list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5)), 
                 list(np.arange(0, 1.01, 0.5))], 
                 ]
    
    rho_inits = [0,  0.3, 0.6, 0.9]
    steps = 10
    saving_multiplier = 1
    reps = 2
    control_initial_rho = True
    keep_rho_at = [0.67, 0.9, 0.65, 0.92]
elif what == "n100long":
    foldername = "outputs/n100_long"
    psb_s = np.arange(0,1.01,0.1)
    ph_s = [0.45, 0.5, 0.55]
    q_s = [0.95,]
    rho_init = 0.5
    steps = 4000
    reps = 10
    N = 100
elif what == "n100long2":
    foldername = "outputs/n100_long"
    psb_s = [0.2, 0.8]
    ph_s = [*np.arange(0,1.01,0.1), 0.45, 0.55, 0.49, 0.51]
    # np.append(ph_s)
    q_s = [0.8, 0.95,]
    rho_init = 0.5
    steps = 4000
    reps = 10
    N = 100

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
elif what in ["change_q", "n100", "n32long", "n32long2", "n100long", "n100long2"]:
    all_args = [prepare_args(N, steps, reps, psb, ph, q, rho_init, foldername) for psb, ph, q in itertools.product(psb_s, ph_s, q_s)]
# all_experiments = [EXPERIMENTS['LtdStatus'](args_here) for args_here in all_args]

with multiprocessing.Pool(processes=NUM_PROC) as pool:
    pool.map(create_and_run_one_exp, all_args)
    
