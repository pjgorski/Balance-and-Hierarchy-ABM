# This file was used to create simulations of ABM on WikiElections dataset assuming initial positive link density same as in data. 
# See the file run-par-2.py for the description.

import argparse
import sys
import itertools

import multiprocessing

import numpy as np

from experiments import EXPERIMENTS
from networktools.network_generators import generator

NUM_PROC = 30

dataset = "WikiElections"
wiki_fname = "../data/wikielections/wikielections_triads2.h5"

try:
    network = generator[dataset]()
except (FileNotFoundError, KeyError):
    dataset = wiki_fname
    network = generator['GeneralRealNetwork'](dataset)

what = "chosen_grid_short2"

if what == "chosen_grid_short":
    foldername = "outputs/wiki2-s1000-triads"
    
    sim_sets = [[0, 0.5, [0.7,0.75, 0.8, 0.9]], 
                [0.1, list(np.arange(0, 1.01, 0.1)), [0.7, 0.75, 0.8, 0.85]], 
                 [0.2, list(np.arange(0, 1.01, 0.1)), [0.7, 0.8, 0.9]],
                 [0.3, list(np.arange(0, 0.51, 0.1)), [0.7, 0.8, 0.9]],
                 [0.3, list(np.arange(0.6, 1.01, 0.1)), [0.5, 0.6, 0.7, 0.8]],
                 [0.4, [0, 0.1, 0.2], [0.85, 0.9, 0.95]], 
                 [0.4, [0.3, 0.4], [0.7, 0.8, 0.9]],
                 [0.4, 0.5, [0.7, 0.75, 0.8]],
                 [0.4, [0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8]], 
                 [0.4, [0.9, 1.], [0.4, 0.5, 0.6, 0.7]], 
                 [0.5, [0, 0.1], [0.85, 0.9, 0.95, 1]], 
                 [0.5, 0.2, [0.85, 0.875, 0.9, 0.925, 0.95]], 
                 [0.5, [0.3, 0.4], [0.7, 0.8, 0.9]], 
                 [0.5, 0.5, [0.7, 0.75, 0.8]], 
                 [0.5, [0.6, 0.7], [0.5, 0.6, 0.7, 0.8]], 
                 [0.5, 0.8, [0.5, 0.55, 0.6, 0.65, 0.7]], 
                 [0.5, [0.9, 1.], [0.3, 0.4, 0.5, 0.6]],
                 [0.6, [0.2, 0.1], [0.9, 1]],
                 [0.6, [0.3, 0.4], [0.8, 0.9, 1]],
                 [0.6, 0.5, [0.7, 0.75, 0.8]], 
                 [0.6, 0.6, [0.6, 0.7, 0.8]], 
                 [0.6, 0.7, [0.4, 0.5, 0.6, 0.7]], 
                 [0.6, 0.8, [0.4, 0.5, 0.6, 0.3]], 
                 [0.6, [0.9, 1.], [0.2, 0.3, 0.4, 0.5, 0.6]],
                 [0.7, 0.2, [0.9, 1]],
                 [0.7, [0.3, 0.4], [0.8, 0.9, 1]],
                 [0.7, 0.5, [0.7, 0.75, 0.8]], 
                 [0.7, 0.6, [0.6, 0.7, 0.5]], 
                 [0.7, 0.7, [0.4, 0.5, 0.6, 0.3]], 
                 [0.7, 0.8, [0.4, 0.5, 0.2, 0.3]], 
                 [0.7, [0.9, 1.], [0.2, 0.3, 0.4, 0., 0.1]],
                 [0.8, [0.3, 0.4], [0.8, 0.9, 1]],
                 [0.8, 0.5, [0.7, 0.75, 0.8]], 
                 [0.8, 0.6, [0.6, 0.7, 0.5]], 
                 [0.8, 0.7, [0.2, 0.4, 0.5, 0.6, 0.3]], 
                 [0.8, 0.8, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]], 
                 [0.8, [0.9, 1.], [0.2, 0.3, 0., 0.1]],
                 [0.9, 0.4, [0.9, 1]],
                 [0.9, 0.5, [0.7, 0.75, 0.8]], 
                 [0.9, 0.55, [0.4, 0.45, 0.5, 0.55]], 
                 [0.9, 0.6, [0.6, 0.3, 0.4, 0.5]], 
                 [0.9, 0.7, [0.2, 0.4, 0.5, 0.3]], 
                 [0.9, 0.8, [0.1, 0., 0.2, 0.3]], 
                 [0.9, [0.9, 1.], [0.2, 0., 0.1]],
                 [0.95, 0.4, [0.9, 1]],
                 [0.95, 0.45, [0.8, 0.9, 1]],
                 [0.95, 0.5, [0.7, 0.75, 0.8]], 
                 [0.95, [0.55, 0.6], [0.4, 0., 0.1, 0.2, 0.3]], 
                 [0.95, [0.7, 0.8], [0.2, 0., 0.1]],
                 [1., [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7], 0.5]
                 ]
    
    rho_init = 0.7947289140762432
    steps = 1000
    saving_multiplier = 10
    reps = 1
elif what == "chosen_grid_short2":
    foldername = "outputs/wiki2-s1000-triads"
    
    sim_sets = [[0, 0.5, 1.], 
                [0.1, list(np.arange(0, 1.01, 0.1)), [0.9, 1.]], 
                 [0.2, list(np.arange(0, 1.01, 0.1)), 1.], 
                 [0.2, [0.9, 1.], 0.6],
                 [0.3, list(np.arange(0, 1.01, 0.1)), 1.],
                 [0.3, list(np.arange(0.6, 1.01, 0.1)), 0.9],
                 [0.4, list(np.arange(0, 1.01, 0.1)), 1.],
                 [0.4, [0.3, 0.4], [0.75, 0.85, 0.95]],
                 [0.4, 0.5, [0.9, 0.85, 0.95]],
                 [0.4, [0.6, 0.7, 0.8], [0.75, 0.9, 0.95, 0.85]], 
                 [0.4, [0.9, 1.], [0.95, 0.9, 0.85, 0.8, 0.75]],
                 [0.5, 0.2, 1.], 
                 [0.5, [0.3, 0.4], [0.75, 0.85, 0.95]], 
                 [0.5, 0.5, [0.9, 0.95, 0.85]], 
                 [0.5, [0.6, 0.7], [0.75, 0.9, 0.95, 0.85]], 
                 [0.5, 0.8, [0.75, 0.85, 0.8, 0.95, 0.9]], 
                 [0.5, [0.9, 1.], [0.9, 0.8, 0.7]],
                 [0.6, 0.5, [1., 0.9]], 
                 [0.6, 0.6, [0.9, 0.75, 1.]], 
                 [0.6, 0.7, [0.9, 0.8, 0.75]], 
                 [0.6, 0.8, [0.7, 0.75, 0.8, 0.9]], 
                 [0.6, [0.9, 1.], [0.7, 0.75, 0.8]],
                 [0.7, 0.5, [0.9, 1.]], 
                 [0.7, 0.6, [0.8, 0.75, 0.9]], 
                 [0.7, 0.7, [0.75, 0.8, 0.7]], 
                 [0.7, 0.8, [0.75, 0.8, 0.7, 0.6]], 
                 [0.7, [0.9, 1.], [0.7, 0.6, 0.5]], 
                 [0.7, [0.7, 0.8, 0.9, 1.], [0.35, 0.45]],
                 [0.8, 0.5, [0.9, 1.]], 
                 [0.8, 0.6, [0.8, 0.75, 0.9, 1.]], 
                 [0.8, 0.7, [0.7, 0.8, 0.75, 0.9]], 
                 [0.8, 0.8, [0.5, 0.6, 0.7, 0.75]], 
                 [0.8, [0.9, 1.], [0.25, 0.35, 0.4, 0.5, 0.6, 0.7]],
                 [0.9, 0.5, [0.9, 1.]], 
                 [0.9, 0.55, [0.6, 0.75, 0.7, 0.8]], 
                 [0.9, 0.6, [0.8, 0.7, 0.75]], 
                 [0.9, 0.7, [0.6, 0.7, 0.75]], 
                 [0.9, 0.8, [0.7, 0.6, 0.5, 0.4]], 
                 [0.9, [0.9, 1.], [0.3, 0.4, 0.5, 0.6, 0.7]],
                 [0.9, [0.6, 0.7, 0.8, 0.9, 1.], [0.45, 0.55]],
                 [0.95, [0.55, 0.6], [0.5, 0.6, 0.7, 0.75]], 
                 [0.95, [0.7, 0.8], [0.3, 0.4, 0.5, 0.6, 0.7]],
                 [0.95, [0.9, 1.], [0.3, 0.4, 0.5, 0., 0.1, 0.2]]
                 ]
    
    rho_init = 0.7947289140762432
    steps = 1000
    saving_multiplier = 10
    reps = 1
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
        q, ps, ph = sim_set
        if not isinstance(q, (list,np.ndarray)):
            q = [q]
        if not isinstance(ps, (list,np.ndarray)):
            ps = [ps]
        if not isinstance(ph, (list,np.ndarray)):
            ph = [ph]
        
        for q_ in q:
            for ps_ in ps:
                for ph_ in ph:
                    exp_sets.append([q_, ps_, ph_])
    
    return exp_sets

def prepare_args2(dataset, network, single_sim_set, rho_init, steps, saving_multiplier, reps, no_triad_stats, foldername):
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
    
    args.rho_init = [rho_init]
    args.steps = steps
    args.saving_multiplier = saving_multiplier
    args.repetitions = reps
    
    args.no_triad_stats = no_triad_stats
    
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
    

if what in ["chosen_grid_short", "chosen_grid_short2"]:
    exp_sets = explode_sims(sim_sets)
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, steps, saving_multiplier, reps, False, foldername) for single_sim_set in exp_sets]
elif what in ["change_q", "n100", "n32long", "n32long2", "n100long", "n100long2"]:
    all_args = [prepare_args(N, steps, reps, psb, ph, q, rho_init, foldername) for psb, ph, q in itertools.product(psb_s, ph_s, q_s)]
# all_experiments = [EXPERIMENTS['LtdStatus'](args_here) for args_here in all_args]

with multiprocessing.Pool(processes=NUM_PROC) as pool:
    pool.map(create_and_run_one_exp, all_args)
    
