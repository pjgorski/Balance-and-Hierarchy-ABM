# This file was used to create simulations of ABM on Slashdot dataset. 
# Simulations were longer (2000 steps) and more dense. 

import argparse
import sys
import itertools

import multiprocessing

import numpy as np

from experiments import EXPERIMENTS
from networktools.network_generators import generator

NUM_PROC = 30

dataset = "Slashdot"
wiki_fname = "../data/slashdot/slashdot_triads.h5"

try:
    network = generator[dataset]()
except (FileNotFoundError, KeyError):
    dataset = wiki_fname
    network = generator['GeneralRealNetwork'](dataset)

what = "rho_inits_varying"
# what = "test_run"

if what == "rho_inits_varying":
    foldername = "outputs/slash-rhoinits-triads-s2000"
    
    sim_sets = [(0.85, 0.0, 1.0, 0.9), (0.85, 0.1, 0.8, 0.8), (0.85, 0.1, 0.9, 0.4), 
                (0.85, 0.1, 1.0, 0.2), (0.85, 0.2, 0.9, 0.6), (0.875, 0.1, 0.9, 0.9), 
                (0.875, 0.1, 1.0, 0.6), (0.875, 0.3, 0.9, 0.0), (0.9, 0.0, 0.8, 0.9), 
                (0.9, 0.0, 0.9, 0.0), (0.9, 0.0, 0.9, 0.8), (0.9, 0.0, 0.9, 0.9), 
                (0.9, 0.2, 0.7, 0.9), (0.9, 0.2, 1.0, 0.6), (0.9, 0.4, 0.6, 0.9), 
                (0.9, 0.4, 0.8, 0.0), (0.9, 0.4, 1.0, 0.0), (0.9, 0.4, 1.0, 0.2), 
                (0.9, 0.4, 1.0, 0.4), (0.9, 0.5, 0.8, 0.0), (0.9, 0.5, 1.0, 0.0), 
                (0.9, 0.6, 1.0, 0.0), (0.9, 0.7, 0.6, 0.6), (0.9, 0.7, 0.7, 0.2), 
                (0.9, 0.8, 0.8, 0.0), (0.9, 0.8, 0.8, 0.2), (0.9, 0.9, 0.5, 0.0), 
                (0.9, 0.9, 0.5, 0.4), (0.9, 1.0, 0.5, 0.4), (0.9, 1.0, 0.6, 0.4), 
                (0.9, 1.0, 0.7, 0.0), (0.9, 1.0, 0.7, 0.2), (0.9, 1.0, 1.0, 0.0), 
                (0.925, 0.0, 0.8, 0.8), (0.925, 0.0, 0.8, 0.9), (0.925, 0.0, 0.9, 0.6), 
                (0.925, 0.1, 0.7, 0.9), (0.925, 0.1, 0.8, 0.9), (0.925, 0.1, 0.9, 0.2), 
                (0.925, 0.1, 1.0, 0.2), (0.925, 0.1, 1.0, 0.4), (0.925, 0.2, 0.8, 0.4), 
                (0.925, 0.2, 0.9, 0.2), (0.925, 0.2, 1.0, 0.2), (0.925, 0.2, 1.0, 0.4), 
                (0.925, 0.2, 1.0, 0.6), (0.925, 0.3, 0.8, 0.0), (0.925, 0.3, 0.8, 0.2), 
                (0.925, 0.3, 0.8, 0.8), (0.925, 0.3, 0.9, 0.2), (0.925, 0.3, 0.9, 0.4), 
                (0.925, 0.3, 1.0, 0.2), (0.925, 0.3, 1.0, 0.4), (0.95, 0.0, 0.7, 0.9), 
                (0.95, 0.0, 0.8, 0.9), (0.95, 0.0, 1.0, 0.0), (0.95, 0.0, 1.0, 0.2), 
                (0.95, 0.1, 0.7, 0.0), (0.95, 0.1, 0.7, 0.2), (0.95, 0.1, 0.7, 0.8), 
                (0.95, 0.1, 0.7, 0.9), (0.95, 0.1, 0.8, 0.0), (0.95, 0.1, 0.8, 0.2), 
                (0.95, 0.1, 1.0, 0.0), (0.95, 0.1, 1.0, 0.2), (0.95, 0.1, 1.0, 0.4), 
                (0.95, 0.1, 1.0, 0.6), (0.95, 0.2, 0.8, 0.0), (0.95, 0.2, 0.8, 0.2), 
                (0.95, 0.2, 0.8, 0.4), (0.95, 0.2, 0.9, 0.2), (0.95, 0.2, 0.9, 0.4), 
                (0.95, 0.2, 0.9, 0.6), (0.95, 0.2, 1.0, 0.0), (0.95, 0.2, 1.0, 0.2), 
                (0.95, 0.2, 1.0, 0.6), (0.95, 0.3, 0.6, 0.0), (0.95, 0.3, 0.7, 0.2), 
                (0.95, 0.3, 0.8, 0.0), (0.95, 0.3, 0.9, 0.2), (0.95, 0.3, 0.9, 0.4), 
                (0.95, 0.3, 1.0, 0.0), (0.95, 0.3, 1.0, 0.4), (0.95, 0.3, 1.0, 0.6), 
                ]
    
    steps = 2000
    saving_multiplier = 20
    reps = 1
    control_initial_rho = True
    keep_rho_at = [0.67, 0.9, 0.65, 0.92]
    silent = True
elif what == "test_run":
    foldername = "outputs/slash-triads-test"
    
    sim_sets = [(0.9000000000000001, 0.3, 0.8, 0.0), (0.9000000000000001, 0.3, 0.8, 0.2), (0.9000000000000001, 0.3, 0.8, 0.4), 
                (0.9000000000000001, 0.3, 0.8, 0.6), (0.9000000000000001, 0.3, 0.8, 0.8), (0.9000000000000001, 0.3, 0.9000000000000001, 0.0), 
                (0.9000000000000001, 0.3, 0.9000000000000001, 0.2), (0.9000000000000001, 0.3, 0.9000000000000001, 0.4), 
                (0.9000000000000001, 0.3, 0.9000000000000001, 0.8), (0.9000000000000001, 0.3, 1.0, 0.0), 
                (0.9000000000000001, 0.3, 1.0, 0.2), (0.9000000000000001, 0.3, 1.0, 0.4), (0.9000000000000001, 0.3, 1.0, 0.6)
                 ]
    
    # rho_inits = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
    steps = 100
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    keep_rho_at = [0.77, 0.97, 0.75, 0.98]

# rho_init_s = [0.6, 0.8, 0.9]

def explode_sims2(sim_sets):
    """ Returns simulation sets in the form of single list. 
    The input are sets with nested lists. 
    
    Difference with `explode_sims` is in the order of parameters in single sim_set. 
    """
    exp_sets = []
    
    for sim_set in sim_sets:
        if len(sim_set) == 3:
            q, ph, ps = sim_set
            rho_init = []
        elif len(sim_set) == 4:
            q, ph, ps, rho_init = sim_set
        if not isinstance(q, (list,np.ndarray)):
            q = [q]
        if not isinstance(ph, (list,np.ndarray)):
            ph = [ph]
        if not isinstance(ps, (list,np.ndarray)):
            ps = [ps]
        if not isinstance(rho_init, (list,np.ndarray)):
            rho_init = [rho_init]
        
        for q_ in q:
            q_ = round(q_, 5)
            for ps_ in ps:
                ps_ = round(ps_, 5)
                for ph_ in ph:
                    ph_ = round(ph_, 5)
                    if len(rho_init) > 0:
                        for rho_init_ in rho_init:
                            rho_init_ = round(rho_init_, 5)
                            exp_sets.append([q_, ph_, ps_, rho_init_])
                    else:
                        exp_sets.append([q_, ph_, ps_])
    
    return exp_sets

def explode_sims_and_get_rhos(sim_sets):
    """ Returns simulation sets in the form of single list. 
    Order of parameters should be as follows: q, ph, ps, rho_init.
    
    The input are sets with nested lists. 
    
    The output are two lists. One containing [q, ph, ps] and the other lists of rho_inits
    that correspond to proper parameter set. 
    """
    exp_sets = []
    rhoinit_dict = {}
    
    for sim_set in sim_sets:
        if len(sim_set) == 4:
            q, ph, ps, rho_init = sim_set
        else:
            raise ValueError("Wrong number of parameters.")
        if not isinstance(q, (list,np.ndarray)):
            q = [q]
        if not isinstance(ph, (list,np.ndarray)):
            ph = [ph]
        if not isinstance(ps, (list,np.ndarray)):
            ps = [ps]
        if not isinstance(rho_init, (list,np.ndarray)):
            rho_init = [round(rho_init, 5)]
        
        for q_ in q:
            q_ = round(q_, 5)
            for ps_ in ps:
                ps_ = round(ps_, 5)
                for ph_ in ph:
                    ph_ = round(ph_, 5)
                    exp_sets.append([q_, ph_, ps_])
                    key = (q_, ph_, ps_)
                    if key in rhoinit_dict:
                        rhoinit_dict[key].extend(rho_init)
                    else:
                        rhoinit_dict[key] = rho_init
    
    return exp_sets, rhoinit_dict

def prepare_args2(dataset, network, single_sim_set, 
                  rho_init, steps, saving_multiplier, reps, no_triad_stats, 
                  foldername, keep_rho_at = [], control_initial_rho = False, silent = False):
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
    
    args.probability = [single_sim_set[1]]
    args.q = [single_sim_set[0]]
    args.psprob = [single_sim_set[2]]
    
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


def create_and_run_one_exp(args):
    experiment = EXPERIMENTS['LtdReal'](args)
    experiment()
    experiment.clear()
    del experiment
    pass
    

if what in ["rho_inits_varying", "rho_inits_varying2", "test_run"]:
    exp_sets, rhoinit_dict = explode_sims_and_get_rhos(sim_sets)
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rhoinit_dict[single_sim_set], 
                              steps, saving_multiplier, reps, False, 
                              foldername, keep_rho_at, control_initial_rho) for single_sim_set in rhoinit_dict]



with multiprocessing.Pool(processes=NUM_PROC, maxtasksperchild = 1) as pool:
    r = pool.map_async(create_and_run_one_exp, all_args, chunksize = 1)
    r.wait()
