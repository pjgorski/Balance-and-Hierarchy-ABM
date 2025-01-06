# File showing how to run sets of simulations of agent-based models
# Parameters of the simulations that can be set:
# - N: number of agents
# - steps: number of steps, where a single step is number of edges in the network
# - reps: number of repetitions with random initial conditions. 
# - psb: p_{SBT} ABM parameter. It may ba a scalar value or a list or numpy.ndarray. 
# - ph: p_{ST} ABM parameter. It may ba a scalar value or a list or numpy.ndarray.
# - q: q ABM parameter. It may ba a scalar value or a list or numpy.ndarray.
# - rho_init: initial density of positive edges. It may ba a scalar value or a list or numpy.ndarray.
# - foldername: name of the folder where the results will be saved.
# 
# Additional parameters:
# - NUM_PROC: number of processes to run in parallel
# - what: name of the set of simulations. It is used to create a custom set of parameters and keep it saved. 

import argparse
import sys
import itertools

import multiprocessing

import numpy as np

from experiments import EXPERIMENTS

NUM_PROC = 20

N = 100
steps = 400
reps = 100

what = "n100_ph"
what = "test_run"

if what == "sepa_0_2":
    foldername = "outputs/sepa_0_2"
    psb_s = np.arange(0.895, 0.923, 0.001)
    # psb_s = [0.3, 0.4, 0.5]
    ph = 0.2
    q = 0.5
    rho_init_s = np.arange(0.80, 1, 0.01)
elif what == "change_q":
    foldername = "outputs/change_q"
    psb_s = [0.1, 0.9]
    ph_s = [0.2, 0.8]
    q_s = np.arange(0,1.01,0.1)
    rho_init = 0.5
elif what == "n100":
    foldername = "outputs/n100"
    # psb_s = np.arange(0,1.01,0.1)
    # psb_s = np.array([0.55, 0.62, 0.64, 0.65, 0.66])
    psb_s = np.array([0.75, 0.82, 0.84, 0.85, 0.87])
    ph_s = [0.3]
    q_s = [0.5,]
    rho_init = 0.5
elif what == "n32long":
    foldername = "outputs/n32_long"
    psb_s = np.arange(0,1.01,0.1)
    ph_s = [0.45, 0.5, 0.55]
    q_s = [0.95,]
    rho_init = 0.5
    steps = 4000
    reps = 10
    N = 32
elif what == "n32long2":
    foldername = "outputs/n32_long"
    psb_s = [0.2, 0.8]
    ph_s = [*np.arange(0,1.01,0.1), 0.45, 0.55, 0.49, 0.51]
    # np.append(ph_s)
    q_s = [0.8, 0.95,]
    rho_init = 0.5
    steps = 4000
    reps = 10
    N = 32
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
elif what == "n100_ph":
    foldername = "outputs/n100_ph"
    psb_s = [0.2, 0.8]
    ph_s = [*np.arange(0,1.01,0.1), 0.45, 0.47, 0.48, 0.49, 0.51, 0.53, 0.55, 0.57, 0.73, 0.75, 0.77]
    # np.append(ph_s)
    q_s = [0.8, 0.95,]
    rho_init = 0.5
    N = 100
elif what == "test_run":
    foldername = "outputs/test-runs"
    psb_s = [0.2, 0.8]
    ph_s = np.arange(0,1.01,0.1)
    q_s = [0.8, 0.95,]
    rho_init = 0.5
    N = 32
    reps = 2
    steps = 100


# rho_init_s = [0.6, 0.8, 0.9]

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
    
    args.silent = True
    
    return args


def create_and_run_one_exp(args):
    experiment = EXPERIMENTS['LtdStatus'](args)
    experiment()
    pass
    

if what == "sepa_0_2":
    all_args = [prepare_args(N, steps, reps, psb, ph, q, rho_init, foldername) for psb, rho_init in itertools.product(psb_s, rho_init_s)]
elif what in ["change_q", "n100", "n32long", "n32long2", "n100long", "n100long2", "n100_ph", "test_run"]:
    all_args = [prepare_args(N, steps, reps, psb, ph, q, rho_init, foldername) for psb, ph, q in itertools.product(psb_s, ph_s, q_s)]
# all_experiments = [EXPERIMENTS['LtdStatus'](args_here) for args_here in all_args]

with multiprocessing.Pool(processes=NUM_PROC) as pool:
    pool.map(create_and_run_one_exp, all_args)
    
