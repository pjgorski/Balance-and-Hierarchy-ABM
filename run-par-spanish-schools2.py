# This file was used to create simulations of ABM on Spanish high school dataset. 
# This file was to be used on schools that are split into separate networks. 
# See the file run-par-2.py for the description.
# 
# Description of new parameters:
# - specific_schools: list of school names to be used in the simulation
# - sim_sets: a dict with schools as keys and values defined same as previously (see run-par-2.py)
"""
This script performs simulations for specific classes of spanish high schools. 
"""

import argparse
import sys
import itertools
import os

import multiprocessing

import numpy as np

from experiments import EXPERIMENTS
from networktools.network_generators import generator

NUM_PROC = 50

specific_schools = ["t11_10", "t11_9", "t11_8", "t11_7", "t11_6", "t11_5", "t11_4", "t11_3", "t11_2", "t11_1", "t1", "t2", "t6"]

dataset_parent_folder = "/home/pgorski/Desktop/data/spanish-highschools"
rel_parent_folder = "../data/spanish-highschools"


if not os.path.exists(dataset_parent_folder):
    dataset_parent_folder = rel_parent_folder


networks = []
datanames = []
for school in specific_schools:
    i = 1
    while True:
        if not os.path.isfile(os.path.join(dataset_parent_folder, "Triads_" + school + "_" + str(i) + ".csv")):
            break
        network = generator['GeneralRealNetwork'](os.path.join(dataset_parent_folder, "Triads_" + school + "_" + str(i) + ".csv"))
        networks.append(network)
        datanames.append(school + "_" + str(i))
        
        i += 1


# get rhos to calculate proper `keep_rho_at`
rhos = [np.sum(network.adm == 1) / np.sum(network.adm != 0) for network in networks]

what = "rho_inits"
what = "rho_inits_dense_controldown"
# what = "rho_inits_dense"
# what = "test_run"

if what == "rho_inits":
    foldername = "outputs/spanish-highschools-classes"
    
    sim_sets = [[list(np.arange(0.1, 0.95, 0.1)), list(np.arange(0, 1.01, 0.1)), list(np.arange(0, 1.01, 0.1))], 
                [1.0, list(np.arange(0,1.01, 0.1)), 0.5], [0.0, 0.5, list(np.arange(0,1.01, 0.1))]
                 ]
    
    rho_inits = [0, 0.5, 0.9]
    steps = 1000
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    # keep_rho_at = [0.82, 0.97, 0.8, 0.98]
    silent = True
elif what == "rho_inits_controldown":
    foldername = "outputs/spanish-highschools-classes-controldown"
    
    sim_sets = [[list(np.arange(0.1, 0.95, 0.1)), list(np.arange(0, 1.01, 0.1)), list(np.arange(0, 1.01, 0.1))], 
                [1.0, list(np.arange(0,1.01, 0.1)), 0.5], [0.0, 0.5, list(np.arange(0,1.01, 0.1))]
                 ]
    
    rho_inits = [0, 0.5, 0.9]
    steps = 1000
    saving_multiplier = 10
    reps = 1
    # control_initial_rho = True
    # keep_rho_at = [0.82, 0.97, 0.8, 0.98]
    
    def control_initial_rho(data_rho):
        return [data_rho - 0.15, 1, data_rho - 0.17, 1.2]
    
    silent = True
elif what == "rho_inits_dense":
    foldername = "outputs/spanish-highschools-classes"
    
    sim_sets = {"t11_1_1": [[list(np.arange(0.6, 0.901, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_1_2": [[list(np.arange(0.6, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_1": [[list(np.arange(0.7, 0.9, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_3": [[list(np.arange(0.7, 0.9, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_4": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_3_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_3_2": [[list(np.arange(0.6, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_1": [[list(np.arange(0.4, 0.775, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_2": [[list(np.arange(0.4, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_3": [[list(np.arange(0.4, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_4": [[list(np.arange(0.2, 0.7, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_1": [[list(np.arange(0.6, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_2": [[list(np.arange(0.6, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_3": [[list(np.arange(0.6, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_4": [[list(np.arange(0.7, 0.9, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_2": [[list(np.arange(0.6, 0.9, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_3": [[list(np.arange(0.1, 0.925, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_4": [[list(np.arange(0.1, 0.8, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_1": [[list(np.arange(0.6, 0.9, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_3": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_4": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_3": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_4": [[list(np.arange(0.75, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
    }
    
    rho_inits = [0, 0.5, 0.9]
    steps = 1000
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    # keep_rho_at = [0.82, 0.97, 0.8, 0.98]
    silent = True
elif what == "rho_inits_dense_controldown":
    foldername = "outputs/spanish-highschools-classes-controldown"
    
    sim_sets = {"t11_1_1": [[list(np.arange(0.7, 0.901, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_1_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_1": [[list(np.arange(0.7, 0.93, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_3": [[list(np.arange(0.7, 0.91, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_2_4": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_3_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_3_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_1": [[list(np.arange(0.5, 0.775, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_2": [[list(np.arange(0.5, 0.8, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_3": [[list(np.arange(0., 0.23, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_4_4": [[list(np.arange(0.1, 0.7, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_1": [[list(np.arange(0.7, 0.93, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_2": [[list(np.arange(0.7, 0.975, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_3": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_5_4": [[list(np.arange(0.6, 0.8, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_2": [[list(np.arange(0.6, 0.9, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_3": [[list(np.arange(0.4, 0.95, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_8_4": [[list(np.arange(0., 0.6, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_1": [[list(np.arange(0.3, 0.9, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_2": [[list(np.arange(0.7, 0.9, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_3": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_9_4": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_3": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
                "t11_10_4": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
                 ],
    }
    
    sim_sets = {
                # x--"t11_1_1": [[list(np.arange(0.7, 0.901, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_1_2": [[list(np.arange(0.975, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    #             x-"t11_2_1": [[list(np.arange(0.7, 0.93, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_2_2": [[list(np.arange(0.95, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_2_3": [[list(np.arange(0.925, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_2_4": [[list(np.arange(0.95, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_3_1": [[list(np.arange(0.975, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    #             x--"t11_3_2": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_4_1": [[list(np.arange(0.525, 0.725, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_4_2": [[list(np.arange(0.625, 0.8, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_4_3": [[list(np.arange(0.25, 0.375, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_4_4": [[list(np.arange(0.325, 0.525, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_5_1": [[list(np.arange(0.75, 0.95, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_5_2": [[list(np.arange(0.825, 0.95, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    #             x--"t11_5_3": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_5_4": [[list(np.arange(0.825, 0.95, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    #             x-"t11_8_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_8_2": [[list(np.arange(0.725, 0.85, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_8_3": [[list(np.arange(0.675, 0.9, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    #             x-"t11_8_4": [[list(np.arange(0., 0.6, 0.05)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_9_1": [[list(np.arange(0.6, 0.9, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_9_2": [[list(np.arange(0.9, 0.96, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_9_3": [[list(np.arange(0.95, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    #             x-"t11_9_4": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
    #             x-"t11_10_1": [[list(np.arange(0.8, 0.975, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
    #             [1.0, 0.5, list(np.arange(0,1.01, 0.1))],
    #              ],
                "t11_10_2": [[list(np.arange(0.975, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],
                "t11_10_3": [[list(np.arange(0.95, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
                "t11_10_4": [[list(np.arange(0.95, 0.99, 0.025)), list(np.arange(0, 1.01, 0.05)), list(np.arange(0, 1.01, 0.05))], 
                 ],#x
    }
    
    rho_inits = [0, 0.5, 0.9]
    steps = 1000
    saving_multiplier = 10
    reps = 1
    # control_initial_rho = True
    # keep_rho_at = [0.82, 0.97, 0.8, 0.98]
    
    def control_initial_rho(data_rho):
        return [data_rho - 0.15, 1, data_rho - 0.17, 1.2]
    
    silent = True
elif what == "test_run":
    foldername = "outputs/spanish-highschools-class-test"
    
    sim_sets = [[list(np.arange(0.1, 0.95, 0.5)), list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5))], 
                # [0.0, list(np.arange(0,1.01, 0.1)), 0.5], [1.0, 0.5, list(np.arange(0,1.01, 0.1))]
                 ]
    
    rho_inits = [0.2, 0.6, 0.9]
    steps = 1
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    # keep_rho_at = [0.67, 0.9, 0.65, 0.92]
    silent = True
elif what == "test_run2":
    foldername = "outputs/spanish-highschools-class-test"
    
    sim_sets = [[list(np.arange(0.1, 0.95, 0.5)), list(np.arange(0, 1.01, 0.5)), list(np.arange(0, 1.01, 0.5))], 
                # [0.0, list(np.arange(0,1.01, 0.1)), 0.5], [1.0, 0.5, list(np.arange(0,1.01, 0.1))]
                 ]
    
    rho_inits = [0.2, 0.6, 0.9]
    steps = 1
    saving_multiplier = 10
    reps = 1
    control_initial_rho = True
    # keep_rho_at = [0.67, 0.9, 0.65, 0.92]
    silent = True


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
                  foldername, keep_rho_at = [], control_initial_rho = False, silent = False, data_rho = -1, keep_average_rho = False):
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
    
    if callable(control_initial_rho):
        assert data_rho != -1
        keep_rho_at = control_initial_rho(data_rho)
        args.control_initial_rho = True
    else:
        if control_initial_rho:
            if len(keep_rho_at) == 0:
                assert data_rho != -1
                keep_rho_at = [data_rho - 0.1, data_rho + 0.1, data_rho - 0.12, data_rho + 0.12]
        
        args.control_initial_rho = control_initial_rho
    
    args.keep_rho_at = keep_rho_at
    args.keep_average_rho = keep_average_rho
    
    args.silent = silent
    
    return args

def create_and_run_one_exp(args):
    experiment = EXPERIMENTS['LtdReal'](args)
    experiment()
    experiment.clear()
    del experiment
    pass
    

if what in ["rho_inits", "rho_inits3", "rho_inits32", "rho_inits4", "rho_inits5", "test_run", "test_run2"]:
    exp_sets = explode_sims2(sim_sets)
    
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, 
                              steps, saving_multiplier, reps, False, 
                              foldername, control_initial_rho = control_initial_rho, silent = silent, data_rho=rho) 
                            for (dataset, network, rho) in zip(datanames, networks, rhos)
                            for single_sim_set in exp_sets]
elif what in ["rho_inits_controldown"]:
    exp_sets = explode_sims2(sim_sets)
    
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, 
                              steps, saving_multiplier, reps, False, 
                              foldername, control_initial_rho = control_initial_rho, silent = silent, data_rho=rho) 
                            for (dataset, network, rho) in zip(datanames, networks, rhos)
                            for single_sim_set in exp_sets]
elif what in ["rho_inits_dense", "rho_inits_dense_controldown"]:
    rho_init = rho_inits
    
    all_args = [prepare_args2(dataset, network, single_sim_set, rho_init, 
                              steps, saving_multiplier, reps, False, 
                              foldername, control_initial_rho = control_initial_rho, silent = silent, data_rho=rho) 
                            for i, (dataset, network, rho) in enumerate(zip(datanames, networks, rhos)) if dataset in sim_sets
                            for single_sim_set in explode_sims2(sim_sets[dataset]) ]
# elif what in ["rho_inits_varying"]:
#     exp_sets, rhoinit_dict = explode_sims_and_get_rhos(sim_sets)
    
#     all_args = [prepare_args2(dataset, network, single_sim_set, rhoinit_dict[single_sim_set], 
#                               steps, saving_multiplier, reps, False, 
#                               foldername, keep_rho_at, control_initial_rho, silent) for single_sim_set in rhoinit_dict]



with multiprocessing.Pool(processes=NUM_PROC) as pool:
    r = pool.map_async(create_and_run_one_exp, all_args)
    r.wait()
    
