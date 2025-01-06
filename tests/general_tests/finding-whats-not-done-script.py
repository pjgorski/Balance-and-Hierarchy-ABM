import argparse
import sys
import itertools

import numpy as np

import os
import sys

import copy

from pathlib import Path
import pandas as pd

sys.path.append('../../experiments/phase_transition_exps')
from analyze_simulations_funs import *

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
# os.chdir("../..") 


# from experiments import EXPERIMENTS
# from networktools.network_generators import generator

if len(sys.argv[1]) > 1:
    # to_path = sys.argv[1]
    # if 
    outputs_dir = Path(sys.argv[1])
else:
    outputs_dir = Path("../../outputs/LtdReal/outputs/sampson-rhoinits-triads")
    
print("Starting reading folders")

df_rho, df_files, last_file_processed = process_folder(outputs_dir, 
        no_triad_stats=True, max_num_rows=1000000, no_arrays = True)

fpath = outputs_dir.joinpath("df_pars.h5")
path = Path(fpath)
df_rho.to_hdf(path, key = 'df_rho', mode = "w") #this creates a new file, due to the bug in the documentation
df_files.to_hdf(path, key = 'df_files')

raise Exception("File not finished. ")

