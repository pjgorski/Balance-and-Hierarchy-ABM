import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import lines
import pandas as pd
import scipy.special
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

import sys

from analyze_simulations_funs import *

outputs_dir = Path(sys.argv[1])

print("Starting reading folders")

avg_max_num_rows = 10000
fpath = outputs_dir.joinpath("df_rho")
last_file_processed = 0
df_files = [0,0]
id = 1

while last_file_processed < len(df_files) - 1:
    df_rho, df_files, last_file_processed = process_folder(outputs_dir, 
                                  no_triad_stats=False, start_index = last_file_processed, 
                                  max_num_rows=avg_max_num_rows)
    path = Path(str(fpath) + str(id) + ".h5")
    df_rho.to_hdf(path, key = 'df_rho', mode = "w") #this creates a new file, due to the bug in the documentation
    df_files.to_hdf(path, key = 'df_files')
    id += 1
    
    print("Processed " + str(last_file_processed) + " out of " + str(len(df_files)))

print("Starting calculating QS values")

in_files = [outputs_dir.joinpath("df_rho" + str(id_) + ".h5") for id_ in range(1,id)]
# out_files = [outputs_dir.join("df2_rho" + str(id_) + ".h5") for id_ in range(1,id+1)]

out_file = [outputs_dir.joinpath("df2_rho.h5")]

get_quasilevels_from_multiple_files(in_files, out_file, merge_to_one = True, drop_list_columns = True, leave_correct=True)


