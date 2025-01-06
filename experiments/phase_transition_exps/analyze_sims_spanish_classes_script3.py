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

specific_schools = ["t11_10", "t11_9", "t11_8", "t11_7", "t11_6", "t11_5", "t11_4", "t11_3", "t11_2", "t11_1", "t1", "t2", "t6"]

if len(sys.argv) > 1:
    outputs_dir = Path(sys.argv[1])
else:
    outputs_dir = Path("outputs/LtdReal/outputs/spanish-highschools-classes")

dataset_parent_folder = "../data/spanish-highschools"
networks = []
datanames = []
for school in specific_schools:
    i = 1
    while True:
        if not os.path.isfile(os.path.join(dataset_parent_folder, "Triads_" + school + "_" + str(i) + ".csv")):
            break
        datanames.append(school + "_" + str(i))
        i += 1

print("Starting reading folders")

in_files = []

for school in datanames:
    fpath = outputs_dir.joinpath("df_rho_" + school + ".h5")
    df_files = pd.read_hdf(Path(fpath), key = "df_files")
    
    fpath = outputs_dir.joinpath("df_rho_2_" + school + ".h5")
    in_files.append(fpath)
    
    df_rho, df_files, last_file_processed = process_folder(outputs_dir, 
                                  no_triad_stats=False, max_num_rows=1000000, dataset = school, 
                                  df_files=df_files, external_dataset_stats_file = "results_realnet_spanish_class.h5")
    path = Path(fpath)
    df_rho.to_hdf(path, key = 'df_rho', mode = "w") #this creates a new file, due to the bug in the documentation
    df_files.to_hdf(path, key = 'df_files')
    
    # print("Processed " + str(last_file_processed) + " out of " + str(len(df_files)))

print("Starting calculating QS values")

# in_files = [outputs_dir.joinpath("df_rho" + str(id_) + ".h5") for id_ in range(1,id)]
out_files = [outputs_dir.joinpath("df2_rho_2_" + school + ".h5") for school in datanames]

# out_file = [outputs_dir.joinpath("df2_rho.h5")]

get_quasilevels_from_multiple_files(in_files, out_files, merge_to_one = False, drop_list_columns = False, leave_correct=False)


