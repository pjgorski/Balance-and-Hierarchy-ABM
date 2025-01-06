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

outputs_dir = Path("outputs/LtdReal/outputs/spanish-highschools")

print("Starting reading folders")

in_files = []

for school in specific_schools:
    fpath = outputs_dir.joinpath("df_rho_" + school + ".h5")
    
    df_files = pd.read_hdf(Path(fpath), key = "df_files")
    
    fpath = outputs_dir.joinpath("df_rho_2_" + school + ".h5")
    
    in_files.append(fpath)
    
    df_rho, df_files, last_file_processed = process_folder(outputs_dir, 
                                  no_triad_stats=False, max_num_rows=1000000, dataset = school, df_files=df_files)
    path = Path(fpath)
    df_rho.to_hdf(path, key = 'df_rho', mode = "w") #this creates a new file, due to the bug in the documentation
    df_files.to_hdf(path, key = 'df_files')
    
    # print("Processed " + str(last_file_processed) + " out of " + str(len(df_files)))

print("Starting calculating QS values")

# in_files = [outputs_dir.joinpath("df_rho" + str(id_) + ".h5") for id_ in range(1,id)]
out_files = [outputs_dir.joinpath("df2_rho_2_" + school + ".h5") for school in specific_schools]

# out_file = [outputs_dir.joinpath("df2_rho.h5")]

get_quasilevels_from_multiple_files(in_files, out_files, merge_to_one = False, drop_list_columns = False, leave_correct=False)


print("Grouping the results")

out_files_group = []
group_quasilevels_of_multiple_files
# not finished
# TODO: automatic next filename detection. So we do not have to increment names. 