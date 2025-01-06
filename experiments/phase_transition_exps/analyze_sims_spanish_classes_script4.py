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

outputs_dir = None
check_old_int = -1

if len(sys.argv) > 1:
    if sys.argv[1].isnumeric():
        check_old_int = int(sys.argv[1])
    else:
        outputs_dir = Path(sys.argv[1])
        
        if len(sys.argv) > 2:
            check_old_int = int(sys.argv[2])


if outputs_dir is None:
    outputs_dir = Path("outputs/LtdReal/outputs/spanish-highschools-classes")

old_filenames_num = ""
new_filenames_num = ""
if check_old_int != -1:
    new_filenames_num = str(check_old_int) + "_"
    if check_old_int > 2:
        old_filenames_num = str(check_old_int - 1) + "_"

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
classes_to_keep = []

for id, school in enumerate(datanames):
    if check_old_int != -1:
        fpath = outputs_dir.joinpath("df_rho_" + old_filenames_num + school + ".h5")
        df_files = pd.read_hdf(Path(fpath), key = "df_files")
    else:
        df_files = None
    
    fpath = outputs_dir.joinpath("df_rho_" + new_filenames_num + school + ".h5")
    # in_files.append(fpath)
    
    df_rho, df_files, last_file_processed = process_folder(outputs_dir, 
                                  no_triad_stats=False, max_num_rows=1000000, dataset = school, 
                                  df_files=df_files, external_dataset_stats_file = "results_realnet_spanish_class.h5")
    if df_rho is None: 
        # it means that no files were processed:
        print("No files for: " + school)
        continue
    else:
        classes_to_keep.append(id)
        in_files.append(fpath)
    
    path = Path(fpath)
    df_rho.to_hdf(path, key = 'df_rho', mode = "w") #this creates a new file, due to the bug in the documentation
    df_files.to_hdf(path, key = 'df_files')
    
    # print("Processed " + str(last_file_processed) + " out of " + str(len(df_files)))

datanames = [school for id, school in enumerate(datanames) if id in classes_to_keep]

print("Starting calculating QS values")

# in_files = [outputs_dir.joinpath("df_rho" + str(id_) + ".h5") for id_ in range(1,id)]
out_files = [outputs_dir.joinpath("df2_rho_" + new_filenames_num + school + ".h5") for school in datanames]

# out_file = [outputs_dir.joinpath("df2_rho.h5")]

get_quasilevels_from_multiple_files(in_files, out_files, merge_to_one = False, drop_list_columns = False, leave_correct=False)

print("Grouping the results")

in_files_group = []
for school in datanames:
    i = 1
    in_files = []
    while i <= max(check_old_int, 1):
        if i == 1:
            inter_string = ""
        else:
            inter_string = str(i) + "_"
        fpath = outputs_dir.joinpath("df2_rho_" + inter_string + school + ".h5")
        in_files.append(fpath)
        i += 1
    in_files_group.append(in_files)

out_files_group = [outputs_dir.joinpath("df2_rho_correct_g_" + new_filenames_num + school + ".h5") for school in datanames]
group_quasilevels_of_multiple_files(in_files_group, out_files_group)

