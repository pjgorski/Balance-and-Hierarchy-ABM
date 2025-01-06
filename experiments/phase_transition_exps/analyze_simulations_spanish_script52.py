"""
File to use due to error in calc of QS levels. Probably one time usage. 

This file version uses get_quasilevels2 to obtain quasi levels. It goes through all the files and saves them to a different folder. 

Possible input parameters:
outputs_dir (string); position: 1st or 2nd; folder where to input data is
    Deafult: outputs/LtdReal/outputs/spanish-highschools
check_old_int (int); position: 1-3; Reads the previously generated files and 
    works on new results. 
    Default: -1, which means that no previous files are read and all the results 
    are included. 
check_if_outfiles_exist (bool); position: 1-3; If True, then on the first stage 
    (when reasing sim results) it is checked if the results were already processed and 
    saved. In that case this is not done again. 
"""

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
check_if_outfiles_exist = False

if len(sys.argv) > 1:
    if sys.argv[1].isnumeric():
        check_old_int = int(sys.argv[1])
    elif sys.argv[1] in ["True", "False", "true", "false", "T", "t", "F", "f"]:
        if sys.argv[1] in ["True", "true", "t", "T"]:
            check_if_outfiles_exist = True
    else:
        outputs_dir = Path(sys.argv[1])
        
        if len(sys.argv) > 2:
            if sys.argv[2].isnumeric():
                check_old_int = int(sys.argv[2])
            else:
                if sys.argv[2] in ["True", "true", "t", "T"]:
                    check_if_outfiles_exist = True
            
            if len(sys.argv) > 3:
                if sys.argv[3].isnumeric():
                    check_old_int = int(sys.argv[2])
                else:
                    if sys.argv[3] in ["True", "true", "t", "T"]:
                        check_if_outfiles_exist = True


if outputs_dir is None:
    outputs_dir = Path("outputs/LtdReal/outputs/spanish-highschools")

print("Summary of input parameters.")
print("Files are in folder: " + str(outputs_dir))
if check_old_int == -1:
    print("We do not look whether there were some partial results. All folders are checked.")
else:
    if check_old_int == 2:
        print("Only new folders are analyzed. Analysis of previous simulations resulted in output files of the form of 'df_rho' etc.")
    else:
        print("Only new folders are analyzed. Analysis of previous simulations resulted in output files of the form of 'df_rho_" + str(check_old_int) + "' etc.")
if check_if_outfiles_exist:
    print("Output files are checked whether they exist and if they do, they are not generated once again.")


old_filenames_num = ""
new_filenames_num = ""
if check_old_int != -1:
    new_filenames_num = str(check_old_int) + "_"
    if check_old_int > 2:
        old_filenames_num = str(check_old_int - 1) + "_"

print("Starting reading folders", flush=True)

in_files = []
schools_to_keep = []

outputs_dir.parent.joinpath(outputs_dir.name + "-processed").mkdir(exist_ok=True)
for id, school in enumerate(specific_schools):
    if check_old_int != -1:
        fpath = outputs_dir.joinpath("df_rho_" + old_filenames_num + school + ".h5")
        df_files = pd.read_hdf(Path(fpath), key = "df_files")
    else:
        df_files = None
    
    fpath = outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df_rho_" + new_filenames_num + school + ".h5")
    # fpath = outputs_dir.joinpath("df_rho_" + new_filenames_num + school + ".h5")
    
    file_created = False
    if check_if_outfiles_exist & os.path.isfile(fpath):
        df_rho = pd.read_hdf(Path(fpath), key = "df_rho")
    else:
        df_rho, df_files, last_file_processed = process_folder(outputs_dir, 
                                  no_triad_stats=False, max_num_rows=1000000, 
                                  dataset = school, df_files=df_files)
        file_created = True
    
    if df_rho is None: 
        # it means that no files were processed:
        print("No files for: " + school, flush=True)
        continue
    else:
        schools_to_keep.append(id)
        in_files.append(fpath)
    
    if file_created:
        path = Path(fpath)
        df_rho.to_hdf(path, key = 'df_rho', mode = "w") #this creates a new file, due to the bug in the documentation
        df_files.to_hdf(path, key = 'df_files')
    
    # print("Processed " + str(last_file_processed) + " out of " + str(len(df_files)))

specific_schools = [school for id, school in enumerate(specific_schools) if id in schools_to_keep]

print("Starting calculating QS values", flush=True)

# in_files = [outputs_dir.joinpath("df_rho" + str(id_) + ".h5") for id_ in range(1,id)]
# out_files = [outputs_dir.joinpath("df2_rho_" + new_filenames_num + school + ".h5") for school in specific_schools]
out_files = [outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df2_rho_" + new_filenames_num + school + ".h5") for school in specific_schools]

# out_file = [outputs_dir.joinpath("df2_rho.h5")]


def get_quasilevels2_early20_2(df, calc_triads = True, drop_list_columns = False, leave_correct = False):
    return get_quasilevels2(df, calc_triads, drop_list_columns, leave_correct, 20, True)


get_quasilevels_from_multiple_files(in_files, out_files, merge_to_one = False, 
                                    drop_list_columns = False, leave_correct=False, 
                                    get_quasilevels_fun = get_quasilevels2_early20_2,
                                    check_if_outfiles_exist = check_if_outfiles_exist,
                                    use_existing_qs_levels = True)


print("Grouping the results", flush=True)

in_files_group = []
for school in specific_schools:
    i = 1
    in_files = []
    while i <= max(check_old_int, 1):
        if i == 1:
            inter_string = ""
        else:
            inter_string = str(i) + "_"
        fpath = outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df2_rho_" + inter_string + school + ".h5")
        # fpath = outputs_dir.joinpath("df2_rho_" + inter_string + school + ".h5")
        in_files.append(fpath)
        i += 1
    in_files_group.append(in_files)

out_files_group = [outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df2_rho_g_" + new_filenames_num + school + ".h5") for school in specific_schools]
# out_files_group = [outputs_dir.joinpath("df2_rho_correct_g_" + new_filenames_num + school + ".h5") for school in specific_schools]
group_quasilevels_of_multiple_files(in_files_group, out_files_group, only_correct_rows=False, remove_nan_rows=True)
