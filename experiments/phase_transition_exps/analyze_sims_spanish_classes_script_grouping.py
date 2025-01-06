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
check_old_int = 1

if len(sys.argv) > 1:
    if sys.argv[1].isnumeric():
        check_old_int = int(sys.argv[1])
    else:
        outputs_dir = Path(sys.argv[1])
        
        if len(sys.argv) > 2:
            check_old_int = int(sys.argv[2])


if outputs_dir is None:
    outputs_dir = Path("outputs/LtdReal/outputs/spanish-highschools-classes")

print("Summary of input parameters.")
print("Files are in folder: " + str(outputs_dir))
if check_old_int <= 0:
    ValueError(check_old_int, "Error! Previous partial result analysis are needed. check_old_int should be positive.")
else:
    if check_old_int == 1:
        print("Analysis of previous simulations resulted in output files of the form of 'df_rho' etc.")
    else:
        print("Analysis of previous simulations resulted in output files of the form of 'df_rho_" + str(check_old_int) + "' etc.")


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

print("Grouping the results")

in_files_group = []
out_files_group = []
for school in datanames:
    i = 1
    in_files = []
    while i <= max(check_old_int, 1):
        if i == 1:
            inter_string = ""
        else:
            inter_string = str(i) + "_"
        fpath = outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df2_rho_" + inter_string + school + ".h5")
        if os.path.exists(fpath):
            in_files.append(fpath)
        else:
            print(str(fpath) + " does not exist.", flush = True)
        i += 1
    if len(in_files) > 0:
        in_files_group.append(in_files)
        out_files_group.append(outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df2_rho_g_" + new_filenames_num + school + ".h5"))


# out_files_group = [outputs_dir.parent.joinpath(outputs_dir.name + "-processed").joinpath("df2_rho_g_" + new_filenames_num + school + ".h5") for school in datanames]
group_quasilevels_of_multiple_files(in_files_group, out_files_group, only_correct_rows=False, remove_nan_rows=True)

