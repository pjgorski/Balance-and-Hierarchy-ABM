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
num_input_files = int(sys.argv[2])


print("Starting calculating QS values")

in_files = [outputs_dir.joinpath("df_rho" + str(id_) + ".h5") for id_ in range(1,num_input_files+1)]
# out_files = [outputs_dir.join("df2_rho" + str(id_) + ".h5") for id_ in range(1,id+1)]

out_file = [outputs_dir.joinpath("df2_rho.h5")]
intermediate_files = [outputs_dir.joinpath("df2_rho" + str(id_) + ".h5") for id_ in range(1,num_input_files+1)]

get_quasilevels_from_multiple_files(in_files, out_file, merge_to_one = True, drop_list_columns = True, leave_correct=True, save_intermediate_steps = intermediate_files)


