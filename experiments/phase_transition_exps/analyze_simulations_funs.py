import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import lines
import pandas as pd
import scipy.special
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf

import warnings

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

import sys
import pandas as pd

def filter_get_qs_pvalue_warnings():
    warnings.filterwarnings("ignore", "kurtosistest only valid for n>=20 ... continuing ")
    warnings.filterwarnings("ignore", "RuntimeWarning: invalid value encountered in double_scalars")
    warnings.filterwarnings("ignore", "RuntimeWarning: divide by zero encountered in log")
    warnings.filterwarnings("ignore", "")
    warnings.filterwarnings("ignore", "")
    warnings.filterwarnings("ignore", "")


"""These functions present analytical solutions for the model. 
The parameter `label` can be one of the two values: Adam or Piotr. 
When it is Adam the `q` parameter is as it is in the paper, that is
it is the probability of the status dynamics. 
"""

def get_pb_crit(q, ph, label="adam"):
    """
    This function doesn't check if critical value exists. One needs to check beforehand. 
    """
    if label == "adam":
        q = 1-q
    return ((2*q + (1-q)*(1-2*ph))**2 / (8*q**2) + 1)/2

def get_rho_crit(q,ph, label="adam"):
    """
    This function doesn't check if critical value exists. One needs to check beforehand. 
    Basic assumption: pb_crit>0.5
    """
    if label == "adam":
        q = 1-q
    return 2*q/(2*q + (1-q)*(1-2*ph))

def get_q_crit(pb, ph, label="adam"):
    """
    This function doesn't check if critical value exists. One needs to check beforehand. 
    Basic assumption: pb_crit>0.5
    """
    if pb < 0.5:
        return np.NaN
    rho_cr = 1/2/np.sqrt(pb-0.5)
    q_cr = 2*(1/rho_cr-1)/(2/rho_cr-2*ph-1)
    return q_cr

def get_pb_when_rho_eq_1(q, ph, label="adam"):
    """This function doesn't check if such pb exists. 

    Args:
        q (_type_): _description_
        ph (_type_): _description_
        label (str, optional): _description_. Defaults to "adam".
    """
    if label == "adam":
        q = 1-q
    return (3*q + (1-2*ph)*(1-q)) / (4*q)

def Delta(pb, ph, q, label="adam"):
    if label == "adam":
        q = 1-q
    return (2 *q + (1 -q)*(1 -2 *ph))** 2 - 4*(4 *pb *q -2 *q) *q

def afun(pb, ph, q, label="adam"):
    if label == "adam":
        q = 1-q
    return (4 *pb *q -2 *q)
def bfun(pb, ph, q, label="adam"):
    if label == "adam":
        q = 1-q
    return -2 *q - (1 -q)*(1 -2 *ph)
def cfun(pb, ph, q, label="adam"):
    if label == "adam":
        q = 1-q
    return q

def get_rhos(pb,ph,q, label="adam"):
    a = afun(pb, ph, q, label=label)
    b = bfun(pb, ph, q, label=label)
    c = cfun(pb, ph, q, label=label)
    
    if a == 0:
        return (-c / b, )
    
    d = Delta(pb, ph, q, label=label)
    if d < 0:
        if np.round(d, decimals = 15) < 0:
            return (np.nan, )
        else:
            d = np.round(d, decimals = 15)
    
    rho1 = (-b  - np.sqrt(d)) / 2 /a
    rho2 = (-b  + np.sqrt(d)) / 2 /a
    return sorted([rho1, rho2])
# rho2(pb,ph,q) = (2q + (1-q)(1-2ph) + sqrt(Delta(pb,ph,q))) / 2*(4pb*q-2q)

def get_quasi_rho(pb,ph,q, label="adam"):
    """Returns one value which is the level of quasi-stationary state if such a level exists. 
    Otherwise returns np.NaN

    Args:
        pb (_type_): _description_
        ph (_type_): _description_
        q (_type_): _description_
    """
    rhos = get_rhos(pb,ph,q, label=label)
    
    if len(rhos) == 1:
        if np.isnan(rhos[0]):
            return np.NaN
        elif (rhos[0] <= 1) & (rhos[0] >= 0):
            return rhos[0]
        else:
            return np.NaN
    
    if (rhos[0] <= 1) & (rhos[0] >= 0):
        if rhos[0] < rhos[1]: #rhos[0] is in proper range and is smaller. 
            rho_exp = rhos[0]
        elif (rhos[1] <= 1) & (rhos[1] >= 0): #rhos[1] is in proper range and is smaller. 
            rho_exp = rhos[1]
        else: #rhos[0] is in proper range and rhos[1] is not. 
            rho_exp = rhos[0]
    elif (rhos[1] <= 1) & (rhos[1] >= 0): #rhos[1] is in proper range and rhos[0] is not. 
        rho_exp = rhos[1]
    else:
        rho_exp = np.NaN
        
    return rho_exp

def get_separatrix(pbs, ph, q, label="adam"):
    """Returns rho values of separatrix (if exists) for given parameters

    Args:
        pbs (list or numpy.array): pb values
        ph (float): probabiilty ph
        q (float): probability q
    """
    
    rho_crits = np.zeros(len(pbs))
    for i, pb in enumerate(pbs):
        rhos_crit = get_rhos(pb, ph, q, label=label)
        
        no_sep_value = True
        if len(rhos_crit) == 2:
            if 0 < rhos_crit[0] < 1:
                if 0 < rhos_crit[1] < 1:
                    no_sep_value = False
                    rho_crits[i] = rhos_crit[1]
        if no_sep_value:
            rho_crits[i] = np.nan
    return rho_crits

def get_quasi_q(rho, pb, ps):
    """Returns value of q in the quasi-stationary state. 

    Args:
        rho (_type_): _description_
        pb (_type_): _description_
        ps (_type_): _description_
    """
    val = 2*(2*pb - 1)*rho**2 - 2*rho+1
    return val/(val + (1-2*ps)*rho)

def get_quasi_pb(rho, ps, q):
    """Returns value of pb in the quasi-stationary state. 

    Args:
        rho (_type_): _description_
        pb (_type_): _description_
        ps (_type_): _description_
    """
    return 1/2 * (((2*(1-q) + (1-2*ps)*q)*rho - (1-q)) / (2*(1-q)*rho**2) + 1)

def get_quasi_ps(rho, pb, q):
    """Returns value of ps in the quasi-stationary state. 

    Args:
        rho (_type_): _description_
        pb (_type_): _description_
        ps (_type_): _description_
    """
    return 1/2 * ((1-q)/q * (2*rho - 2*(2*pb-1)*rho**2 - 1)/rho + 1)



def find_parameters(rho = -1., q = -1., ps = -1., pb = -1.):
    """The user should specify 3 parameters. The value of the 4th one will be returned. 
    The forth parameter will be such that the system will be in quasi-stationary state. 

    Args:
        rho (_type_, optional): _description_. Defaults to -1..
        q (_type_, optional): _description_. Defaults to -1..
        ps (_type_, optional): _description_. Defaults to -1..
        pb (_type_, optional): _description_. Defaults to -1..
    """
    pars = np.array([rho, q, ps, pb])
    count_m1 = sum(pars == -1)
    if count_m1 != 1:
        # print(pars)
        raise ValueError("One should define exactly 3 parameters. ", pars)
    if rho == -1.:
        return get_quasi_rho(pb,ps,q)
    elif q == -1.:
        return get_quasi_q(rho, pb,ps)
    elif ps == -1.:
        return get_quasi_ps(rho, pb,q)
    elif pb == -1.:
        return get_quasi_pb(rho,ps,q)
    
import os, ast

def process_data(data, n_links, n_triads, k=1):
    for column in data.columns[k:-2]:
        data.loc[:, column] = data[column].str.split(',').map(lambda x: [int(float(x_el))/n_triads for x_el in x])
    data.loc[:, 'rho'] = data['rho'].str.split(',').map(lambda x: [int(float(x_el))/n_links for x_el in x])
    return data

def means_of_data(data, start, steps):
    df = pd.DataFrame(columns=data.columns)
    for _, row in data.iterrows():
        df = df.append(pd.DataFrame({
        'prob': row.p,
        'rho': row.rho[start:][::steps],
        'n0': row.n0[start:][::steps],
        'n1': row.n1[start:][::steps],
        'n2': row.n2[start:][::steps],
        'n3': row.n3[start:][::steps]
        }))
    df = df.groupby('prob').mean()
    return df

def get_rho_init(filename):
    with open(filename) as f:
        first_line = f.readline()
        spl = first_line.split("--rho-init', '")
        if len(spl) == 1:
            return 0.5
        else:
            strval = spl.split("'")[0]
            return float(strval)
        
def read_params(filename, external_dataset_stats_file = "results_realnet_spanish.h5"):
    # print("Started " + filename)
    params_d = {}
    with open(filename) as f:
        first_line = f.readline()
        # firstline = "# Commit: 126356fd51d619772e0c4137e13dfe116d309ebe# Arguments: ['main.py', 'LtdStatus', '-n', '100', '-p', '0.88', '-q', '0.5', '-ps', '0.25', '-s', '100', '-r', '1', '--rho-init', '0.9', '--ltd-agent-based', '--on-triad-status', 'outputs/test']"
        params = first_line.split("Arguments: ")[1]
        # params.strip('][').strip("'").split(', ')
        try:
            params = ast.literal_eval(params)
            isvalue = False
            last_key = ""
            for par in params:
                if par.startswith('--'):
                    if isvalue:
                        isvalue = False
                        # params_d[last_key] = True
                if not isvalue:
                    if par.startswith('--'):
                        par2 = par.strip("-")
                        par2 = par2.replace('-', '_')
                        params_d[par2] = True
                        isvalue = True
                        last_key = par2
                    elif par.startswith('-'):
                        par2 = par.strip("-")
                        par2 = par2.replace('-', '_')
                        params_d[par2] = np.NAN
                        isvalue = True
                        last_key = par2
                else:
                    isvalue = False
                    try:
                        if "." in par:
                            params_d[par2] = float(par)
                        else:
                            params_d[par2] = int(par)
                    except ValueError:
                        pass
        except (SyntaxError, ValueError):
            # print(params)
            # print(params == "Namespace(command=None, dataset='../data/wikielections/wikielections_triads2.h5', network=<networktools.network_generators.real_graphs.GeneralRealNetwork object at 0x7f1f70ffcfa0>, is_directed=True, agent_based=True, ltd_agent_based=True, on_triad_status=True, save_pk_distribution=False, build_triad='choose_agents', exp_name='outputs/wiki2-s1000-triads', probability=[0.8], q=[0.0], psprob=[0.1], rho_init=[0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001], steps=1000, saving_multiplier=10, repetitions=1, no_triad_stats=False, keep_rho_at=[0.67, 0.9, 0.65, 0.92])")
            # print(ord(params[-1]))
            params = params.split("Namespace(")[1]
            params = params[:-1]
            if params[-1] == ")":
                params = params[:-1]
            params = params.split(',')
            params = list(map(lambda x: x.strip(), params))
            """Following is needed in the case some parameters are given as lists."""
            i = len(params) - 1
            while i >= 1:
                if '=' not in params[i]:
                    params[i-1] = ','.join(params[i-1:i+1])
                    params.pop(i)
                i -= 1
            params = list(map(lambda x: x.split('='), params))
            keys = [param[0] for param in params]
            vals = [param[1] for param in params]
            for key, val in zip(keys, vals):
                if key == "steps":
                    key = "s"
                elif key == "n_agents":
                    key = "n"
                if val.startswith("'"):
                    params_d[key] = val
                    continue
                if val.startswith("["):
                    val = ast.literal_eval(val)
                    if len(val) == 1:
                        val = round(val[0],10)
                    else:
                        val = [round(v,10) for v in val]
                    # val = val[1:-1]
                    params_d[key] = val
                else:
                    try:
                        val = ast.literal_eval(val)
                        if val:
                            val = round(val,10)
                            params_d[key] = val
                    except SyntaxError:     
                        try:
                            params_d[key] = int(val)
                        except ValueError:
                            try:
                                if val:
                                    # print(val)
                                    val = round(val,10)
                                    params_d[key] = float(val)
                            except (ValueError, TypeError):
                                pass
    
    if "dataset" in params_d:
        if "wikielections" in params_d["dataset"].lower():
            if "wikielections_triads2" in params_d["dataset"].lower():
                params_d["triads"] = 745129
                params_d["links"] = 94933
            else:
                params_d["triads"] = 747589
                params_d["links"] = 95152
            params_d["dataset"] = "wikielections"
        elif "slashdot" in params_d["dataset"].lower():
            params_d["dataset"] = "slashdot"
            params_d["triads"] = 1251925
            params_d["links"] = 285003
        elif "epinions" in params_d["dataset"].lower():
            params_d["dataset"] = "epinions"
            params_d["triads"] = 10961993
            params_d["links"] = 667110
        elif "sampson" in params_d["dataset"].lower():
            params_d["dataset"] = "sampson"
            params_d["triads"] = 1158
            params_d["links"] = 184
        elif "bitcoin-alpha" in params_d["dataset"].lower():
            params_d["dataset"] = "bitcoin-alpha"
            params_d["triads"] = 88753
            params_d["links"] = 16793
        elif "bitcoin-otc" in params_d["dataset"].lower():
            params_d["dataset"] = "bitcoin-otc"
            params_d["triads"] = 125886
            params_d["links"] = 24876
        elif "spanish-highschools" in params_d["exp_name"].lower():
            try:
                df_spanish = pd.read_hdf(Path("../../../triad-statistics/" + external_dataset_stats_file))
            except FileNotFoundError:
                df_spanish = pd.read_hdf(Path("/home/pgorski/data/spanish-highschools/" + external_dataset_stats_file))
            school_name = params_d["dataset"][1:-1]
            params_d["dataset"] = params_d["dataset"]
            params_d["triads"] = int(df_spanish.loc[school_name]["T"])
            params_d["links"] = int(df_spanish.loc[school_name]["L"])
        elif "complete_triads" in params_d["dataset"].lower():
            params_d["dataset"] = "complete_triads"
            params_d["n"] = 32
            # params_d["links"] = 24876
            
            
    return params_d

def process_folder(directory, no_triad_stats = True, df_files = None, 
                   use_only_files = None, start_index = 0, max_num_rows = 10000, 
                   no_arrays = False, dataset = "", external_dataset_stats_file = "results_realnet_spanish.h5"):
    """ Processes the whole folder looking for files. 
    If only new files are to be processes then df_files should be given. 

    Args:
        directory (str): folder to look at. It should contain subfolders. 
        no_triad_stats (bool, optional): Whether not to or to look at 
            and keep triad count changes. Defaults to True.
        df_files (_type_, optional): Files already analyzed. 
            They should be omitted. Defaults to None.
        use_only_files (_type_, optional): Files to look at. 
            If this is None, then all the folders will be analyzed. 
            Defaults to None.
        start_index (int, optional): The file first to be analyzed. 
            The previous files are omitted. Defaults to 0.
        max_num_rows (int, optional): Approximate number of rows to be generated. 
            Then, the processing will be stopped, 
            and the last file processed is given as the output. Defaults to 10000.
        no_arrays (bool, optional): Whether all the arrays should not be imported 
            to final dataframe. Defaults to False.
        dataset (str, optional): "" - if all files should be analyzed, 
            "ALL" if all datasets in those files should be analyzed
            name of the dataset if data from one dataset only should be gathered. 
            Defaults to "".

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        pd.DataFrame: data
        pd.DataFrame: information about found files (file list)
        int: last row of the file list that was processed
    """    
    
    if no_arrays and (not no_triad_stats):
        raise ValueError("Boolean values of no_arrays and no_triad_stats mismatch.")
    
    if use_only_files is not None:
        files = use_only_files
    else:
        files = [os.path.join(directory, dir, "outputs.tsv") for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir)) ]
    
    "Processing file headers and choosing files from single dataset if necessary. "
    dicts = [read_params(file, external_dataset_stats_file = external_dataset_stats_file) for i, file in enumerate(files)]# if empty_folds[i] == 1]
    if (dataset != "") and (dataset != "ALL"):
        correct_file_inds = [i for i in range(len(dicts)) if dicts[i]["dataset"][1:-1] == dataset]
        correct_files_bool = [dict["dataset"][1:-1] == dataset for dict in dicts]
        
        files = [files[ind] for ind in correct_file_inds]
        dicts = [dicts[ind] for ind in correct_file_inds]
    
    if len(dicts) == 0:
        return None, None, -1
    
    # get mod time dataframe
    m_times = [os.stat(file).st_mtime for file in files]
    df_files_all = pd.DataFrame({"File": files, "Modification time": m_times})
    df_files_all.set_index('File', inplace=True)
    
    if (df_files is not None) and (df_files.equals(df_files_all)):
        return None, None, -1
    
    df_cols = ['q','p', 'ps', 'rho_init', 'steps']
    usecols = ['q','p', 'ps', 'rho_init']
    if not no_arrays:
        df_cols.extend(['cur_steps', 'rho'])
        usecols.append('rho')
    # usecols_backup = ['q','p', 'ps', 'tr0', 'tr1']
    
    if not no_triad_stats:
    #     usecols = ['q','p', 'ps', 'rho', 'bp']
    #     usecols_backup = ['q','p', 'ps', 'tr0', 'tr1']
    # else:
        for i in range(8):
            df_cols.append('tr' + str(i))
            usecols.append('tr' + str(i))
    
    df = pd.DataFrame({col: [] for col in df_cols})
    last_file_processed = start_index
    # usecols = ['q','p', 'ps', 'rho', 'bp']
    # usecols_backup = ['q','p', 'ps', 'tr0', 'tr1']
    
    empty_folds = np.ones(len(files))
    reps = np.zeros(len(files), dtype = int)
    
    for i, file in enumerate(files[start_index:]):
        if len(df.index) >= max_num_rows:
            break
        
        if (df_files is not None) and (file in df_files.index):
            # checking if file was already processes and there was no more recent version
            if df_files.loc[file].values[0] == df_files_all.loc[file].values[0]:
                continue
        
        inilen = len(df)
        df2 = df.append(pd.read_csv(file,sep = '\t', comment='#', usecols = usecols))
        
        if type(np.array(df2.rho)[-1]) != str:
            if np.isnan(np.array(df2.rho)[-1]):
                """There is most likely error in columns. rho values are in another column"""
                df3 = pd.read_csv(file,sep = '\t', comment='#', usecols = usecols_backup)
                # df3.append(pd.read_csv(file,sep = '\t', comment='#', usecols = usecols_backup))
                df3_len = len(df3)
                df2_len = inilen+df3_len
                # print(df3)
                # print(df3_len)
                # print(df3.tr0.iloc[-df3_len:-1])
                # print(df3.tr0)
                # return df3, df2
                df2.loc[np.array(range(0,df2_len)) >= df2_len - df3_len, 'rho'] = df3["tr0"]
                # df2.loc[np.array(range(0,df2_len)) >= df2_len - df3_len, 'bp'] = df3["tr1"]
                # print(df2)
                # for i, row in enumerate(df3.iterrows()):
                #     df2.rho.iloc[-i] = df3.tr0.iloc[-i]
                #     df2.bp.iloc[-i] = df3.tr1.iloc[-i]
        df = df2
        
        endlen = len(df)
        
        reps[i] = endlen - inilen
        
        if inilen == endlen:
            empty_folds[i] = False
        
        last_file_processed = start_index + i
    
    if len(df) == 0:
        # No new files probably. 
        return None, None, -1
    
    # rho_inits_one = [round(d["rho_init"], 6) if "rho_init" in d else 0.5 for d in dicts]
    steps_one = [d["s"] for d in dicts]
    
    # rho_inits = [rho_init for rho_init, rep in zip(rho_inits_one, reps) for _ in range(0,rep)]
    steps = [step for step, rep in zip(steps_one, reps) for _ in range(0,rep)]
    # print(df)
    # return df, rho_inits
    # df.rho_init = rho_inits
    df.steps = steps
    
    if not no_arrays:
        ms_one = [d["saving_multiplier"] if "saving_multiplier" in d else 1 for d in dicts]
        ms = [m for m, rep in zip(ms_one, reps) for _ in range(0,rep)]

        cur_steps = [[*list(range(0,step, m)), step] for m, step in zip(ms, steps)]        
        df.cur_steps = cur_steps
    
    df = df.dropna().reset_index()
    
    if not no_arrays:
        if "n" in dicts[0]:
            Ls_one = [d["n"]*(d["n"] - 1) for d in dicts]
        elif "dataset" in dicts[0]:
            Ls_one = [d["links"] for d in dicts]
        else:
            raise ValueError("Neither number of nodes nor dataset was given.", dicts[0])
        Ls = [L for L, rep in zip(Ls_one, reps) for _ in range(0,rep)]
        df.rho = np.array(df.rho.str.split(',').map(lambda x: [float(x_el) for x_el in x]))
        df["Lplus"] = df.rho
        df.rho = [np.array(rho) / L for rho, L in zip(df.rho, Ls)]
        
        possibly_shorter = [d["keep_rho_at"][:2] != [0.,1.] if "keep_rho_at" in d else False  for d in dicts]
        possibly_shorters = [p for p, rep in zip(possibly_shorter, reps) for _ in range(0,rep)]
        
        """Correcting number of cur_steps for old simulations"""
        for ind, row in df.iterrows():
            if possibly_shorters[ind]:
                if len(row.rho) != len(row.cur_steps):
                    df.at[ind, "cur_steps"] = row.cur_steps[0:len(row.rho)]
                    # row.cur_steps = 
            else:
                if len(row.rho) != len(row.cur_steps):
                    df.iloc[ind].cur_steps.pop()
                if len(row.rho) != len(row.cur_steps):
                    raise ValueError("Wrong lengths")
    
    if not no_triad_stats:
        if "n" in dicts[0]:
            Ts_one = [d["n"]*(d["n"] - 1)*(d["n"] - 2) for d in dicts]
        elif "dataset" in dicts[0]:
            Ts_one = [d["triads"] for d in dicts]
        else:
            raise ValueError("Neither number of nodes nor dataset was given.", dicts[0])
        Ts = [T for T, rep in zip(Ts_one, reps) for _ in range(0,rep)]
        for i in range(8):
            col = 'tr' + str(i)
            df[col] = np.array(df[col].str.split(',').map(lambda x: [float(x_el) for x_el in x]))
            df[col] = [np.array(Ni) / T for Ni, T in zip(df[col], Ts)]
    
    return df, df_files_all, last_file_processed

def group_results(df, cols):
    """group results according to (q,p,ps,rho_init,steps) anc check the outcome

    Args:
        df (_type_): _description_
    """
    mylist = dict()
    
    for group in df.groupby(cols):
        (group_label, df_temp) = group
        q = df.q[0]
        ps = df.ps[0]
        
        rho_crit = get_rho_crit(q, ps)
        
        reps = len(df_temp.rho)
        paradise = np.sum([rho[-1] == 1. for rho in df_temp.rho])
        quasi_stat = np.sum([rho[-1] < rho_crit for rho in df_temp.rho])
        other = np.array(reps) - paradise - quasi_stat
        
        paradise_ratio = paradise / (paradise + quasi_stat)
        
        d = dict(zip(cols, group_label))
        
        d.update({"repetitions": reps, "paradise": paradise, "quasi_stat": quasi_stat, "other": other, "paradise_ratio": paradise_ratio})
        mylist.update({group_label: d})
    
    return pd.DataFrame(data = mylist.values(), index = mylist.keys(), columns = [*cols, 'repetitions', 'paradise', 'quasi_stat', 'other', 'paradise_ratio'])
    


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def get_zero_crossing(vals):
    zero_crossings = np.where(np.diff(np.sign(vals)))[0]
    return zero_crossings
    

def get_quasilevel_exp(rho, rho_exp):
    
    if np.isnan(rho_exp):
        return get_quasilevel(rho)
    
    crossing = get_zero_crossing(rho - rho_exp)
    if len(crossing) > 1:
        beg = crossing[0]
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN
    
    beg = int(beg - np.sqrt(beg))
    if beg < 0:
        beg = 0

    fin = len(rho)

    m = np.mean(rho[beg:fin])
    s = np.std(rho[beg:fin], ddof = 1)
    
    # tt = []
    while not ((m - s < rho[fin-1]) & ( rho[fin-1] < m + s )):
        fin -= 1
        
        if beg == fin:
            return np.NaN, np.NaN, np.NaN, np.NaN
        
        m = np.mean(rho[beg:fin])
        s = np.std(rho[beg:fin], ddof = 1)
        # tt.append((m, s, fin))
    
    return m, s, beg, fin

def get_quasilevel(rho, ini_beg = "half"):
    is_correct = False

    if ini_beg == "half":
        beg = int(len(rho) / 2)
    elif ini_beg == "last20":
        beg = len(rho) - 20
    elif ini_beg == 1:
        return rho[-1], 0, len(rho)-1, len(rho)
    elif type(ini_beg) == int:
        if ini_beg < 0:
            beg = len(rho) - ini_beg
        elif ini_beg >= 0:
            beg = ini_beg
    else:
        beg = 0
    fin = len(rho)
    
    if (beg < 0) | (beg >= fin):
        # Too short dataset for finding QS levels. 
        return np.NaN, np.NaN, np.NaN, np.NaN

    m = np.mean(rho[beg:fin])
    s = np.std(rho[beg:fin], ddof = 1)

    is_correct_beg = ((m - s < rho[beg]) & ( rho[beg] < m + s ))
    is_correct_fin = ((m - s < rho[fin-1]) & ( rho[fin-1] < m + s ))

    is_correct = is_correct_beg & is_correct_fin

    while not is_correct:
        if np.abs(rho[beg] - m) > np.abs(rho[fin-1] - m):
            beg += 1
        else:
            fin -= 1
        
        if beg == fin:
            return np.NaN, np.NaN, np.NaN, np.NaN
        
        m = np.mean(rho[beg:fin])
        s = np.std(rho[beg:fin], ddof = 1)

        is_correct_beg = ((m - s < rho[beg]) & ( rho[beg] < m + s ))
        is_correct_fin = ((m - s < rho[fin-1]) & ( rho[fin-1] < m + s ))

        is_correct = is_correct_beg & is_correct_fin
        
    return m, s, beg, fin

def get_quasilevel2(rho, ini_beg = "half", condition = "variance", conditions = [], condition_params = [], keep_fin_constant = True):
    """Algorithm for looking whether QS exists. 

    Args:
        rho (np.array): a series with densities of positive links
        ini_beg (str, int, optional): Method for choosing initial point of averaging. Possible values:
            "half" : looking for QS state will start at beg = len(rho)/2
            "last20" : checking in the last 20 steps if QS state was reached. 
            (int): checking in the last steps (the number gives number of steps) if QS state was reached. 
            Defaults to "half". If not given, beg will be set to 0. 
        condition (str, int, optional): Method for deciding if QS was found. Possible values:
            "variance" : the beginning value of the range should be between [m-s, m+s]
            "pvalue" : coefficient test is performed. If the pvalue should be above 0.05.
            "R2" : R2 is measured. R2 value should be below 0.3.
            Defaults to "variance". 
        conditions (list): It may contain multiple conditions. 
            Defaults to  [].
        condition_params (float, int, array, optional): Parameters for condition functions. 
            If not given default values will be used. 
            Right now, this fully works only when one condition is chosen. 
            If multiple conditions are chosen, then condition_params contains parameter for coeficient. 
            Defaults to  []. 
        keep_fin_constant (bool, optional): If True, then all rho values up to end are used. 
            If False, then either beg or fin are changed accordingly. 
            Defaults to True. 

    Returns:
        _type_: _description_
    """
    if len(conditions) > 0:
        condition = ""

    if ini_beg == "half":
        beg = int(len(rho) / 2)
    elif ini_beg == "last20":
        beg = len(rho) - 20
    elif ini_beg == 1:
        return rho[-1], 0, len(rho)-1, len(rho)
    elif type(ini_beg) == int:
        if ini_beg < 0:
            beg = len(rho) - ini_beg
        elif ini_beg >= 0:
            beg = ini_beg
    else:
        beg = 0
    fin = len(rho)
    
    if (beg < 0) | (beg >= fin):
        # Too short dataset for finding QS levels. 
        return np.NaN, np.NaN, np.NaN, np.NaN

    m = np.mean(rho[beg:fin])
    s = np.std(rho[beg:fin], ddof = 1)

    is_correct = True
    if (condition == "variance") | ("variance" in conditions):
        is_correct_beg = ((m - s < rho[beg]) & ( rho[beg] < m + s ))
        is_correct_fin = True

        is_correct = is_correct & is_correct_beg & is_correct_fin
    if (condition == "pvalue") | ("pvalue" in conditions):  
        if fin - beg < 8:
            is_correct = False
            return np.NaN, np.NaN, np.NaN, np.NaN
        mod = sm.OLS(rho[beg:fin], sm.add_constant(list(range(beg, fin))))
        fii = mod.fit()
        p_values = fii.summary2().tables[1]['P>|t|']
        if (condition == []) | (condition_params == []):
            p_value_th = 0.05
        else:
            p_value_th = condition_params
        
        is_correct = is_correct & (p_values.x1 > p_value_th)
        
    if (condition == "R2") | ("R2" in conditions):
        fit = LinearRegression()
        X = np.array(list(range(beg, fin))).reshape(-1, 1)
        y = rho[beg:fin].reshape(-1, 1)
        fit.fit(X, y)
        fit = LinearRegression()
        X = np.array(list(range(beg, fin))).reshape(-1, 1)
        y = rho[beg:fin].reshape(-1, 1)
        fit.fit(X, y)
        
        if (condition == []) | (condition_params == []):
            R2_th = 0.1
        else:
            R2_th = condition_params
        
        is_correct = is_correct & (fit.score(X, y) < R2_th)
    if (condition == "pvalue_or_coef") | ("pvalue_or_coef" in conditions):  
        mod = sm.OLS(rho[beg:fin], sm.add_constant(list(range(beg, fin))))
        fii = mod.fit()
        p_values = fii.summary2().tables[1]['P>|t|']
        coef = fii.params[1]
        
        if (condition == []) | (condition_params == []):
            multiplier = 1
        else:
            multiplier = condition_params
        
        is_correct = is_correct & ((p_values.x1 > 0.05) | (np.abs(coef) < 0.00001*multiplier))
    if (condition == "R2_or_coef") | ("R2_or_coef" in conditions):  
        fit = LinearRegression()
        X = np.array(list(range(beg, fin))).reshape(-1, 1)
        y = rho[beg:fin].reshape(-1, 1)
        fit.fit(X, y)
        coef = fit.coef_
        
        if (condition == []) | (condition_params == []):
            multiplier = 1
        else:
            multiplier = condition_params
        
        is_correct = is_correct & ((fit.score(X, y) < 0.1) | (np.abs(coef) < 0.00001*multiplier))
    

    while not is_correct:
        if keep_fin_constant:
            beg += 1
        else:
            if np.abs(rho[beg] - m) > np.abs(rho[fin-1] - m):
                beg += 1
            else:
                fin -= 1
        
        if beg == fin:
            return np.NaN, np.NaN, np.NaN, np.NaN
        
        m = np.mean(rho[beg:fin])
        s = np.std(rho[beg:fin], ddof = 1)

        is_correct = True
        if (condition == "variance") | ("variance" in conditions):
            # print("g")
            is_correct_beg = ((m - s < rho[beg]) & ( rho[beg] < m + s ))
            is_correct_fin = True

            is_correct = is_correct & is_correct_beg & is_correct_fin
        if (condition == "pvalue") | ("pvalue" in conditions): 
            if fin- beg < 8:
                is_correct = False
                return np.NaN, np.NaN, np.NaN, np.NaN
             
            mod = sm.OLS(rho[beg:fin], sm.add_constant(list(range(beg, fin))))
            fii = mod.fit()
            p_values = fii.summary2().tables[1]['P>|t|']
            if (condition == []) | (condition_params == []):
                p_value_th = 0.05
            else:
                p_value_th = condition_params
            
            is_correct = is_correct & (p_values.x1 > p_value_th)
        if (condition == "R2") | ("R2" in conditions):
            fit = LinearRegression()
            X = np.array(list(range(beg, fin))).reshape(-1, 1)
            y = rho[beg:fin].reshape(-1, 1)
            fit.fit(X, y)
            if (condition == []) | (condition_params == []):
                R2_th = 0.1
            else:
                R2_th = condition_params
            
            is_correct = is_correct & (fit.score(X, y) < R2_th)
        if (condition == "pvalue_or_coef") | ("pvalue_or_coef" in conditions):  
            mod = sm.OLS(rho[beg:fin], sm.add_constant(list(range(beg, fin))))
            fii = mod.fit()
            p_values = fii.summary2().tables[1]['P>|t|']
            coef = fii.params[1]
            
            if (condition == []) | (condition_params == []):
                multiplier = 1
            else:
                multiplier = condition_params
            
            is_correct = is_correct & ((p_values.x1 > 0.05) | (np.abs(coef) < 0.00001*multiplier))
        if (condition == "R2_or_coef") | ("R2_or_coef" in conditions):  
            fit = LinearRegression()
            X = np.array(list(range(beg, fin))).reshape(-1, 1)
            y = rho[beg:fin].reshape(-1, 1)
            fit.fit(X, y)
            coef = fit.coef_
            
            if (condition == []) | (condition_params == []):
                multiplier = 1
            else:
                multiplier = condition_params
            
            is_correct = is_correct & ((fit.score(X, y) < 0.1) | (np.abs(coef) < 0.00001*multiplier))
        
        
    return m, s, beg, fin


def get_quasilevels(df, calc_rho = True, calc_triads = True, drop_list_columns = False, leave_correct = False):
    """Automatically finds average values for quasi-stationary state. If it cannot find such a state then, NaN values are given. 

    Args:
        df (_type_): _description_
        drop_list_columns: if a list of columns is given here, then they are dropped from the final frame
            If this variable is set to true, all list columns will be dropped. These are:
            ['cur_steps', 'rho', 'Lplus', 'tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7']

    Returns:
        _type_: _description_
    """
    df2 = df.copy(deep = True)
    
    if calc_rho:
        ms = np.zeros(len(df2.index))
        ss = np.zeros(len(df2.index))
        begs = np.zeros(len(df2.index))
        fins = np.zeros(len(df2.index))
        
        ms2 = np.zeros(len(df2.index))
        ss2 = np.zeros(len(df2.index))
        begs2 = np.zeros(len(df2.index))
        fins2 = np.zeros(len(df2.index))
        
        ms3 = np.zeros(len(df2.index))
        ss3 = np.zeros(len(df2.index))
        begs3 = np.zeros(len(df2.index))
        fins3 = np.zeros(len(df2.index))
        
        for ind, row in df2.iterrows():
            q = row.q
            psb = row.p
            ps = row.ps
            
            try:
                rho_exp = get_quasi_rho(psb,ps,q)
            except ZeroDivisionError:
                rho_exp = np.NaN
            
            m, s, beg, fin = get_quasilevel_exp(row.rho, rho_exp)
            ms2[ind] = m
            ss2[ind] = s
            begs2[ind] = beg
            fins2[ind] = fin
            
            m, s, beg, fin = get_quasilevel(row.rho)
            ms[ind] = m
            ss[ind] = s
            begs[ind] = beg
            fins[ind] = fin
            
            m, s, beg, fin = get_quasilevel2(row.rho, condition = "pvalue")
            ms3[ind] = m
            ss3[ind] = s
            begs3[ind] = beg
            fins3[ind] = fin
            
        df2['rho_qs_m'] = ms
        df2['rho_qs_s'] = ss
        df2['rho_qs_b'] = begs
        df2['rho_qs_f'] = fins
        df2['rho_qs2_m'] = ms2
        df2['rho_qs2_s'] = ss2
        df2['rho_qs2_b'] = begs2
        df2['rho_qs2_f'] = fins2
        df2['rho_qs3_m'] = ms3
        df2['rho_qs3_s'] = ss3
        df2['rho_qs3_b'] = begs3
        df2['rho_qs3_f'] = fins3
    
    if calc_triads:
        """ get quasi levels of triads"""
        ms = np.zeros(len(df2.index))
        ss = np.zeros(len(df2.index))
        begs = np.zeros(len(df2.index))
        fins = np.zeros(len(df2.index))
        
        ms3 = np.zeros(len(df2.index))
        ss3 = np.zeros(len(df2.index))
        begs3 = np.zeros(len(df2.index))
        fins3 = np.zeros(len(df2.index))
        for i in range(0,8):
            col = 'tr' + str(i)
            for ind, row in df2.iterrows():
                q = row.q
                psb = row.p
                ps = row.ps
                
                m, s, beg, fin = get_quasilevel(row[col])
                ms[ind] = m
                ss[ind] = s
                begs[ind] = beg
                fins[ind] = fin
                
                m, s, beg, fin = get_quasilevel2(row[col], condition = "pvalue")
                ms3[ind] = m
                ss3[ind] = s
                begs3[ind] = beg
                fins3[ind] = fin
            df2[col + '_qs_m'] = ms
            df2[col + '_qs_s'] = ss
            df2[col + '_qs_b'] = begs
            df2[col + '_qs_f'] = fins
            df2[col + '_qs3_m'] = ms3
            df2[col + '_qs3_s'] = ss3
            df2[col + '_qs3_b'] = begs3
            df2[col + '_qs3_f'] = fins3
    
    if drop_list_columns:
        if drop_list_columns is True:
            drop_list_columns = ['cur_steps', 'rho', 'Lplus', 'tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7']
        
        if leave_correct:
            inds = [row.cur_steps[-1] == row.steps for ind,row in df2.iterrows()]
            df2 = df2.iloc[inds]
            df2.reset_index(inplace=True, drop=True)
        
        df2.drop(drop_list_columns, axis = 1, inplace = True)
    
    return df2

def get_quasilevels2(df, calc_triads = True, drop_list_columns = False, leave_correct = False, ini_beg = "half", 
                     use_existing_qs_levels = False):
    """Automatically finds average values for quasi-stationary state. If it cannot find such a state then, NaN values are given. 
    Difference with `get_quasilevels` is that it only calculates if for rho, and then uses the obtained ranges to get the mean for trx.
    It also uses a slightly different calculation of quasi-levels. 
    It is especially useful for spanish classes dataset. 

    Args:
        df (_type_): _description_
        drop_list_columns: if a list of columns is given here, then they are dropped from the final frame
            If this variable is set to true, all list columns will be dropped. These are:
            ['cur_steps', 'rho', 'Lplus', 'tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7']

    Returns:
        _type_: _description_
    """
    df2 = df.copy(deep = True)
    
    ms = np.zeros(len(df2.index))
    ss = np.zeros(len(df2.index))
    begs = np.zeros(len(df2.index))
    fins = np.zeros(len(df2.index))
    
    ms2 = np.zeros(len(df2.index))
    ss2 = np.zeros(len(df2.index))
    begs2 = np.zeros(len(df2.index))
    fins2 = np.zeros(len(df2.index))
    
    ms3 = np.zeros(len(df2.index))
    ss3 = np.zeros(len(df2.index))
    begs3 = np.zeros(len(df2.index))
    fins3 = np.zeros(len(df2.index))
    
    ms4 = np.zeros(len(df2.index))
    ss4 = np.zeros(len(df2.index))
    begs4 = np.zeros(len(df2.index))
    fins4 = np.zeros(len(df2.index))
    
    if not use_existing_qs_levels:
        for ind, row in df2.iterrows():
            q = row.q
            psb = row.p
            ps = row.ps
            
            try:
                rho_exp = get_quasi_rho(psb,ps,q)
            except ZeroDivisionError:
                rho_exp = np.NaN
            
            m, s, beg, fin = get_quasilevel_exp(row.rho, rho_exp)
            ms2[ind] = m
            ss2[ind] = s
            begs2[ind] = beg
            fins2[ind] = fin
            
            m, s, beg, fin = get_quasilevel(row.rho, ini_beg = ini_beg,)
            ms[ind] = m
            ss[ind] = s
            begs[ind] = beg
            fins[ind] = fin
            
            m, s, beg, fin = get_quasilevel2(row.rho, condition = "pvalue", ini_beg = ini_beg,)
            ms3[ind] = m
            ss3[ind] = s
            begs3[ind] = beg
            fins3[ind] = fin
            
            m, s, beg, fin = get_quasilevel2(row.rho, condition = "pvalue", ini_beg = ini_beg, keep_fin_constant = False)
            ms4[ind] = m
            ss4[ind] = s
            begs4[ind] = beg
            fins4[ind] = fin
                
            df2['rho_qs_m'] = ms
            df2['rho_qs_s'] = ss
            df2['rho_qs_b'] = begs
            df2['rho_qs_f'] = fins
            df2['rho_qs2_m'] = ms2
            df2['rho_qs2_s'] = ss2
            df2['rho_qs2_b'] = begs2
            df2['rho_qs2_f'] = fins2
            df2['rho_qs3_m'] = ms3
            df2['rho_qs3_s'] = ss3
            df2['rho_qs3_b'] = begs3
            df2['rho_qs3_f'] = fins3
            df2['rho_qs4_m'] = ms4
            df2['rho_qs4_s'] = ss4
            df2['rho_qs4_b'] = begs4
            df2['rho_qs4_f'] = fins4
    
    if calc_triads:
        """ get quasi levels of triads"""
        # ms = np.zeros(len(df2.index))
        # ss = np.zeros(len(df2.index))
        
        # ms3 = np.zeros(len(df2.index))
        # ss3 = np.zeros(len(df2.index))
        
        # ms4 = np.zeros(len(df2.index))
        # ss4 = np.zeros(len(df2.index))
        for i in range(0,8):
            col = 'tr' + str(i)
            
            # ms = df2[col].apply(lambda x: mean)
            df2[col + '_qs_m'] = df2.apply(lambda row: np.mean(row[col][int(row['rho_qs_b']):int(row['rho_qs_f'])])if not np.isnan(row['rho_qs_m']) else np.nan, axis = 1 )
            df2[col + '_qs_s'] = df2.apply(lambda row: np.std(row[col][int(row['rho_qs_b']):int(row['rho_qs_f'])], ddof = 1)if not np.isnan(row['rho_qs_m']) else np.nan, axis = 1 )
            
            df2[col + '_qs2_m'] = df2.apply(lambda row: np.mean(row[col][int(row['rho_qs2_b']):int(row['rho_qs2_f'])])if not np.isnan(row['rho_qs2_m']) else np.nan, axis = 1 )
            df2[col + '_qs2_s'] = df2.apply(lambda row: np.std(row[col][int(row['rho_qs2_b']):int(row['rho_qs2_f'])], ddof = 1)if not np.isnan(row['rho_qs2_m']) else np.nan, axis = 1 )
            
            df2[col + '_qs3_m'] = df2.apply(lambda row: np.mean(row[col][int(row['rho_qs3_b']):int(row['rho_qs3_f'])])if not np.isnan(row['rho_qs3_m']) else np.nan, axis = 1 )
            df2[col + '_qs3_s'] = df2.apply(lambda row: np.std(row[col][int(row['rho_qs3_b']):int(row['rho_qs3_f'])], ddof = 1)if not np.isnan(row['rho_qs3_m']) else np.nan, axis = 1 )
            
            df2[col + '_qs4_m'] = df2.apply(lambda row: np.mean(row[col][int(row['rho_qs4_b']):int(row['rho_qs4_f'])])if not np.isnan(row['rho_qs4_m']) else np.nan, axis = 1 )
            df2[col + '_qs4_s'] = df2.apply(lambda row: np.std(row[col][int(row['rho_qs4_b']):int(row['rho_qs4_f'])], ddof = 1)if not np.isnan(row['rho_qs4_m']) else np.nan, axis = 1 )
            
    if drop_list_columns:
        if drop_list_columns is True:
            drop_list_columns = ['cur_steps', 'rho', 'Lplus', 'tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7']
        
        if leave_correct:
            inds = [row.cur_steps[-1] == row.steps for ind,row in df2.iterrows()]
            df2 = df2.iloc[inds]
            df2.reset_index(inplace=True, drop=True)
        
        df2.drop(drop_list_columns, axis = 1, inplace = True)
    
    return df2
  
def get_quasilevels2_early20(df, calc_triads = True, drop_list_columns = False, leave_correct = False):
    return get_quasilevels2(df, calc_triads, drop_list_columns, leave_correct, 20)

def group_quasilevels(df, cols = ['q', 'p', 'ps'], calc_rho = True, calc_triads = True):
    mylist = dict()
    
    for group in df.groupby(cols):
        (group_label, df_temp) = group
        columns = [*cols, ]
        q, psb, ps, _ = group_label
        # q = df_temp.q
        # ps = df_temp.ps
        # psb = df_temp.p
        
        d = dict(zip(cols, group_label))
        
        if calc_rho:
            rho_exp = get_quasi_rho(psb,ps,q)
            
            reps = len(df_temp.rho_qs_m)
            
            to_average = np.sum([not ((np.isnan(val)) | (val == 1)) for val in df_temp.rho_qs_m])
            level = np.mean([val for val in df_temp.rho_qs_m if not ((np.isnan(val)) | (val == 1))])
            level_std = np.std([val for val in df_temp.rho_qs_m if not ((np.isnan(val)) | (val == 1))])
            sim_ave_std = np.sqrt(np.sum((df_temp.rho_qs_f - df_temp.rho_qs_b - 1) * df_temp.rho_qs_s**2))
            
            to_average2 = np.sum([not ((np.isnan(val)) | (val == 1)) for val in df_temp.rho_qs2_m])
            level2 = np.mean([val for val in df_temp.rho_qs2_m if not ((np.isnan(val)) | (val == 1))])
            level2_std = np.std([val for val in df_temp.rho_qs2_m if not ((np.isnan(val)) | (val == 1))])
            sim_ave_std2 = np.sqrt(np.sum((df_temp.rho_qs2_f - df_temp.rho_qs2_b - 1) * df_temp.rho_qs2_s**2))
            
            to_average3 = np.sum([not ((np.isnan(val)) | (val == 1)) for val in df_temp.rho_qs3_m])
            level3 = np.mean([val for val in df_temp.rho_qs3_m if not ((np.isnan(val)) | (val == 1))])
            level3_std = np.std([val for val in df_temp.rho_qs3_m if not ((np.isnan(val)) | (val == 1))])
            sim_ave_std3 = np.sqrt(np.sum((df_temp.rho_qs3_f - df_temp.rho_qs3_b - 1) * df_temp.rho_qs3_s**2))
            
            d.update({"repetitions": reps, "rho_lvl": level, "rho_lvl_std": level_std, "rho_lvl_valid": to_average, "rho_lvl2": level2, "rho_lvl2_std": level2_std, "rho_lvl2_valid": to_average2,"rho_lvl3": level3, "rho_lvl3_std": level3_std, "rho_lvl3_valid": to_average3, "rho_lvl_exp" : rho_exp})
            columns.extend(["repetitions", "rho_lvl", "rho_lvl_std", "rho_lvl_valid", "rho_lvl2", "rho_lvl2_std", "rho_lvl2_valid", "rho_lvl3", "rho_lvl3_std", "rho_lvl3_valid", "rho_lvl_exp"])
        
        if calc_triads:
            if "repetitions" in d:
                reps = d["repetitions"]
            else:
                reps = len(df_temp.tr0)
                d.update({"repetitions": reps,})
                columns.extend(["repetitions"])
            
            to_average = np.sum([not ((np.isnan(val)) | (val == 1)) for val in df_temp.tr7_qs_m])
            d.update({"tr_valid": to_average,})
            columns.extend(["tr_valid"])
            
            for i in range(0,8):
                tr_name = 'tr' + str(i)
                col = tr_name + '_qs_m'
                
                level = np.mean([val for val in df_temp[col] if not ((np.isnan(val)) | (val == 1))])
                level_std = np.std([val for val in df_temp[col] if not ((np.isnan(val)) | (val == 1))])
                d.update({tr_name + "_lvl": level, tr_name + "_lvl_std": level_std})
                columns.extend([tr_name + "_lvl", tr_name + "_lvl_std"])
                
                # print("h")
                # print
            
            for i in range(0,8):
                tr_name = 'tr' + str(i)
                col = tr_name + '_qs3_m'
                
                level = np.mean([val for val in df_temp[col] if not ((np.isnan(val)) | (val == 1))])
                level_std = np.std([val for val in df_temp[col] if not ((np.isnan(val)) | (val == 1))])
                d.update({tr_name + "_lvl3": level, tr_name + "_lvl3_std": level_std})
                columns.extend([tr_name + "_lvl3", tr_name + "_lvl3_std"])
            # print(d)
        
        mylist.update({group_label: d})
    
    return pd.DataFrame(data = mylist.values(), index = mylist.keys(), columns = columns)

def group_quasilevels2(df, cols = ['q', 'p', 'ps'], calc_rho = True, calc_triads = True):
    """Difference with the above is that if a method is unable to get non-NaN values for all measures (rho, tr_i's), 
    then this row is NaN and is not counted. In the above method, we find non-NaN values separatly and average over them. 
    This may produce errors. 

    Args:
        df (_type_): _description_
        cols (list, optional): _description_. Defaults to ['q', 'p', 'ps'].
        calc_rho (bool, optional): _description_. Defaults to True.
        calc_triads (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    mylist = dict()
    
    for group in df.groupby(cols):
        (group_label, df_temp) = group
        columns = [*cols, ]
        q, psb, ps, _ = group_label
        # q = df_temp.q
        # ps = df_temp.ps
        # psb = df_temp.p
        
        d = dict(zip(cols, group_label))
        
        proper_inds1 = np.copy(df_temp.index)
        proper_inds2 = np.copy(df_temp.index)
        proper_inds3 = np.copy(df_temp.index)
        proper_inds4 = np.copy(df_temp.index)
        if calc_rho:
            proper_inds1 = [ind for ind in proper_inds1 if not ((np.isnan(df_temp.loc[ind].rho_qs_m)) | (df_temp.loc[ind].rho_qs_m == 1)) ]
            proper_inds2 = [ind for ind in proper_inds2 if not ((np.isnan(df_temp.loc[ind].rho_qs2_m)) | (df_temp.loc[ind].rho_qs2_m == 1)) ]
            proper_inds3 = [ind for ind in proper_inds3 if not ((np.isnan(df_temp.loc[ind].rho_qs3_m)) | (df_temp.loc[ind].rho_qs3_m == 1)) ]
            if "rho_qs4_m" in df_temp.columns:
                proper_inds4 = [ind for ind in proper_inds4 if not ((np.isnan(df_temp.loc[ind].rho_qs4_m)) | (df_temp.loc[ind].rho_qs4_m == 1)) ]
        
        if calc_triads:
            for i in range(0,8):
                tr_name = 'tr' + str(i)
                
                col = tr_name + '_qs_m'
                proper_inds1 = [ind for ind in proper_inds1 if not ((np.isnan(df_temp.loc[ind][col])) | (df_temp.loc[ind][col] == 1)) ]
                
                # col = tr_name + '_qs2_m'
                # proper_inds2 = [ind for ind in proper_inds2 if not ((np.isnan(df_temp.loc[ind][col])) | (df_temp.loc[ind][col] == 1)) ]
                
                col = tr_name + '_qs3_m'
                proper_inds3 = [ind for ind in proper_inds3 if not ((np.isnan(df_temp.loc[ind][col])) | (df_temp.loc[ind][col] == 1)) ]
                
                if "rho_qs4_m" in df_temp.columns:
                    col = tr_name + '_qs4_m'
                    proper_inds4 = [ind for ind in proper_inds4 if not ((np.isnan(df_temp.loc[ind][col])) | (df_temp.loc[ind][col] == 1)) ]
        
        
        # print([len(proper_inds1), len(proper_inds2), len(proper_inds3)])
        
        if calc_rho:
            rho_exp = get_quasi_rho(psb,ps,q)
            
            reps = len(df_temp.rho_qs_m)
            
            to_average = len(proper_inds1)
            to_average_2 = np.sum(df_temp.loc[proper_inds1].rho_qs_f - df_temp.loc[proper_inds1].rho_qs_b - 1)
            level = np.mean([val for val in df_temp.loc[proper_inds1].rho_qs_m])
            level_std = np.std([val for val in df_temp.loc[proper_inds1].rho_qs_m ])
            sim_ave_std = np.sqrt(np.sum((df_temp.loc[proper_inds1].rho_qs_f - df_temp.loc[proper_inds1].rho_qs_b - 1) * 
                                         df_temp.loc[proper_inds1].rho_qs_s**2) / to_average_2) / to_average
            
            to_average2 = len(proper_inds2)
            to_average2_2 = np.sum(df_temp.loc[proper_inds2].rho_qs2_f - df_temp.loc[proper_inds2].rho_qs2_b - 1)
            level2 = np.mean([val for val in df_temp.loc[proper_inds2].rho_qs2_m ])
            level2_std = np.std([val for val in df_temp.loc[proper_inds2].rho_qs2_m])
            sim_ave_std2 = np.sqrt(np.sum((df_temp.loc[proper_inds2].rho_qs2_f - df_temp.loc[proper_inds2].rho_qs2_b - 1) * 
                                         df_temp.loc[proper_inds2].rho_qs2_s**2)/ to_average2_2) / to_average2
            
            to_average3 = len(proper_inds3)
            to_average3_2 = np.sum(df_temp.loc[proper_inds3].rho_qs3_f - df_temp.loc[proper_inds3].rho_qs3_b - 1)
            level3 = np.mean([val for val in df_temp.loc[proper_inds3].rho_qs3_m ])
            level3_std = np.std([val for val in df_temp.loc[proper_inds3].rho_qs3_m])
            sim_ave_std3 = np.sqrt(np.sum((df_temp.loc[proper_inds3].rho_qs3_f - df_temp.loc[proper_inds3].rho_qs3_b - 1) * 
                                         df_temp.loc[proper_inds3].rho_qs3_s**2)/ to_average3_2) / to_average3
            
            d.update({"repetitions": reps, "rho_lvl": level, "rho_lvl_std": level_std, "rho_lvl_valid": to_average, "rho_sim_std": sim_ave_std,
                      "rho_lvl2": level2, "rho_lvl2_std": level2_std, "rho_lvl2_valid": to_average2, "rho_sim_std2": sim_ave_std2,
                      "rho_lvl3": level3, "rho_lvl3_std": level3_std, "rho_lvl3_valid": to_average3, "rho_sim_std3": sim_ave_std3,
                      "rho_lvl_exp" : rho_exp})
            columns.extend(["repetitions", "rho_lvl", "rho_lvl_std", "rho_lvl_valid", "rho_sim_std",
                            "rho_lvl2", "rho_lvl2_std", "rho_lvl2_valid", "rho_sim_std2",
                            "rho_lvl3", "rho_lvl3_std", "rho_lvl3_valid", "rho_sim_std3", "rho_lvl_exp"])
            
            if "rho_qs4_m" in df_temp.columns:
                to_average4 = len(proper_inds4)
                level4 = np.mean([val for val in df_temp.loc[proper_inds4].rho_qs4_m ])
                level4_std = np.std([val for val in df_temp.loc[proper_inds4].rho_qs4_m])
                to_average4_2 = np.sum(df_temp.loc[proper_inds4].rho_qs4_f - df_temp.loc[proper_inds4].rho_qs4_b - 1)
                # Variance from multiple series is weighted average. Weight is given how long the series is. 
                # The more series we have, the smaller the variance. 
                sim_ave_std4 = np.sqrt(np.sum((df_temp.loc[proper_inds4].rho_qs4_f - df_temp.loc[proper_inds4].rho_qs4_b - 1) * 
                                            df_temp.loc[proper_inds4].rho_qs4_s**2) / to_average4_2 ) / to_average4
                
                d.update({"rho_lvl4": level4, "rho_lvl4_std": level4_std, "rho_lvl4_valid": to_average4, "rho_sim_std4": sim_ave_std4})
                columns.extend([ "rho_lvl4", "rho_lvl4_std", "rho_lvl4_valid", "rho_sim_std4"])
            
            
        
        if calc_triads:
            if "repetitions" in d:
                reps = d["repetitions"]
            else:
                reps = len(df_temp.tr0)
                d.update({"repetitions": reps,})
                columns.extend(["repetitions"])
            
            if not ("rho_lvl_valid" in d):
                to_average = len(proper_inds1)
                to_average2 = len(proper_inds2)
                to_average3 = len(proper_inds3)
                to_average4 = len(proper_inds4)
                
                d.update({"rho_lvl_valid": to_average, "rho_lvl2_valid": to_average2,"rho_lvl3_valid": to_average3})
                columns.extend(["rho_lvl_valid", "rho_lvl2_valid", "rho_lvl3_valid"])
                
                if "rho_qs4_m" in df_temp.columns:
                    d.update({"rho_lvl4_valid": to_average4})
                    columns.extend([ "rho_lvl4_valid"])
        
            proper_inds_s = [proper_inds1, proper_inds3]
            algs_num = ["", "3"]
            
            if "rho_qs4_m" in df_temp.columns:
                proper_inds_s.append(proper_inds4)
                algs_num.append("4")
            
            for proper_inds, alg_num in zip(proper_inds_s, algs_num):
                for i in range(0,8):
                    tr_name = 'tr' + str(i)
                    col_m = tr_name + '_qs' + alg_num + '_m'
                    col_f = 'rho_qs' + alg_num + '_f'
                    col_b = 'rho_qs' + alg_num + '_b'
                    col_s = tr_name + '_qs' + alg_num + '_s'
                    
                    to_average_2 = np.sum(df_temp.loc[proper_inds1][col_f] - df_temp.loc[proper_inds1][col_b] - 1)
                    
                    level = np.mean([val for val in df_temp.loc[proper_inds][col_m] ])
                    level_std = np.std([val for val in df_temp.loc[proper_inds][col_m] ])
                    sim_ave_std = np.sqrt(np.sum((df_temp.loc[proper_inds][col_f] - df_temp.loc[proper_inds][col_b] - 1) * 
                                            df_temp.loc[proper_inds][col_s]**2) / to_average_2) / len(proper_inds)
                    d.update({tr_name + "_lvl" + alg_num: level, tr_name + "_lvl" + alg_num + "_std": level_std, 
                              tr_name + "_sim_std" + alg_num : sim_ave_std})
                    columns.extend([tr_name + "_lvl" + alg_num, tr_name + "_lvl" + alg_num + "_std", tr_name + "_sim_std" + alg_num])
            
            # for i in range(0,8):
            #     tr_name = 'tr' + str(i)
            #     col = tr_name + '_qs_m'
                
            #     level = np.mean([val for val in df_temp.loc[proper_inds1][col] ])
            #     level_std = np.std([val for val in df_temp.loc[proper_inds1][col] ])
            #     sim_ave_std = np.sqrt(np.sum((df_temp.loc[proper_inds1][tr_name + '_qs_f'] - df_temp.loc[proper_inds1][tr_name + '_qs_b'] - 1) * 
            #                              df_temp.loc[proper_inds1][tr_name + '_qs_s']**2))
            #     d.update({tr_name + "_lvl": level, tr_name + "_lvl_std": level_std, tr_name + "_sim_std" : sim_ave_std})
            #     columns.extend([tr_name + "_lvl", tr_name + "_lvl_std", tr_name + "_sim_std"])
                
            #     # print("h")
            #     # print
            
            # for i in range(0,8):
            #     tr_name = 'tr' + str(i)
            #     col = tr_name + '_qs3_m'
                
            #     level = np.mean([val for val in df_temp.loc[proper_inds3][col] ])
            #     level_std = np.std([val for val in df_temp.loc[proper_inds3][col] ])
            #     sim_ave_std = np.sqrt(np.sum((df_temp.loc[proper_inds3][tr_name + '_qs3_f'] - df_temp.loc[proper_inds3][tr_name + '_qs3_b'] - 1) * 
            #                              df_temp.loc[proper_inds3][tr_name + '_qs3_s']**2))
            #     d.update({tr_name + "_lvl3": level, tr_name + "_lvl3_std": level_std, tr_name + "_sim_std3" : sim_ave_std})
            #     columns.extend([tr_name + "_lvl3", tr_name + "_lvl3_std", tr_name + "_sim_std3"])
                
            # if "rho_qs4_m" in df_temp.columns:
            #     for i in range(0,8):
            #         tr_name = 'tr' + str(i)
            #         col = tr_name + '_qs4_m'
                    
            #         level = np.mean([val for val in df_temp.loc[proper_inds4][col] ])
            #         level_std = np.std([val for val in df_temp.loc[proper_inds4][col] ])
            #         sim_ave_std = np.sqrt(np.sum((df_temp.loc[proper_inds4][tr_name + '_qs4_f'] - df_temp.loc[proper_inds4][tr_name + '_qs4_b'] - 1) * 
            #                              df_temp.loc[proper_inds4][tr_name + '_qs4_s']**2))
            #         d.update({tr_name + "_lvl4": level, tr_name + "_lvl4_std": level_std, tr_name + "_sim_std4" : sim_ave_std})
            #         columns.extend([tr_name + "_lvl4", tr_name + "_lvl4_std", tr_name + "_sim_std4"])
            # print(d)
        
        mylist.update({group_label: d})
    
    return pd.DataFrame(data = mylist.values(), index = mylist.keys(), columns = columns)


def get_quasilevels_from_multiple_files(in_files, out_files, merge_to_one = True, 
                                        drop_list_columns = False, leave_correct = False, 
                                        save_intermediate_steps = False, get_quasilevels_fun = get_quasilevels,
                                        check_if_outfiles_exist = False, use_existing_qs_levels = False):
    
    print("Start processing...", flush = True)
    
    file_exists = False
    if ((not merge_to_one) or save_intermediate_steps) and check_if_outfiles_exist:
        if not merge_to_one:
            path = Path(out_files[0])
        else:
            path = Path(save_intermediate_steps[0])
        
        file_exists = Path(path).is_file()
    
    filter_get_qs_pvalue_warnings()
    
    if (not file_exists) or (use_existing_qs_levels):
        if use_existing_qs_levels:
            df2_existing = pd.read_hdf(Path(out_files[0]), key = "df2_rho")
            df2_rho = get_quasilevels_fun(df2_existing, drop_list_columns=drop_list_columns, leave_correct = leave_correct)
        else:
            in_file = in_files[0]
            df_rho_temp = pd.read_hdf(in_file, key = "df_rho")
            df2_rho = get_quasilevels_fun(df_rho_temp, drop_list_columns=drop_list_columns, leave_correct = leave_correct)
    
        if (not merge_to_one) or save_intermediate_steps:
            if not merge_to_one:
                path = Path(out_files[0])
            else:
                path = Path(save_intermediate_steps[0])
            df2_rho.to_hdf(path, key = 'df2_rho',mode = "w")
        
    print("Finished processing one file out of " + str(len(in_files)), flush = True)
    
    for i in range(1,len(in_files)):
        
        file_exists = False
        if ((not merge_to_one) or save_intermediate_steps) and check_if_outfiles_exist:
            if not merge_to_one:
                path = Path(out_files[i])
            else:
                path = Path(save_intermediate_steps[i])
            
            file_exists = Path(path).is_file()
        
        if (not file_exists) or use_existing_qs_levels:
            if use_existing_qs_levels:
                df2_existing = pd.read_hdf(Path(out_files[i]), key = "df2_rho")
                df2_rho_temp = get_quasilevels_fun(df2_existing, drop_list_columns=drop_list_columns, leave_correct = leave_correct)
            else:
                in_file = in_files[i]
                
                df_rho_temp = pd.read_hdf(in_file, key = "df_rho")
                df2_rho_temp = get_quasilevels_fun(df_rho_temp, drop_list_columns=drop_list_columns, leave_correct = leave_correct)
            
            print("Finished processing file no " + str(i+1) + " out of " + str(len(in_files)), flush = True)
            
            if (not merge_to_one) or save_intermediate_steps:
                if not merge_to_one:
                    out_file = out_files[i]
                    path = Path(out_file)
                else:
                    path = Path(save_intermediate_steps[i])
                df2_rho_temp.to_hdf(path, key = 'df2_rho', mode = "w")
            if merge_to_one:
                df2_rho = pd.concat([df2_rho, df2_rho_temp])
    warnings.resetwarnings()
    
    if merge_to_one:
        df2_rho.reset_index(inplace=True, drop=True)
        
        path = Path(out_files[0])
        df2_rho.to_hdf(path, key = 'df2_rho',mode = "w")
        
        return df2_rho

def merge_df2_files(files, drop_list_columns = True, save = False,leave_correct = False):
    """Merges separate df2_rho files. If drop_list_columns is set, 
    then columns with lists are dropped. 
    

    Args:
        files (_type_): _description_
        drop_list_columns (bool, optional): _description_. Defaults to True.
        leave_correct (bool): leaves rows with sims that did not stop early. 
    """
    
    df2_rho = pd.read_hdf(files[0], key = "df2_rho")
    
    if leave_correct:
        inds = [row.cur_steps[-1] == row.steps for ind,row in df2_rho.iterrows()]
        df2_rho = df2_rho.iloc[inds]
    
    if drop_list_columns:
        if drop_list_columns is True:
            drop_list_columns = ['cur_steps', 'rho', 'Lplus', 'tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7']
        df2_rho.drop(columns = drop_list_columns, inplace = True)
    
    for i, file in enumerate(files[1:]):
        df2_rho_temp = pd.read_hdf(file, key = "df2_rho")
        
        if leave_correct:
            inds = [row.cur_steps[-1] == row.steps for ind,row in df2_rho_temp.iterrows()]
            df2_rho_temp = df2_rho_temp.iloc[inds]
        
        if drop_list_columns:
            if drop_list_columns is True:
                drop_list_columns = ['cur_steps', 'rho', 'Lplus', 'tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7']
            df2_rho_temp.drop(columns = drop_list_columns, inplace = True)
        
        df2_rho = pd.concat([df2_rho, df2_rho_temp], ignore_index = True)
    
    df2_rho.reset_index(inplace=True, drop=True)
    
    if save:
        df2_rho.to_hdf(save, key="df2_rho",mode = "w")  
    
    return df2_rho  

def group_quasilevels_of_multiple_files(in_files_s, out_files, grouping_fun = group_quasilevels2, 
                                        reset_drop_index = True, only_correct_rows = True,
                                        correct_num_errors = True, remove_nan_rows = False):
    """Function to group output of the post-simulation analysis of separate simulations. 

    Args:
        in_files (_type_): It can be a list or a list of lists containing filenames of df2_rho that can be read, merged (if many) and grouped
        out_files (_type_): list of strings where outputs are to be saved
        grouping_fun (function, optional): Function grouping rows. Defaults to group_quasilevels2.
        reset_drop_index (bool, optional): Should index be reseted and dropped. Defaults to True.
        correct_num_errors (bool, optional): Should the possible numerical errors be removed. Defaults to True.
    """
    
    for (in_files, out_file) in zip(in_files_s, out_files):
        
        if isinstance(in_files, list):
            df2_rhos = [pd.read_hdf(in_file, key = "df2_rho") for in_file in in_files]
            df2_rho = pd.concat(df2_rhos)
        else:
            in_file = in_files
            df2_rho = pd.read_hdf(in_file, key = "df2_rho")
        
        if reset_drop_index:
            df2_rho.reset_index(inplace=True, drop=True)
            df2_rho.drop(columns="index", inplace=True)
            
        if correct_num_errors:
            for ind, row in df2_rho.iterrows():
                q = round(row.q, 5)
                psbt = round(row.p, 5)
                pst = round(row.ps, 5)
                rho_init = round(row.rho_init, 5)
                
                df2_rho.at[ind, "q"] = q
                df2_rho.at[ind, "p"] = psbt
                df2_rho.at[ind, "ps"] = pst
                df2_rho.at[ind, "rho_init"] = rho_init
                
        if remove_nan_rows:
            non_nan_cols = ['rho_qs_m', 'rho_qs2_m', 'rho_qs3_m']
            if 'rho_qs4_m' in df2_rho.columns:
                non_nan_cols.append('rho_qs4_m')
            nan_inds = df2_rho[non_nan_cols].apply(lambda row: np.all(np.isnan(row)), axis=1)
            df2_rho.drop(np.where(nan_inds)[0], inplace=True)
            df2_rho.reset_index(inplace=True, drop=True)
        
        if only_correct_rows:
            df2_rho_correct = df2_rho.iloc[[row.cur_steps[-1] == row.steps for ind,row in df2_rho.iterrows()]] 
            
            df2_rho_correct_g = grouping_fun(df2_rho_correct, cols = ['q', 'p', 'ps', 'rho_init'])
            
            path = Path(out_file)
            df2_rho_correct_g.to_hdf(path, key = 'df2_rho_correct_g', mode = "w")
        else:
            df2_rho_g = grouping_fun(df2_rho, cols = ['q', 'p', 'ps', 'rho_init'])
            
            path = Path(out_file)
            df2_rho_g.to_hdf(path, key = 'df2_rho_g', mode = "w")
    

def get_inds(df, sets, multipleOK = False):
    """Finds inds looking for indexes but not exact values
    """
    inds = []
    for set_ in sets:
        q, p, ps, rho_init = set_
        ind_p = np.where((np.abs(df.q - q) < 1e-8) & (np.abs(df.p - p) < 1e-8) & 
                         (np.abs(df.ps - ps) < 1e-8) & (np.abs(df.rho_init - rho_init) < 1e-8) )
        if len(ind_p[0]) == 0:
            print("No index for " + set_)
        else:
            if len(ind_p[0]) > 1:
                if not multipleOK:
                    print("Multiple points for " + str(set_))
            inds.extend(ind_p[0])
    return inds