import numpy as np
import networkx as nx
import copy
import itertools
import multiprocessing


def estrada_balance(arr):
    """ 
    Estrada index of balance. Walk-based measure of balance. Considers all closed walks. Normalises all walks of size k by k!.

    Parameters:
    arr (np.array): Adjacency matrix of graph

    Returns:
    float: Estrada index of balance
    """

    eig_sign = np.linalg.eigvals(arr)
    print(eig_sign)
    eig_unsign = np.linalg.eigvals(abs(arr))
    exp_s = [np.exp(x) for x in eig_sign]
    print(exp_s)
    exp_us = [np.exp(x) for x in eig_unsign]
    print(exp_us)
    K = sum(exp_s)/sum(exp_us)
    print(K)
    res = (1-K)/(1+K)
    res2 = (K+1)/2
    return ([res,res2])


def algebraic_conflict(arr):
    """
    Returns the algebraic conflict index of a graph. Calculated via the smallest eigenvalue of the signed laplacian matrix.

    Parameters:
    arr (numpy array): Adjacency matrix of a graph

    Returns:
    res (float): Algebraic conflict index of the graph
    """
    degs = [x[1] for x in list(nx.from_numpy_array(abs(arr)).degree())]
    laplacian = np.diag(degs) - arr
    eigvals = np.linalg.eigvals(laplacian)
    smallest_ev = min(eigvals)
    norm_mat = np.meshgrid(degs,degs)
    norm_mat = norm_mat[0] + norm_mat[1]
    norm_val = max(norm_mat[np.nonzero(arr)])/2 - 1
    res = 1 - smallest_ev/norm_val
    return(res)


def triangle_index(arr):
    """
    Calculates the triangle index of a graph. Defined as the number of balanced triangles dvivded by the number of total triangles.
    Done so via closed path of length three
    
    Parameters:
    arr (numpy array): Adjacency matrix of a graph

    Returns:
    trace_s (float): the triangle index
    """
    trace_s = np.trace(np.linalg.matrix_power(arr, 3))
    trace_us = np.trace(np.linalg.matrix_power(abs(arr), 3))
    trace_s = (trace_s + trace_us)/(2*trace_us)
    return(trace_s)


def splitArrayBySign(arr):
    # a function which returns two arrays, of the same shape as the input array, with the positive and negative values of the input array split into two arrays
    pos = np.zeros(arr.shape)
    neg = np.zeros(arr.shape)
    pos[arr>0] = arr[arr>0]
    neg[arr<0] = arr[arr<0]
    return(pos,neg)

def kirkleyNewman(arr, alpha, strong=True):
    """
    Calculates the Newman-Karrer index of a graph. Defined as a weighted sum of the number of unbalanced (negative) k-paths over all lengths k.
    Longer paths are weighted less (1/z^k). The parameter z is rescaled with the leading eigenvector alpha = z/lambda_P.
    
    Parameters:
    arr (numpy array): Adjacency matrix of a graph
    alpha>1 (float): parameter of the Newman-Karrer index, 1/ln(alpha) is a decay length, determines the scale on which the contributions form longer walks are discounted
    strong (bool): if True, the strong balance condition is used, otherwise the weak balance condition is used

    Returns:
    nk_index (float): the newman-karrer index
    """
    pos_arr, neg_arr = splitArrayBySign(arr)
    neg_arr = abs(neg_arr)

    if strong:
        lambda_1 = max(np.linalg.eigvals(pos_arr + neg_arr))
        lambda_2 = max(np.linalg.eigvals(pos_arr - neg_arr))
        lambda_P = max(lambda_1, lambda_2)
        I = np.identity(arr.shape[0])
        
        nk_index = np.linalg.det(alpha*lambda_P*I - (pos_arr - neg_arr))/np.linalg.det(alpha*lambda_P*I - (pos_arr + neg_arr))


        nk_index = (1/4)*np.log(nk_index)
    else:
        pos_arr, neg_arr = splitArrayBySign(arr)
        neg_arr = abs(neg_arr)
        lambda_P = max(np.linalg.eigvals(pos_arr))
        I = np.identity(arr.shape[0])
        
        inverse = np.linalg.inv(alpha*lambda_P*I - pos_arr)

        nk_index = 0.5*np.trace(np.dot(neg_arr, inverse))
    return(nk_index)
    
def kirkleyNewmanWeak(arr, alpha):
    """
    Calculates the Newman-Karrer index of a graph for weak balance. Defined as a weighted sum of the number of unbalanced (negative) k-paths over all lengths k.
    Longer paths are weighted less (1/z^k). The parameter z is rescaled with the leading eigenvector alpha = z/lambda_P.
    
    Parameters:
    arr (numpy array): Adjacency matrix of a graph
    alpha>1 (float): parameter of the Newman-Karrer index, 1/ln(alpha) is a decay length, determines the scale on which the contributions form longer walks are discounted

    Returns:
    nk_index (float): the newman-karrer index
    """
    pos_arr, neg_arr = splitArrayBySign(arr)
    neg_arr = abs(neg_arr)
    lambda_P = max(np.linalg.eigvals(pos_arr))
    I = np.identity(arr.shape[0])
    
    inverse = np.linalg.inv(alpha*lambda_P*I - pos_arr)

    nk_index = 0.5*np.trace(np.dot(neg_arr, inverse))
    return(nk_index)
    
def kirkleyNewmanStrong(arr, alpha):
    """
    Calculates the Newman-Karrer index of a graph for strong balance. Defined as a weighted sum of the number of unbalanced (negative) k-paths over all lengths k.
    Longer paths are weighted less (1/z^k). The parameter z is rescaled with the leading eigenvector alpha = z/lambda_P.
    
    Parameters:
    arr (numpy array): Adjacency matrix of a graph
    alpha>1 (float): parameter of the Newman-Karrer index, 1/ln(alpha) is a decay length, determines the scale on which the contributions form longer walks are discounted

    Returns:
    nk_index (float): the newman-karrer index
    """
    pos_arr, neg_arr = splitArrayBySign(arr)
    neg_arr = abs(neg_arr)
    lambda_1 = max(np.linalg.eigvals(pos_arr + neg_arr))
    lambda_2 = max(np.linalg.eigvals(pos_arr - neg_arr))
    lambda_P = max(lambda_1, lambda_2)
    I = np.identity(arr.shape[0])
    
    nk_index = np.linalg.det(alpha*lambda_P*I - (pos_arr - neg_arr))/np.linalg.det(alpha*lambda_P*I - (pos_arr + neg_arr))


    nk_index = (1/4)*np.log(nk_index)
    return(nk_index)


def kirkleyNewmanStrongNorm(arr, alpha):
    """
    Calculates the Newman-Karrer index of a graph for strong balance. Defined as a weighted sum of the number of unbalanced (negative) k-paths over all lengths k.
    Longer paths are weighted less (1/z^k). The parameter z is rescaled with the leading eigenvector alpha = z/lambda_P.
    
    Parameters:
    arr (numpy array): Adjacency matrix of a graph
    alpha>1 (float): parameter of the Newman-Karrer index, 1/ln(alpha) is a decay length, determines the scale on which the contributions form longer walks are discounted

    Returns:
    nk_index (float): the newman-karrer index
    """
    pos_arr, neg_arr = splitArrayBySign(arr)
    neg_arr = abs(neg_arr)
    lambda_1 = max(np.linalg.eigvals(pos_arr + neg_arr))
    lambda_2 = max(np.linalg.eigvals(pos_arr - neg_arr))
    lambda_P = max(lambda_1, lambda_2)
    I = np.identity(arr.shape[0])
    X = np.linalg.det(I - (pos_arr + neg_arr)/(alpha*lambda_P))
    Y = np.linalg.det(I - (pos_arr - neg_arr)/(alpha*lambda_P))
    
    nk_index = (np.log(X) + 0.5*np.log(Y/X))/np.log(X)
    
    return(nk_index)
    
def isBalancedTriads(arr):
    return triangle_index(arr) == 1


def isBalanced(arr, shape):
    degs = [x[1] for x in list(nx.from_numpy_array(abs(arr)).degree())]
    laplacian = np.diag(degs) - arr

    eigvals = np.linalg.eigvals(laplacian)
    eigvals = np.around(eigvals, decimals=10) # round to 5 decimals as the eigenvalue 0 is sometimes not exactly found

    #smallest_ev = min(eigvals)
    multiplicity_zeroEig = shape[0] - np.count_nonzero(eigvals)

    connected_components = list(nx.connected_components(nx.from_numpy_array(abs(arr))))
    n_components = len(connected_components)

    return n_components==multiplicity_zeroEig

def pointIndex_triads(arr):
    """
    Returns the point index of a graph. Iterates all possible node removals and calculates balance through the smallest eigenvalue. If it's zero, the graph is balanced. Must compute 2^n times the triangle index.

    Parameters:
    arr (numpy array): Adjacency matrix of a graph

    Returns:
    res (float): Point index of the graph
    """
    shape = arr.shape

    if isBalancedTriads(arr):
        return(0)
    
    if np.all(arr[np.nonzero(arr)] == -1):
        return(shape[0])


    res = shape[0]
    # config = None

    for i in range(1,shape[0] -1):
        if i>res:
            break

        for perm in itertools.combinations(range(shape[0]),i):

            arr2 = np.ones(shape)

            arr2[:,perm] = 0
            arr2[perm,:] = 0

            arr2 = np.multiply(arr, arr2)

            if np.all(arr2[np.nonzero(arr2)] == -1):
                continue

            if isBalancedTriads(arr2):
                res = i
                # config = perm
                break
    

    return(res)

def pointIndex_triads_parallel(Gs, n_cores = 4):
    pool = multiprocessing.Pool(n_cores)
    res = pool.map(pointIndex_triads, Gs)
    pool.close()
    pool.join()
    return res


def pointIndex(arr):
    """
    Returns the point index of a graph. Iterates all possible node removals and calculates balance through the smallest eigenvalue. If it's zero, the graph is balanced.

    Parameters:
    arr (numpy array): Adjacency matrix of a graph

    Returns:
    res (float): Point index of the graph
    """
    shape = arr.shape

    if isBalanced(arr, shape):
        return(0)
    
    if np.all(arr[np.nonzero(arr)]) == -1:
        return(shape[0])

    res = shape[0]
    # config = None

    for i in range(1,shape[0] + 1):
        if i>res:
            break

        for perm in itertools.combinations(range(shape[0]),i):

            arr2 = copy.deepcopy(arr)

            arr2[:,perm] = 0
            arr2[perm,:] = 0

            if isBalanced(arr2, shape):
                res = i
                # config = perm
                break
    

    return(res)