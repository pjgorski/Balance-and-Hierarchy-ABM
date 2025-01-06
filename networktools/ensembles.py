import numpy as np
from multiprocessing import Pool
import networkx as nx
import random


def sign_reshuffling_on_fixed_topology(topo, n_sim = 10, parallel=False, undirected= True, n_cores=6):
    """
    We generate a sequence of n_sim signed random random networks (adj. matrices).
    These networks have the same topology provided by topo (an adj. matrix)
    and the same number of positive and negative links.
    
    Parameters
    ----------
    topo : matrix like object
        the adj. matrix of the network
    n_sim : int
        number of generated random networks
    parallel : bool
        if True, we generate the random networks in parallel
    undirected : bool
        if True, we consider the upper triangle of the adj. matrix
    n_cores : int 
        number of cores to use

    Returns
    -------
    simulated_nets : list
        The list with the n_sim random networks (adj. matrices)
    """
    if undirected ==True:
        topo = np.triu(topo)
    ind_row, ind_col = np.nonzero(topo)
    shape = topo.shape
    signed_links = topo[(ind_row, ind_col)]
    if parallel == False:
        simulated_nets = [generate_sign_reshuffling_on_fixed_topology(shape, ind_row, ind_col, signed_links) for i in range(n_sim)]
    else:
        args = (shape,ind_row,ind_col,signed_links)
        simulated_nets = parallel_generation(generate_sign_reshuffling_on_fixed_topology, args, n_sim, n_cores=n_cores)
    return simulated_nets

def generate_sign_reshuffling_on_fixed_topology(shape, ind_row, ind_col, signed_links):
    """
    We generate a new network (adj. matrix) provided the info in the arguments dictionary.
    This dictionary should have  where the number of links is equal to positive
    and negative links in signed_links.

    Parameters
    ----------
    shape : tuple
        the shape of the adj. matrix of the network
    ind_row : list/array
        the row indices that are non-zero
    ind_col : list/array
        the column indices that are non-zero
    signed_links : list/array
        an array/list containing the positive (+1) and negative signs (-1)

    Returns
    -------
    new_topo : numpy matrix
        a new adj. matrix with the same topology as the original one, but shuffled signs
    """
    new_topo = np.zeros(shape)
    np.random.shuffle(signed_links)
    new_topo[(ind_row, ind_col)] = signed_links
    new_topo[(ind_col, ind_row)] = signed_links
    return new_topo

def parallel_generation(func, args, n_sim=10, n_cores=4):
    """
    We generate a sequence of n_sim random networks (adj. matrices) in parallel.

    Parameters
    ----------

    func : function 
        the function that generates the random networks
    args : tuple
        the arguments of the function
    n_sim : int
        number of generated random networks
    n_cores : int
        number of cores to use

    Returns
    -------
    sim_nets : list
        the list with the n_sim random networks (adj. matrices)
    """
    sim_nets = []
    with Pool(n_cores) as p:
        for net in p.starmap(func, [args for i in range(n_sim)]):
            sim_nets.append(net)
        return sim_nets


def powerlaw_cluster_reshuffling(A, n_sim = 10, parallel=False, n_cores=6):
    """
    We generate a sequence of n_sim signed random random networks (adj. matrices) from the powerlaw cluster model. (Holme and Kim, 2002).
    It preserves the number of edges, the number of positive and negative links and the clustering coefficient.

    Parameters
    ----------
    A : numpy array NxN
        the adj. matrix of the network
    n_sim : int
        number of generated random networks
    parallel : bool
        if True, we generate the random networks in parallel
    n_cores : int
        number of cores to use

    Returns
    -------
    simulated_nets : list
        The list with the n_sim random networks (adj. matrices)
    """
    G = nx.from_numpy_array(A)
    n = len(G.nodes())
    m = len(G.edges())

    m_p = (n + np.sqrt(n**2 - 4*m))/2
    m_m = (n - np.sqrt(n**2 - 4*m))/2
    m_vec = [m_p,m_m]

    if np.all(np.isnan(m_vec)):
        raise ValueError("All elements are NaN")
    m_fin = min([x for x in m_vec if x > 0])
    m_fin = int(np.round(m_fin))
    print(m_vec)

    p = nx.average_clustering(G)
    signs = list(nx.get_edge_attributes(G, 'weight').values())

    if parallel == False:    
        simulated_nets = [generate_powerlaw_cluster_reshuffling(n, m_fin, p, signs) for i in range(n_sim)]
    else:
        args = (n,m_fin,p,signs)
        simulated_nets = parallel_generation(generate_powerlaw_cluster_reshuffling, args, n_sim, n_cores=n_cores)
    return simulated_nets

    

def generate_powerlaw_cluster_reshuffling(n,m, p, signed_links):
    """
    We generate a new network (adj. matrix) from the powerlaw cluster model. (Holme and Kim, 2002).
    It preserves the number of edges, the number of positive and negative links and the clustering coefficient.
    
    Parameters
    ----------
    n : int
        number of nodes
    m : int
        number of edges
    p : float
        clustering coefficient of the new network
    signed_links : list/array
        an array/list containing the positive (+1) and negative signs (-1)

    Returns
    -------
    A : numpy array n x n
        a new adj. matrix with the same topology as the original one, but shuffled signs
    """
    shuffled_G = nx.powerlaw_cluster_graph(n,m,p=p)
    random.shuffle(signed_links)
    reshuffled_links = dict(zip(shuffled_G.edges, signed_links))
    nx.set_edge_attributes(shuffled_G, reshuffled_links, 'weight')

    A = nx.to_numpy_array(shuffled_G)

    return A


def gnm_reshuffling(A, n_sim = 10, parallel=False, n_cores=6):
    """
    We generate a sequence of n_sim signed random random networks (adj. matrices) from the G(n,m) model.

    Parameters
    ----------
    A : numpy matrix
        the adj. matrix of the network
    n_sim : int 
        number of generated random networks
    parallel: bool 
        if True, we use parallelization,
    n_cores: int
        number of cores to use for parallelization

    Returns
    -------
    simulated_nets : list
        the list with the n_sim random networks (adj. matrices)
    """
    ind = np.triu_indices_from(A, k=1)
    upper_1d = A[ind]
    #upper_1d = np.array(upper_1d)[0]
    shape = A.shape


    if parallel == False:    
        simulated_nets = [generate_gnm_reshuffling(upper_1d, shape, ind) for i in range(n_sim)]
    else:
        args = (upper_1d, shape, ind)
        simulated_nets = parallel_generation(generate_gnm_reshuffling, args, n_sim, n_cores=n_cores)
    return simulated_nets


def generate_gnm_reshuffling(upper_1d, shape, indices):
    """
    We generate a new network (adj. matrix) from the G(n,m) model.
    
    Parameters
    ----------
    upper_1d : numpy array
        the values of the upper triangular part of the adj. matrix that needs to shuffled
    shape : tuple
        the shape of the adj. matrix
    indices : tuple
        the indices of the upper triangular part of the adj. matrix

    Returns
    -------
    new_topo : numpy array
        the new adj. matrix   
    """
    new_topo = np.zeros(shape)
    np.random.shuffle(upper_1d)
    new_topo[indices] = upper_1d
    return new_topo + new_topo.T

def generate_signed_sbm_known_groups(groups, group_connectivity, p_plus):
    """
    An undirected signed network from SBM model is generated. 
    
    `groups` contains sets of nodes. Each set of nodes is one group. 
        Remark: currently, all nodes from 0 to N-1 should be distributed among the groups. 
    `groups` may also contain array of values, indicating number of nodes in each group with giving node indices. 
    `group_connectivity` is the connectivity between the groups. 
    `p_plus` are the probabilities of creating positive signs between and across groups. 
    
    Option 1: All groups are uniform, that is they have the same connectivity and positive link probabilities. 
    Then, `group_connectivity` and `p_plus` contain two values: `(din, dout)` and `(pin_plus, p_out_plus)`. 
    
    Option 2: All groups may be different. 
    Then, the dimension of `group_connectivity` and/or `p_plus` should be equal to the number of groups. 
    Remark: mind that `group_connectivity` should be symmetric. 

    Parameters
    ----------
    groups : list
        Array of arrays of nodes (when nodes' labels are important). Or array of group sizes. 
    group_connectivity : numpy array
        Two values indicating in- and out-group connectivity. 
        Or array of size number of groups with specific connectivities. 
    p_plus : numpy array
        Two values indicating pin+ and pout+ probabilities. 
        Or array of size number of groups with specific probabilities.
    
    Returns
    -------
    A : numpy array
        adj. matrix  
    """
    
    if np.isscalar(groups[0]):
        sizes = np.array(groups)
        actual_groups = [list(range(end_g - len_g,end_g)) for len_g, end_g in zip(sizes, np.cumsum(sizes))]
    else:
        actual_groups = groups
        sizes = np.array([len(g) for g in groups])
    N = np.sum(sizes)
    
    ng = len(sizes)
    # get dictionary which node belongs to which group. This will be useful in the last stage
    d = dict.fromkeys(range(0,N))
    for i, group in enumerate(actual_groups):
        for node in group:
            d[node] = i
    
    if len(group_connectivity) == 2:
        conn = np.ones((ng, ng)) * group_connectivity[1]
        np.fill_diagonal(conn, group_connectivity[0])
    else:
        conn = group_connectivity
    
    g = nx.stochastic_block_model(sizes, conn)
    A = nx.to_numpy_array(g)
    
    if len(p_plus) == 2:
        pos_probs = np.ones((ng, ng)) * p_plus[1]
        np.fill_diagonal(pos_probs, p_plus[0])
    else:
        pos_probs = p_plus
    
    for agent_i, agent_j in g.edges:
        group_i = d[agent_i]
        group_j = d[agent_j]
        
        sign = +1 if np.random.rand() < pos_probs[group_i, group_j]else -1
        A[agent_i, agent_j] = sign
        A[agent_j, agent_i] = sign
    
    return A

def signed_sbm_known_groups(groups, group_connectivity, p_plus, n_sim = 10, parallel=False, n_cores=6):
    """
    We generate a sequence of n_sim signed random networks (adj. matrices) from the stochastic block model. (Morrison, Gabbay, 2020).
    See `generate_signed_sbm_known_groups` for thorough description of parameters `groups`, `group_connectivity` and `p_plus`. 

    Parameters
    ----------
    groups : list
        Array of arrays of nodes (when nodes' labels are important). Or array of group sizes. 
    group_connectivity : numpy array
        Two values indicating in- and out-group connectivity. 
        Or array of size number of groups with specific connectivities. 
    p_plus : numpy array
        Two values indicating pin+ and pout+ probabilities. 
        Or array of size number of groups with specific probabilities.
    n_sim : int
        number of generated random networks
    parallel : bool
        if True, we generate the random networks in parallel
    n_cores : int
        number of cores to use

    Returns
    -------
    simulated_nets : list
        The list with the n_sim random networks (adj. matrices)
    """
    if parallel == False:    
        simulated_nets = [generate_signed_sbm_known_groups(groups, group_connectivity, p_plus) for i in range(n_sim)]
    else:
        args = (groups, group_connectivity, p_plus)
        simulated_nets = parallel_generation(generate_signed_sbm_known_groups, args, n_sim, n_cores=n_cores)
    return simulated_nets