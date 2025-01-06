import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(A):
    """Plots the graph A"""
    G = nx.from_numpy_array(A)
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='w', edgecolors='k', node_size=500)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.axis('off')
    plt.show()


def plot_signed_graph(A):
    """Plots the signed graph A"""
    G = nx.from_numpy_array(A)
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G)
    edges = G.edges()
    edge_labels = nx.get_edge_attributes(G,'weight')
    colors = ['y' if G[u][v]['weight'] > 0 else 'r' for u,v in edges]
    nx.draw_networkx_nodes(G, pos, node_color='w', edgecolors='k', node_size=500)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color = colors)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.axis('off')
    plt.show()



def random_sign_switch(arr, p):
    """
    Randomly switches the sign of non-zero elements in the matrix based on the probability p
    """

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] != 0:
                if np.random.uniform(0,1) < p:
                    arr[i,j] = -arr[i,j]
    return arr




def create_powerlawClust_graph(n, m, p):
    """
    Creates a random graph with n nodes, m per node and p probability of rewiring
    """
    G = nx.powerlaw_cluster_graph(n,m,p=p)
    A = nx.to_numpy_array(G)
    return A

def create_gnm_graph(n_agents, m):
    """
    Creates a random graph with n_agents nodes and m edges
    """
    G = nx.gnm_random_graph(n_agents, m)
    A = nx.to_numpy_array(G)
    return A


def random_sign_switch_nedges(arr, n_edges):
    """
    Randomly switches the sign of n_edges non-zero elements in the matrix
    """
    new_topo = np.triu(arr)
    upper_tri_ind = np.nonzero(new_topo)
    indices = np.arange(len(upper_tri_ind[0]))
    rand_choice = np.random.choice(indices, n_edges, replace=False)
    new_topo[upper_tri_ind[0][rand_choice], upper_tri_ind[1][rand_choice]] *= -1
    return(new_topo+new_topo.T)


def calc_triangles(arr):
    """
    Calculates the number of triangles in the graph
    """
    return sum([x for x in nx.triangles(nx.from_numpy_array(arr)).values()])/3


# Creates a balanced graphs with two groups, leaving the holes (0) as they were in the previous matrix
def balanced_graph(arr):

    cutpoint = int(arr.shape[0]/2)
    print(cutpoint)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (i < cutpoint) & (j < cutpoint):
                if arr[i,j] != 0:
                    arr[i,j] = 1
            elif (i >= cutpoint) & (j >= cutpoint):
                if arr[i,j] != 0:
                    arr[i,j] = 1
            elif (i < cutpoint) & (j >= cutpoint):
                if arr[i,j] != 0:
                    arr[i,j] = -1
            elif (i >= cutpoint) & (j < cutpoint):
                if arr[i,j] != 0:
                    arr[i,j] = -1
    return arr


###################################
# Unsigned topologies to explore
###################################

def createOuterStar(n_agents):
    """Returns the adjacency matrix of a star graph with n_agents leaves"""
    circ_graph = nx.cycle_graph(n_agents)
    A = nx.to_numpy_array(circ_graph)
    node_i = [x for x in range(n_agents)]
    node_i = node_i + node_i
    node_i.sort()
    fin = node_i.pop(0)
    node_i.append(fin)

    node_j = [x for x in range(n_agents,2*n_agents)]
    node_j = node_j + node_j
    node_j.sort()
 
    edgs = [(i,j) for i,j in zip(node_i,node_j)]
    circ_graph.add_edges_from(edgs)
    edgs_dict = dict(zip(edgs, np.repeat(1, len(edgs))))
    nx.set_edge_attributes(circ_graph , edgs_dict, 'weight')
    circ_graph = nx.from_numpy_array(nx.to_numpy_array(circ_graph))
    A_circ = nx.to_numpy_array(circ_graph)

    return A_circ


def createRingReg(n_agents, n_neib, p = 0):
    """Returns the adjacency matrix of a ring regular graph with n_agents nodes and n_neib neighbors per node"""
    ring_regular = nx.to_numpy_array(nx.newman_watts_strogatz_graph(n_agents, n_neib, p))
    return ring_regular


def createInnerStar(n_agents):
    circ_graph = nx.cycle_graph(n_agents)
    nx.add_star(circ_graph, np.arange(0,n_agents), weight=1)
    circ_graph.add_edge(1,n_agents-1, weight=1)
    A = nx.to_numpy_array(circ_graph)
    return A


def createFullyConnected(n_agents):
    """Returns the adjacency matrix of a fully connected graph with n_agents nodes"""
    A = np.ones((n_agents,n_agents))
    A = A - np.identity(n_agents)
    return A