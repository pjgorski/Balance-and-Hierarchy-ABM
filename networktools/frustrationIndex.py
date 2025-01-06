# UBQP model with L(G) as the optimal objective function
# Based on the quadratic model discussed in the following publications
# http://doi.org/10.1007/978-3-319-94830-0_3
# http://arxiv.org/pdf/1611.09030

# This code solves graph optimization model(s) using "Gurobi solver"
# to compute the frustration index for the input signed graph(s) as well as the optimal partitioning of the nodes
# into two groups such that the total number of intra-group negative and inter-group positive edges are minimized.

# Note that you must have installed Gurobi into Jupyter and registered a Gurobi license
# in order to be able to run this code

import time
import numpy as np
import multiprocessing
from gurobipy import *
import networkx as nx
import random


def frustration_index(signedMatrix):
    """
    Compute the frustration index for the input signed graph(s) as well as the optimal partitioning of the nodes


    Parameters
    ----------
    signedMatrix : numpy array
        the adj. matrix of the signed network
    
    Returns
    -------
    frustration_index : float
        the frustration index of the input signed graph
    effectiveBranchingFactors : ???
        ???
    solveTime : float
        the time required to solve the optimization model    
    """

    G = nx.from_numpy_array(signedMatrix)
    # signedMatrix = nx.to_numpy_matrix(G)
    # unsignedMatrix = abs(signedMatrix)

    weighted_edges = nx.get_edge_attributes(G, 'weight') 
    sorted_weighted_edges = {}
    for (u,v) in weighted_edges:
        if u<v:
            (sorted_weighted_edges)[(u,v)] = weighted_edges[(u,v)]
        if u>v:
            (sorted_weighted_edges)[(v,u)] = weighted_edges[(u,v)]

    order=len(signedMatrix)
    size=int(np.count_nonzero(signedMatrix)/2)

    neighbors={}
    degree = []

    for u in sorted((G).nodes()):
        neighbors[u] = list((G)[u])
        degree.append(len(neighbors[u]))
    unsigned_degree = degree
    
    #fixing node is based on unsigned degree
    maximum_degree = max(unsigned_degree)
    [node_to_fix] = [([i for i, j in enumerate(unsigned_degree) if j == maximum_degree]).pop()]

    # Model parameters
    model = Model("UBQP")
    # What is the time limit in second?
    #model.setParam('TimeLimit', 10*3600)
    # Do you want details of branching to be reported? (0=No, 1=Yes)
    #model.setParam(GRB.param.OutputFlag, 1) 
    # Do you want a non-zero Mixed integer programming tolerance (MIP Gap)?
    # Note that a non-zero MIP gap may prevent the model from computing the exact value of frustration index
    #model.setParam('MIPGap', 0.0001)         
    # How many threads to be used for exploring the feasible space in parallel? Currently it is set to the maximum available
    model.setParam(GRB.Param.Threads, multiprocessing.cpu_count())     
    

    # Create decision variables and update model to integrate new variables
    x=[]
    for i in range(0,order):
        x.append(model.addVar(vtype=GRB.BINARY, name='x'+str(i))) # arguments by name
    model.update()

    # Set the objective function
    OFV=0
    for (i,j) in (sorted_weighted_edges):
        OFV = OFV + (1-(sorted_weighted_edges)[(i,j)])/2 + \
        ((sorted_weighted_edges)[(i,j)])*(x[i]+x[j]-2*x[i]*x[j]) 
    model.setObjective(OFV, GRB.MINIMIZE)

    # Add constraints to the model and update model to integrate new constraints
    model.addConstr(x[node_to_fix]==1)
    model.update() 

    # Solve
    start_time = time.time()
    model.optimize()
    solveTime = time.time() - start_time
    
    
    # Save optimal objective function values
    obj = model.getObjective()
    objectivevalue = obj.getValue()
    
    # Compute the effective branching factors
    effectiveBranchingFactors = 0
    if (model.NodeCount)**(1/(size+2*order)) >= 1:
        effectiveBranchingFactors = (model.NodeCount)**(1/(size+2*order))

    return objectivevalue, effectiveBranchingFactors, solveTime