"""Functions in this file generates links and agents lists"""

import numpy as np
from .agent import Agent


def add_connection(agent1, agent2):
    """
    Function used to establish connection between two agents.

    :param agent1: First agent of two to be connected
    :param agent2: Second agent of two to be connected
    """
    # Check if the connection wouldn't be duplicated
    if any([agent1.is_directed, agent2.is_directed]):
        assert not agent2.id in agent1.neighbours_ids, f'This connection already exists: {agent1.id}->{agent2.id}'
    else:
        assert not agent2.id in agent1.neighbours_ids, f'This connection already exists: {agent1.id}-{agent2.id}'
        assert not agent1.id in agent2.neighbours_ids, f'This connection already exists: {agent1.id}-{agent2.id}'




def generate_classic_er(agents_num, prob):
    """
    Generates Erdos-Renyi graph. It's done by generating random
    matrix of connections with random values (0;1) and changed
    to 0 or 1 if value is over expected probability. Values on
    diagonal and below it are omitted as it should be symmetric
    and has 0 on the diagonal.

    :param agents_num: Number of agents in the network
    :param prob: Probability of connection acceptance
    :return: (agents, links): list of agents and list of connections
    """
    if prob < 0 or prob > 1:
        raise NameError("Wrong prob value")

    agents = [Agent(i) for i in range(agents_num)]
    links = []

    connection_matrix = np.random.rand(agents_num, agents_num)
    connection_matrix = (connection_matrix < prob).astype(int)

    # iterate values over diagonal
    for i in range(agents_num):
        for j in range(i + 1, agents_num):
            if connection_matrix[i][j]:
                agents[i], agents[j] = add_connection(agents[i], agents[j])
                links.append([i, j])

    return agents, links


def generate_mc_er(agents_num: int, prob: float):
    """
    Generates Erdos-Renyi graph using Monte Carlo methods.

    param fixed_steps: stop after this number of steps
    :param agents_num: Number of agents in the network
    :param prob: Probability of connection acceptance
    :return: None
    """
    if prob < 0 or prob > 1:
        raise NameError("Wrong prob value")

    agents = [Agent(i) for i in range(agents_num)]
    links = []

    # Create connection matrix
    connection_matrix = np.zeros((agents_num, agents_num))

    # Matrix with counts of randomly picked elements from connection matrix
    count_matrix = np.zeros(connection_matrix.shape)

    # Array with over diagonal indices
    over_diagonal_mask = np.mask_indices(connection_matrix.shape[0], np.triu, 1)
    indices_matrix = np.zeros((connection_matrix.shape[0], connection_matrix.shape[1], 2))
    for i in range(connection_matrix.shape[0]):
        for j in range(connection_matrix.shape[1]):
            indices_matrix[i, j] = [i, j]
    over_diagonal_indices = indices_matrix[over_diagonal_mask]

    # iterate until all values over diagonal where picked at least once
    steps = 0
    while not all(count_matrix[over_diagonal_mask]):
        steps += 1
        r_indices = over_diagonal_indices[np.random.choice(over_diagonal_indices.shape[0])].astype(int)
        rind1, rind2 = r_indices
        count_matrix[rind1, rind2] += 1
        if prob > 0.5:
            if not connection_matrix[rind1, rind2]:
                connection_matrix[rind1, rind2] = 1
            elif connection_matrix[rind1, rind2] and np.random.random() < ((1 - prob) / prob):
                connection_matrix[rind1, rind2] = 0

        else:
            if connection_matrix[rind1, rind2]:
                connection_matrix[rind1, rind2] = 0
            elif not connection_matrix[rind1, rind2] and np.random.random() < (prob / (1 - prob)):
                connection_matrix[rind1, rind2] = 1

    for i in range(agents_num):
        for j in range(i + 1, agents_num):
            if connection_matrix[i][j]:
                agents[i], agents[j] = add_connection(agents[i], agents[j])
                links.append([i, j])

    return agents, links


def generate_ba(m: int, m0: int, agents_num: int):
    """
    Generating Barabasi-Albert graph.

    :param agents_num: final number of agents
    :param m: m parameter of BA-graph
    :param m0: initial cluster size
    :return: agents and links of the graph
    """
    agents = []
    links = []

    if m > m0:
        raise NameError("Parameter error: m is lower than m0")
    if agents_num < m0:
        raise NameError("Parameter error: number of agents is lower than m0")

    # Add initial agents fully connected:
    for mm in range(m0):
        agents.append(Agent(mm))
        for i in range(len(agents[:-1])):
            agents[mm], agents[i] = add_connection(agents[mm], agents[i])
            links.append([mm, i])

    while len(agents) <= agents_num:
        # Add agents accordingly to Barabasi-Albert algorithm
        k = np.array([len(a) for a in agents])
        probs = k / np.sum(k)

        new_agent_neighbours = np.random.choice(np.arange(k.size), m, replace=False, p=probs)
        agents.append(Agent(k.size))
        for new_neighbour_id in new_agent_neighbours:
            agents[-1], agents[new_neighbour_id] = add_connection(agents[-1], agents[new_neighbour_id])
            links.append([len(agents)-1, new_neighbour_id])

    return agents, links
