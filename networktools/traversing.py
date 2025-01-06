import numpy as np

from .network import Network
from .agent import Agent


def breadth_first_search(net: Network):

    distances_matrix = np.zeros((len(net), len(net)))

    for i, agent in enumerate(net):
        is_checked = np.repeat(False, len(net))
        is_checked[i] = True

        to_check = [nid for nid in agent]
        stop_flag = False
        deg = 1

        while len(to_check):

            distances_matrix[i, to_check] = deg
            prev_to_check = list.copy(to_check)
            is_checked[to_check] = True
            deg += 1

            # add neighbours of previously checked agents
            to_check = []
            for prev_checked in prev_to_check:
                to_check.append(list(net[prev_checked].neighbours))
            # ravel `to_check` array
            to_check = [item for sublist in to_check for item in sublist]

            # drop duplicates
            to_check = list(set(to_check))

            # drop if have been checked before
            to_check = [unique for unique in to_check if not is_checked[unique]]

    return distances_matrix
