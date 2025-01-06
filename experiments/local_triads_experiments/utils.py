"""Utility functions and classes to run the dynamics"""
import numpy as np
from scipy import linalg


def get_trophic_levels(adjacency_matrix):
    agents_num, _ = adjacency_matrix.shape
    w_in = np.array([np.sum(adjacency_matrix[n, :]) for n in range(agents_num)])
    w_out = np.array([np.sum(adjacency_matrix[:, n]) for n in range(agents_num)])
    u = w_in + w_out
    v = w_in - w_out
    diag_u = np.diagflat(u)
    l = diag_u - adjacency_matrix - adjacency_matrix.T

    # return trophic levels
    return linalg.solve(l, v)


def get_tropic_incoherence(adjacency_matrix):
    agents_num, _ = adjacency_matrix.shape
    h = get_trophic_levels(adjacency_matrix)
    w_sum = np.sum(adjacency_matrix)
    nom = 0
    for n in range(agents_num):
        for m in range(agents_num):
            nom += adjacency_matrix[n, m] * pow(h[n] - h[m] - 1, 2)
    return nom/w_sum


if __name__ == '__main__':
    # Testing trophic levels
    # feedback loop:
    am = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    print(f'incoherence: {get_tropic_incoherence(am)}')
