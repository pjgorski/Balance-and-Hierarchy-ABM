"""BA network generator, return adjacency matrix"""
import numpy as np

import networkx as nx

from networktools.network_generators.decorator import register_generator


class RandomNetwork:
    def __init__(self):
        pass

    def get_adjacency_matrix(self):
        return self.adm

    @classmethod
    def name(cls) -> str:
        return cls.__name__


@register_generator
class BarabasiAlbert(RandomNetwork):
    def __init__(self, n_agents: int, m: int = 2, m0: int = 2):
        super().__init__()
        assert m0 >= m, "m0 should be greater or equal to m"
        m = int(m)
        m0 = int(m0)

        self.adm = np.zeros((n_agents, n_agents))
        self.k_vec = np.zeros(n_agents)

        # initialize
        for i in range(m0):
            for j in range(i, m0):
                self.adm[i, j] = 1

        self.map_connections_symmetrically()
        self.update_degrees()

        for new_agent in range(m0, n_agents):
            probs = self.k_vec / np.sum(self.k_vec)
            chosen = np.random.choice(range(n_agents), m, p=probs, replace=False)
            for old_agent in chosen:
                self.adm[old_agent, new_agent] = 1
            self.map_connections_symmetrically()
            self.update_degrees()

    def update_degrees(self):
        self.k_vec = np.array([np.sum(self.adm[i, :] + self.adm[:, i]) for i in range(self.adm.shape[0])])

    def map_connections_symmetrically(self):
        triu_ind = np.triu_indices(self.adm.shape[0], k=1)
        tril_ind = np.tril_indices(self.adm.shape[0], k=-1)
        self.adm[tril_ind] = self.adm[triu_ind]


@register_generator
class ErdosRenyi(RandomNetwork):
    def __init__(self,
                 n_agents: int,
                 prob: float = .15):
        self.adm = nx.to_numpy_matrix(nx.erdos_renyi_graph(n_agents, prob))


@register_generator
class WattsStrogatz(RandomNetwork):
    def __init__(self, n_agents: int, k, prob):
        super().__init__()
        self.adm = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(n_agents, k, prob)))
