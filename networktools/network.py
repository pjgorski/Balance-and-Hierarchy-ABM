from typing import List

from .network_generators_old import *
from networktools.network_generators import generator


class Network:
    def __init__(self, mode: str = None, **kwargs):
        if not any([mode == x for x in generator]) and mode is not None:
            raise ValueError(f'Unrecognised network creation method: \'{mode}\'')

        if mode:
            self.agents, self.links = generator[mode](**kwargs)()

        else:
            self.agents = []
            self.links = []

    def __getitem__(self, key):
        return self.agents[key]

    def __len__(self):
        return len(self.agents)

    def __iter__(self):
        return _NetworkIterator(self)

    @property
    def k(self):
        return np.array([len(i) for i in self.agents])

    def ids(self):
        for a in self.agents:
            yield a.id

    def check_connection(self, i1, i2) -> List[int]:
        """
        Locate an index of an element representing connection between two agents.
        Returns array of indexing specifying position in the link list.

        Development proposal: take list of indexes as arguments.

        :param i1: One of the agent of the connection
        :param i2: Second of the agent of the connection
        :return: Vector of indexes in *links* array where connection exists
        """
        where_vec = []
        for i, el in enumerate(np.isin(self.links, [i1, i2])):
            if el.all():
                where_vec.append(i)
        return where_vec

    def add_connection(self, i1, i2):
        self[i1].add_neighbour(i2)
        if type(i2) != list:
            self[i2].add_neighbour(i1)
            self.links.append([i1, i2])
        else:
            for i in i2:
                self[i].add_neighbour(i1)
                self.links.append([i1, i])

    def remove_connection(self, i1, i2):
        connections = self.check_connection(i1, i2)
        if len(connections):
            del self.links[connections[0]]
            self.agents[i1].neighbours = np.delete(self.agents[i1].neighbours,
                                                   np.where(self.agents[i1].neighbours == i2))
            self.agents[i2].neighbours = np.delete(self.agents[i2].neighbours,
                                                   np.where(self.agents[i2].neighbours == i1))

    @property
    def get_random_edge_ending(self):
        random_edge_end = np.random.choice(np.array(self.links).ravel())
        agent = self.agents[random_edge_end]
        return agent

    def print(self):
        for a in self.agents:
            a.print()


class _NetworkIterator:
    """Iterator for network class"""
    def __init__(self, net: Network):
        self._net = net
        self._index = 0

    def __next__(self):
        if self._index < len(self._net):
            to_return = self._net[self._index]
            self._index += 1
            return to_return

        raise StopIteration
