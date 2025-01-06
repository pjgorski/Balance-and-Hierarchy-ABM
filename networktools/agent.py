# agent.py>
"""
Agents are the network nodes in directed and undirected graph. Iterating over an agent instance
will be an iterating over its' neighbours.
"""

from itertools import compress
from typing import List

from networktools.link import Link


class Agent:
    def __init__(self, id, is_directed=False):
        self.id = id
        self.links = []
        self.is_directed = is_directed
        self.link_is_in = []

    def __getitem__(self, key):
        if not self.is_directed:
            link = self.links[key]
        else:
            link = list(compress(self.links, [not l for l in self.link_is_in]))[key]
        if link.ids[0] == self.id:
            return link.own[1]
        else:
            return link.own[0]

    def __len__(self):
        if not self.is_directed:
            return len(self.links)
        else:
            return len(list(compress(self.links, [not l for l in self.link_is_in])))

    def __iter__(self):
        return _AgentIterator(self)

    def add_link(self, link: Link):
        assert issubclass(link.__class__, Link), "Trying to add as a link something which is not a link"
        if all([in_id != self.id for in_id in link.ids]):
            raise ValueError('Trying to add link to agent which is not related to this link')
        self.links.append(link)
        self.link_is_in.append(link.own[1].id == self.id)

    def add_links(self, links: List[Link]):
        for link in links:
            self.add_link(link)

    @property
    def neighbours_ids(self):
        neighbours_ids = []
        for neighbour in self:
            neighbours_ids.append(neighbour.id)
        return neighbours_ids

    @property
    def k(self):
        return len(self)

    @property
    def in_k(self):
        assert self.is_directed, 'This function is to be used for directed networks'
        return len(list(compress(self.links, self.link_is_in)))

    @property
    def out_k(self):
        assert self.is_directed, 'This function is to be used for directed networks'
        return self.k


class _AgentIterator:
    """General in-agent iterator handler"""
    def __init__(self, agent: Agent):
        self._agent = agent
        self._index = 0

        if not self._agent.is_directed:
            self.size = len(self._agent.links)
        else:
            self.size = len(list(compress(self._agent.links, [not l for l in self._agent.link_is_in])))

    def __next__(self):
        if self._index < self.size:
            to_return = self._agent[self._index]
            self._index += 1
            return to_return
        raise StopIteration
