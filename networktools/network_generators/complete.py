"""Complete networks directed and undirected"""
import numpy as np

from networktools.network_generators.decorator import register_generator
from networktools.link import Link
from networktools.agent import Agent


@register_generator
class CompleteNetwork:
    def __init__(self,
                 n_agents: int,
                 is_directed: bool = False,
                 link_class=Link,
                 agent_class=Agent):
        """Undirected complete network generator"""
        assert issubclass(link_class, Link), "Requires class that is subclass of Link"
        assert issubclass(agent_class, Agent), "Requires class that is subclass of Agent"

        self.agents = [agent_class(i, is_directed) for i in range(n_agents)]
        self.links = []
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                self.links.append(link_class(Agent(i), Agent(j)))
                self.agents[i].add_link(self.links[-1])
                self.agents[j].add_link(self.links[-1])
                if is_directed:
                    self.links.append(link_class(Agent(j), Agent(i)))
                    self.agents[i].add_link(self.links[-1])
                    self.agents[j].add_link(self.links[-1])

    def __call__(self, *args, **kwargs):
        return self.agents, self.links

    def get_adjacency_matrix(self):
        n = len(self.agents)
        adm = np.ones((n, n))
        np.fill_diagonal(adm, 0)
        return adm

    @classmethod
    def name(cls) -> str:
        return cls.__name__
