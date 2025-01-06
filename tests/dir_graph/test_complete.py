"""Test complete network generator"""
from networktools.agent import Agent
from networktools.link import Link
from networktools.network_generators.complete import CompleteNetwork


class TAgent(Agent):
    def __init__(self, i, is_directed=False):
        super().__init__(i, is_directed)
        self.sub_id = i * 10


class TestComplete:
    def __init__(self):
        self.agents_directed, self.links_directed = CompleteNetwork(10, True, Link, TAgent)()
        self.agents_undirected, self.links_undirected = CompleteNetwork(10, False, Link, TAgent)()

    def print_connections(self):
        print('Directed complete network info:')
        for agent in self.agents_directed:
            print(f'Sub ID: {agent.sub_id};\tK_in: {agent.in_k}, K_out: {agent.out_k}')
        print()
        print('Undirected complete network info:')
        for agent in self.agents_undirected:
            print(f'Sub ID: {agent.sub_id};\tK: {agent.k}')
