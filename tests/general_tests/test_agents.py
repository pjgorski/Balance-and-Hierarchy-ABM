# link.py>
"""Test agents outside network class"""

from networktools.agent import Agent
from networktools.link import Link


class TestAgents:
    def __init__(self):
        self.ndagents = [Agent(i) for i in range(3)]
        self.ndlinks = [Link(self.ndagents[0], self.ndagents[1]),
                        Link(self.ndagents[1], self.ndagents[2]),
                        Link(self.ndagents[2], self.ndagents[0])]
        self.ndagents[0].add_links(self.ndlinks[::2])
        self.ndagents[1].add_links(self.ndlinks[:2])
        self.ndagents[2].add_links(self.ndlinks[1:])

    def __call__(self, *args, **kwargs):
        self.test_k()
        self.test_add_wrong_class_link()
        self.test_neighbours()

    def test_k(self):
        k_arr = [a.k for a in self.ndagents]
        print(f"Degrees: {k_arr}\nlengths: {[len(a) for a in self.ndagents]}")

    def test_add_wrong_class_link(self):
        link_wrong_class = Agent(100)
        try:
            self.ndagents[0].add_link(link_wrong_class)
            print("Oooops, error did not occur")
        except AssertionError as ae:
            print(f'Properly raised error: {ae}')
        try:
            self.ndagents[0].add_link(self.ndlinks[1])
            print('Error should be raised, but wasn\'t')
        except ValueError as ve:
            print(f'Properly raised error: {ve}')

    def test_neighbours(self):
        for agent in self.ndagents:
            neighbours = []
            for neighbour in agent:
                neighbours.append(neighbour.id)
            print(f'Agent: {agent.id}, neighbours: {neighbours}')
