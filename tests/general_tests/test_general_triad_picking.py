import numpy as np

from experiments.antal_experiments.ltd_general import find_available_triads


class TestGeneralLtdFunctions:
    def __init__(self):
        """
        n_agents: 5
        Connections: 0-1, 1-0, 0-2, 2-0  1-2, 2-1, 1-3, 3-1, 1-4, 4-1, 2-3, 3-2, 2-4,
        Common connections:
            *0,1: 2
            *0,2: 1
            *1,2: 0, 3, 4
            *1,3: 2
        """
        self.adm = np.zeros((5, 5))
        self.adm[0, 1] = 1
        self.adm[1, 0] = 1
        self.adm[0, 2] = 1
        self.adm[2, 0] = 1
        self.adm[1, 2] = 1
        self.adm[2, 1] = 1
        self.adm[1, 3] = 1
        self.adm[1, 4] = 1
        self.adm[4, 1] = 1
        self.adm[3, 2] = 1
        self.adm[2, 4] = 1
        self.triads = find_available_triads(True, self.adm)

    def __call__(self):
        print("ADM:")
        print(self.adm)
        print("\n available triads:")
        for triad in self.triads:
            print(triad)
