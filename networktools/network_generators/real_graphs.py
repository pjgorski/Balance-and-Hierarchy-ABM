"""Real networks from files"""
import numpy as np

from pathlib import Path
import numpy as np
import pandas as pd

from networktools.network_generators.decorator import register_generator


class RealNetwork:
    def __init__(self, filename: str, datatype:str = "triads"):
        self.filename = filename
        self.datatype = datatype
        
        if datatype == "triads":
            self.read_data_from_triads()
        
        pass

    def read_data_from_triads(self):
        path = Path(self.filename)
        if self.filename.endswith(".csv"):
            trs = pd.read_csv(path)
        else:
            trs = pd.read_hdf(path)
        
        all_nodes = np.unique([*trs.a, *trs.b, *trs.c])
        a_agents = np.unique(trs.a) # nodes that can be focal nodes
        self.n_agents = len(all_nodes)
        
        orig_agents_to_new_inds_dict = {key: val 
                                        for key, val in 
                                        zip(all_nodes, 
                                            list(range(0,self.n_agents)))}
        self.focal_agents = np.array([orig_agents_to_new_inds_dict[ag] 
                                      for ag in a_agents])
        
        if self.n_agents <= 1000:
            self.ini_adm = np.zeros((self.n_agents, self.n_agents), dtype = np.int8)
        else: 
            """Too big!"""
            self.ini_elist = {}
            #edge list!
        
        self.all_triads = []
        self.triads_of_agents = [[] for _ in range(self.n_agents)]
        # each focal agent has some agents they can meet to discuss another agent:
        self.b_agents_of_focal_agents = [[] for _ in range(self.n_agents)] 
        self.c_agents_of_ab_pairs = {}
        for ind, row in trs.iterrows():
            ags = [row.a, row.b, row.c]
            ags = [orig_agents_to_new_inds_dict[ag] for ag in ags]
            links = [[ags[0], ags[1]], [ags[0], ags[2]], [ags[1], ags[2]] ]
            pols = [row.ab, row.ac, row.bc]
            for link, pol in zip(links, pols):
                if self.n_agents <= 1000:
                    self.ini_adm[link[0], link[1]] = pol
                else:
                    self.ini_elist[(link[0], link[1])] = pol
            self.b_agents_of_focal_agents[ags[0]].append(ags[1])
            self.triads_of_agents[ags[0]].append(ags)
            self.all_triads.append(ags)
            if (ags[0], ags[1]) in self.c_agents_of_ab_pairs:
                self.c_agents_of_ab_pairs[(ags[0], ags[1])].append(ags[2])
            else:
                self.c_agents_of_ab_pairs[(ags[0], ags[1])] = [ags[2]]
        
        if self.n_agents <= 1000:
            self.adm = np.copy(self.ini_adm)
        else:
            self.elist = {key: value for key, value in self.ini_elist.items()}
        
        # removing repeated entries
        for ind, ag in enumerate(self.b_agents_of_focal_agents):
            self.b_agents_of_focal_agents[ind] = np.unique(ag)

    def get_adjacency_matrix(self):
        if self.n_agents <= 1000:
            return self.adm
        else:
            raise ValueError("Too large network. Adjancency matrix not created.")

    def get_edges_data(self):
        if self.n_agents <= 1000:
            return self.adm
        else:
            return self.elist
    
    def refresh(self):
        if self.n_agents <= 1000:
            self.adm = np.copy(self.ini_adm)
        else:
            self.elist = {key: value for key, value in self.ini_elist.items()}

    @classmethod
    def name(cls) -> str:
        return cls.__name__


@register_generator
class WikiElections(RealNetwork):
    def __init__(self,filename = "/home/pgorski/Desktop/data/wikielections/wikielections_triads.h5"):
        super().__init__(filename=filename)


@register_generator
class Slashdot(RealNetwork):
    def __init__(self,filename = "/home/pgorski/Desktop/data/slashdot/slashdot_triads.h5"):
        super().__init__(filename=filename)

@register_generator
class GeneralRealNetwork(RealNetwork):
    def __init__(self,filename):
        super().__init__(filename=filename)

