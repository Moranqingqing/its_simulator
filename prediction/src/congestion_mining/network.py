from abc import ABC, abstractmethod
import pickle
import networkx as nx
import plotly.graph_objects as go

class NetworkModel(ABC):

    def __init__(self):

        self.G = None
        self.nodes = None
        self.edges = None
        self.LinkIDs = []
        self.lengths = []
        self.spd_lims = {}
        self.parents = {}
        self.children = {}

    @abstractmethod
    def get_parents(self, edge, max_length):
        pass

    def neighbourhood(self, node, max_length):

        """
        :param node: nodeID to find closest neighbours
        :param max_length:
        :return: list of neighbouring nodes for the specified node
        """

        # path_steps = nx.single_source_shortest_path_length(self.G, node)
        paths = nx.single_source_shortest_path(self.G, node)

        path_lengths = {}
        for k, v in paths.items():
            length = 0
            for i in range(len(v) - 1):
                # find length of edge from one node to the next node in the path
                length += self.G[v[i]][v[i+1]]['length']
            path_lengths[k] = length

        neighbours = [node for node, length in path_lengths.items() if 0 < length < max_length]

        return neighbours

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
