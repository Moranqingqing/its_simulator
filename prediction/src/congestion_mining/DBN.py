from abc import ABC
import pandas as pd
import pickle
from pgmpy.models import BayesianModel
from src.congestion_mining.LRE import *


def get_att(node):
    i = node.rfind('_')
    linkID = node[:i]
    t = node[i + 1:]

    return linkID, t

def get_linkID(node):
    i = node.rfind('_')
    linkID = node[:i]

    return linkID

def get_t(node):
    i = node.rfind('_')
    t = int(node[i + 1:])

    return t

class DynamicBayesianNetwork(ABC):
    """
    Base class for all DBNs

    Parameters
    ----------
    network : NetworkModel
        Network of nodes and edges
    traffic_data : TrafficData
        Traffic data to load

    Attributes
    ----------

    """

    def __init__(self, network, traffic_data, look_back=12, max_parents=1, step=1):

        self.network = network
        self.traffic_data = traffic_data
        self.look_back = look_back
        self.max_parents = max_parents
        self.parents = {}
        self.step = step
        self.dbnet = BayesianModel()
        self.nodes = None
        self.cpds = None

        # Artificially limit number of parents
        if max_parents:
            for k, v in network.parents.items():
                self.parents[k] = v[:max_parents]
        else:
            self.parents = network.parents

        self.ebunch = []
        for node, parents in self.parents.items():
            if node in self.traffic_data.links:
                for i in range(0, self.look_back+1, self.step):
                    if i == 0:
                        self.dbnet.add_edge(node + '_' + str(i), node + '_' + str(i + self.step))
                    else:
                        if i+self.step <= self.look_back:
                            self.ebunch.append((node + '_' + str(i), node + '_' + str(i + self.step)))
                for parent in parents:
                    for i in range(0, self.look_back+1, self.step):
                        if i == 0:
                            self.dbnet.add_edge(parent + '_' + str(i), node + '_' + str(i + self.step))
                        else:
                            if i+step <= self.look_back:
                                self.ebunch.append((parent + '_' + str(i), node + '_' + str(i + self.step)))

        # Estimate cpds from data
        # Assuming cpd from time t to t+step is the same for all t
        self.data = self.create_df()

    def run_MLE(self):

        self.dbnet.fit(data=self.data, estimator=DynamicMaximumLikelihoodEstimator)

        # additional edges for time steps
        self.dbnet.add_edges_from(self.ebunch)
        self.nodes = self.dbnet.nodes

        # nodes are stationary so cpd for a node at time t is equal to the cpd at the previous time step, for all t>0
        cpd_lookback = []
        for cpd in self.dbnet.get_cpds():
            if cpd.t == '1':
                for i in range(2, self.look_back + 1):
                    variable = cpd.linkID + '_' + str(i)
                    cpd_t_i = DynamicMaximumLikelihoodEstimator.estimate_cpd_t(self.dbnet, variable, cpd)
                    cpd_lookback.append(cpd_t_i)
        for cpd in cpd_lookback:
            self.dbnet.add_cpds(cpd)
        self.cpds = self.dbnet.get_cpds()

        assert self.dbnet.check_model()

    def run_logistic(self):

        estimator = LogisticRegressionEstimator(self.dbnet, self.data)
        self.cpds = estimator.get_parameters()

        # additional edges for intermediary time steps
        if self.ebunch:
            self.dbnet.add_edges_from(self.ebunch)
        self.nodes = list(self.dbnet.nodes)

        self.cpd_lookback = []
        for cpd in self.cpds:
            if cpd.t == str(self.step):
                linkID = cpd.linkID
                if cpd.coefficients is not None:
                    for i in range(2*self.step, self.look_back + 1, self.step):
                        node = linkID + '_' + str(i)
                        parents = self.dbnet.get_parents(node)
                        c = {}
                        for p in parents:
                            for k, v in cpd.coefficients.items():
                                if get_linkID(k) == get_linkID(p):
                                    c[p] = v
                        cpd_t_i = LogisticCPD(node=node, parents=parents, intercept=cpd.intercept, coefficients=c)
                        self.cpd_lookback.append(cpd_t_i)
                else:
                    for i in range(2*self.step, self.look_back + 1, self.step):
                        node = linkID + '_' + str(i)
                        parents = self.dbnet.get_parents(node)
                        cpd_t_i = LogisticCPD(node=node, parents=parents, probs=cpd.probs)
                        self.cpd_lookback.append(cpd_t_i)

        self.cpds += self.cpd_lookback

    def get_cpd(self, node):

        i = [cpd.variable for cpd in self.cpds].index(node)
        return self.cpds[i]

    def create_df(self):

        df = {}
        for n in self.dbnet.nodes():
            linkID, t = get_att(n)
            try:
                if t == '0':
                    ind = self.traffic_data.links.index(linkID)
                    data = self.traffic_data.congestion[ind]
                    df[n] = data
            except ValueError:
                pass

        df = pd.DataFrame(df)
        transformed_set = df.copy()

        # Add shifted columns
        for column in df.columns:
            linkID, _ = get_att(column)
            transformed_set[linkID + '_' + str(self.step)] = transformed_set[column].shift(self.step)

        # Remove top rows that contain empty cells due to shifting
        return transformed_set[self.look_back:]

    def save(self, path, name):
        filename = path + name
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
