from abc import ABC, abstractmethod
from typing import List

import plotly.graph_objects as go
from matplotlib import cm
from src.congestion_mining.gibbs_sampler import *

class RootCause(ABC):
    def __init__(self, node, gs, target_distribution):

        self.node = node
        self.gs = gs
        self.td = target_distribution
        self.t = int(get_t(self.node))
        self.linkID = get_linkID(self.node)
        self.nodes = []

    @abstractmethod
    def get_probabilities(self, *args):
        pass

    @abstractmethod
    def plot(self, *args):
        pass

    def save(self, path, name):

        filename = path + name
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

class TemporalRootCause(RootCause):
    def __init__(self, node, gs, target_distribution):
        super().__init__(node, gs, target_distribution)

    def get_probabilities(self, link):

        """
        :return: list of probabilities showing how the probability of congestion for the link changes over time given
        the evidence
        """

        self.nodes = [n for n in self.gs.nodes if get_linkID(n) == link and get_t(n) < self.t]

        prob = []
        for n in tqdm(self.nodes):
            q = Query(q_var=[n], q_state=1, evidence=self.td.evidence, distribution=self.td)
            prob.append(q.y[-1])
        return prob

    def plot(self, prob):
        x = [get_t(n) for n in self.nodes]
        y = prob
        plt.figure(figsize=[15, 7])
        plt.plot(x, y, color='b')
        plt.axis(ymin=0, ymax=1)

class SpatialRootCause(RootCause):
    def __init__(self, node, gs, target_distribution, target_distribution2, network):

        """
        :param node: node name
        :param gs: gibbs sampler object
        :param target_distribution: joint distribution where evidence is congested
        :param target_distribution2: joint distribution where evidence is uncongested
        :param network: NetworkModel class object
        """
        super().__init__(node, gs, target_distribution)

        self.network = network
        self.td2 = target_distribution2

    def get_probabilities(self, t):

        self.t = t
        self.nodes = [n for n in self.gs.nodes if get_t(n) == t and get_linkID(n) != self.linkID]
        self.links = [get_linkID(n) for n in self.nodes]

        prob = []
        prob2 = []
        for n in tqdm(self.nodes):
            q1 = Query(q_var=[n], q_state=1, evidence=self.td.evidence, distribution=self.td)
            q2 = Query(q_var=[n], q_state=1, evidence=self.td2.evidence, distribution=self.td2)
            prob.append(q1.y[-1])
            prob2.append(q2.y[-1])

        return prob, prob2

    def plot(self, prob):

        colours = cm.get_cmap(name='RdYlGn_r')
        edge_list = self.links
        data = []

        edge_x = []
        edge_y = []
        for edge, p in zip(edge_list, prob):
            x, y = self.network.get_xy(edge)
            colour = 'rgba' + str(colours(p))
            edge_trace = go.Scatter(
                x=x, y=y,
                line=dict(width=2, color=colour),
                hoverinfo='text',
                text=edge,
                mode='lines')
            data.append(edge_trace)

        # Evidence Link highlighted in blue
        x, y = self.network.get_xy(self.linkID)
        trace = go.Scatter(
            x=x, y=y,
            line=dict(width=2, color='blue'),
            hoverinfo='none',
            mode='lines')
        data.append(trace)

        node_text = {}
        for edge in edge_list:
            x, y = self.network.get_xy(edge)
            fnode_x, tnode_x = x
            fnode_y, tnode_y = y

            node_text[(fnode_x, fnode_y)] = np.zeros(2)
            node_text[(tnode_x, tnode_y)] = np.zeros(2)

        for edge in edge_list:
            x, y = self.network.get_xy(edge)
            fnode_x, tnode_x = x
            fnode_y, tnode_y = y
            node_text[(fnode_x, fnode_y)][0] = edge
            node_text[(tnode_x, tnode_y)][1] = edge

        node_x = []
        node_y = []
        text = []
        for node, links in node_text.items():
            x, y = node
            node_x.append(x)
            node_y.append(y)
            text.append('fnode: ' + str(int(links[0])) + ', tnode: ' + str(int(links[1])))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=text,
            marker=dict(
                showscale=True,
                colorscale='RdYlGn',
                reversescale=True,
                color=[],
                size=5,
                colorbar=dict(
                    tickmode="array",
                    ticktext=[str(i / 10) for i in range(11)],
                    tickvals=[i / 10 for i in range(-5, 6)],
                    thickness=15,
                    title='Congestion Probability',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        data.append(node_trace)

        fig = go.Figure(data=data,
                        layout=go.Layout(
                            title='<br>Probabilities for edges at t= ' + str(self.t),
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()

class SpatialTemporalRootCause(RootCause):
    def __init__(self, node, gs, target_distribution, network):
        super().__init__(node, gs, target_distribution)

        self.linkID = get_linkID(node)
        self.network = network
        self.links = [x for x in self.network.links if x != get_linkID(node)]
        self.nodes = []
        length = len(self.links)
        self.prob = np.zeros((length, self.t))

    def get_probabilities(self):

        for i, link in enumerate(self.links):
            p = []
            self.nodes = [n for n in self.gs.nodes if get_t(n) < self.t and get_linkID(n) == link]
            for n in self.nodes:
                q = Query(q_var=[n], q_state=1, evidence=self.td.evidence, distribution=self.td)
                p.append(q.y[-1])
            self.prob[i] = p

        # self.nodes = [n for n in self.gs.nodes if get_t(n) < self.t and get_linkID(n) != self.linkID]
        #
        # for n in tqdm(self.nodes):
        #     q = Query(q_var=[n], q_state=1, evidence=self.td.evidence, distribution=self.td)
        #     self.prob.append(q.y[-1])

    def plot(self, links):
        x = [get_t(n) for n in self.nodes]
        plt.figure(figsize=[15, 7])
        for p, l in zip(self.prob, self.links):
            if l in links:
                plt.plot(x, p, label=l)
        plt.axis(ymin=0, ymax=1)
        plt.legend()

    def plot_network(self, edge_list):

        data = []

        for edge in self.links:
            x, y = self.network.get_xy(edge)
            edge_trace = go.Scatter(
                x=x, y=y,
                line=dict(width=2, color='black'),
                hoverinfo='text',
                text=edge,
                mode='lines')
            data.append(edge_trace)

        for edge in edge_list:
            x, y = self.network.get_xy(edge)
            edge_trace = go.Scatter(
                x=x, y=y,
                line=dict(width=2, color='red'),
                hoverinfo='text',
                text=edge,
                mode='lines')
            data.append(edge_trace)

        # Evidence Link highlighted in blue
        x, y = self.network.get_xy(self.linkID)
        trace = go.Scatter(
            x=x, y=y,
            line=dict(width=2, color='lightblue'),
            hoverinfo='none',
            mode='lines')
        data.append(trace)

        node_text = {}
        for edge in self.links:
            x, y = self.network.get_xy(edge)
            fnode_x, tnode_x = x
            fnode_y, tnode_y = y

            node_text[(fnode_x, fnode_y)] = np.zeros(2)
            node_text[(tnode_x, tnode_y)] = np.zeros(2)

        for edge in self.links:
            x, y = self.network.get_xy(edge)
            fnode_x, tnode_x = x
            fnode_y, tnode_y = y
            node_text[(fnode_x, fnode_y)][0] = edge
            node_text[(tnode_x, tnode_y)][1] = edge

        node_x = []
        node_y = []
        text = []
        for node, links in node_text.items():
            x, y = node
            node_x.append(x)
            node_y.append(y)
            text.append('fnode: ' + str(int(links[0])) + ', tnode: ' + str(int(links[1])))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=text,
            marker=dict(
                showscale=False,
                reversescale=False,
                color=[],
                size=5,
                line_width=2))
        data.append(node_trace)

        fig = go.Figure(data=data,
                        layout=go.Layout(
                            title='<br>Root Cause Links for link ' + self.linkID,
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()
