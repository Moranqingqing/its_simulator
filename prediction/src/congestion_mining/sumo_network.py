import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from src.congestion_mining.network import NetworkModel
import networkx as nx

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
import plotly.graph_objects as go
init_notebook_mode(connected=True)

class SUMONetwork(NetworkModel):

    """
    Parameters
    ----------
    path : file path
        The location of the parent path for all sumo xml and configuration files e.g. 'data/Sumo/huawei/'

    net_file : .xml file
        Defines the road network, e.g. 'Simple3intersection.net.xml'

    Attributes
    ----------

    """

    def __init__(self, path, net_file, max_length=200):
        super().__init__()

        self.path = path
        self.net_file = net_file

        self.sumo_net = sumolib.net.readNet(self.path + self.net_file)
        self.nodes = self.sumo_net.getNodes()
        self.nodeIDs = [node.getID() for node in self.nodes]
        self.edges = self.sumo_net.getEdges()
        self.links = [edge.getID() for edge in self.edges]
        self.lengths = [edge.getLength() for edge in self.edges]

        self.G = nx.DiGraph()

        self.lanes = []
        self.fringe = []
        for edge, ID, length in zip(self.edges, self.links, self.lengths):

            if edge.is_fringe():
                self.fringe.append(edge)

            # self.spd_lims.append(edge.getSpeed())
            self.spd_lims[ID] = edge.getSpeed()

            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            self.G.add_edge(from_node, to_node, sumo_edge=edge, linkID=ID, length=length, spd_lim=edge.getSpeed())

            for lane in edge.getLanes():
                self.lanes.append(lane)

        self.parent_ids = {}
        for fnode, tnode, data in self.G.edges.data():
            linkID = data['linkID']
            if linkID in self.links:
                parents = self.get_parents(fnode, max_length)
                if linkID in parents:
                    parents.remove(linkID)
                self.parents[linkID] = parents

    def get_parents(self, fnode, max_length):

        paths = nx.single_target_shortest_path(self.G, fnode)

        path_edges = {}
        for source, steps in paths.items():
            tot_length = 0
            edges = []
            for i in range(len(steps) - 1):
                # find length of edge from one node to the next node in the path
                tot_length += self.G[steps[i]][steps[i + 1]]['length']
                edges.append(self.G[steps[i]][steps[i + 1]]['linkID'])
            path_edges[tot_length] = edges

        parents = []
        for edge in [edge for edges in [edges for length, edges in path_edges.items() if 0 < length < max_length]
                     for edge in edges]:
            if (edge not in parents) and (edge in self.links):
                parents.append(edge)

        return parents

    def _get_id(self, edge):

        return self.G.edges[edge]['linkID']

    def plot_network(self, edge_list=None, node_list=None):

        edge_x = []
        edge_y = []

        if not edge_list:
            edge_list = self.edges

        if not node_list:
            node_list = self.nodes

        for edge in edge_list:
            x0, y0 = edge.getFromNode().getCoord()
            x1, y1 = edge.getToNode().getCoord()
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in node_list:
            x, y = node.getCoord()
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Outgoing',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_outgoing = []
        node_text = []
        for node in node_list:

            nodeID = node.getID()
            outgoing = []
            for edge in node.getOutgoing():
                outgoing.append(edge)
            node_text.append("NodeID: %s<br>Outgoing: %s<br>" % (nodeID, outgoing))

            node_outgoing.append(len(node.getOutgoing()))

        node_trace.marker.color = node_outgoing
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Network graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()

