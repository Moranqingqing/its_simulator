from src.congestion_mining.network import NetworkModel
import networkx as nx
import geopandas as gpd
import pickle
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

class ShapefileNetworkModel(NetworkModel):

    """
    Parameters
    ----------
    shapefile : .shp file
        The .shp file where network geometry information is stored

    Attributes
    ----------
    G
    nodes
    edges
    """

    def __init__(self, path, shapefile, data_file, build='auto', max_length=500):
        super().__init__()

        self.path = path
        self.shapefile = shapefile
        self.build = build
        self.df = gpd.read_file(path+shapefile)

        with open(path + data_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        self.links = content.links

        if self.build == 'auto':
            self.G = nx.read_shp(self.path + self.shapefile)

        elif self.build == 'manual':
            self.G = nx.Graph()
            for row in self.df.itertuples():
                geometry = list(row.geometry.coords)
                x0, y0 = geometry[0]
                x1, y1 = geometry[-1]

                if (row.tnode, row.fnode) in self.G.edges:
                    self.G.add_node(row.fnode + '_1', pos=[x0, y0])
                    self.G.add_node(row.tnode + '_1', pos=[x1, y1])
                    self.G.add_edge(row.fnode + '_1', row.tnode + '_1', linkID=row.linkID, speed=row.speed,
                                    length=row.length, x=[x0, x1], y=[y0, y1])
                else:
                    self.G.add_node(row.fnode, pos=[x0, y0])
                    self.G.add_node(row.tnode, pos=[x1, y1])
                    self.G.add_edge(row.fnode, row.tnode, linkID=row.linkID, speed=row.speed, length=row.length,
                                    x=[x0, x1], y=[y0, y1])

        self.nodes = self.G.nodes
        self.edges = self.G.edges

        for fnode, tnode, data in self.edges.data():
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

    def plot_network(self, edge_list=None, highlight=None):

        data = []

        edge_x = []
        edge_y = []
        if edge_list:
            for edge in edge_list:
                x, y = self.get_xy(edge)
                x0, x1 = x
                y0, y1 = y
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
        else:
            for fnode, tnode, attr in self.edges.data():
                if self.build == 'auto':
                    x0, y0 = fnode
                    x1, y1 = tnode
                elif self.build == 'manual':
                    x0, x1 = attr['x']
                    y0, y1 = attr['y']
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
        data.append(edge_trace)

        if highlight:
            for link in highlight:
                x, y = self.get_xy(link)
                trace = go.Scatter(
                    x=x, y=y,
                    line=dict(width=1, color='red'),
                    hoverinfo='none',
                    mode='lines')
                data.append(trace)

        node_x = []
        node_y = []
        if edge_list:
            for edge in edge_list:
                x, y = self.get_xy(edge)
                node_x.extend(x)
                node_y.extend(y)
        else:
            for node, attr in self.nodes.data():
                if self.build == 'auto':
                    x, y = node
                elif self.build == 'manual':
                    x, y = attr['pos']
                node_x.append(x)
                node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=5,
                line_width=2))
        data.append(node_trace)

        fig = go.Figure(data=data,
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

    def get_xy(self, edge):

        for fnode, tnode, data in self.edges.data():
            if data['linkID'] == edge:
                if self.build == 'auto':
                    x0, y0 = fnode
                    x1, y1 = tnode
                    x = [x0, x1]
                    y = [y0, y1]
                elif self.build == 'manual':
                    x = data['x']
                    y = data['y']

        return x, y

    def node2linkID(self, node):
        for fnode, tnode, data in self.edges.data():
            if fnode == node:
                from_edge = data['linkID']
            elif tnode == node:
                to_edge = data['linkID']

        return from_edge, to_edge





