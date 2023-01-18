import plotly.graph_objects as go
from matplotlib import cm, colors
from src.congestion_mining.gibbs_sampler import *

def temporal_plot(self, links):
    x = [get_t(n) for n in self.nodes]
    plt.figure(figsize=[15, 7])
    for p, l in zip(self.prob, self.links):
        if l in links:
            plt.plot(x, p, label=l)
    plt.axis(ymin=0, ymax=1)
    plt.xlabel('Time Intervals')
    plt.ylabel('Probability of congestion')
    plt.legend()
    plt.title('Root Cause Links for link ' + self.linkID)

def spatial_plot(self, prob):

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

def relative_plot(self, prob):

    colours = cm.get_cmap(name='BrBG')
    norm = colors.Normalize(vmin=min(prob), vmax=max(prob))
    edge_list = self.links
    data = []

    edge_x = []
    edge_y = []
    for edge, p in zip(edge_list, prob):
        x, y = self.network.get_xy(edge)
        colour = 'rgba' + str(colours(norm(p)))
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
        line=dict(width=2, color='red'),
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
            colorscale='BrBG',
            reversescale=False,
            color=[],
            size=5,
            colorbar=dict(
                tickmode="array",
                ticktext=[str(i / 100) for i in range(-10, 11, 2)],
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
