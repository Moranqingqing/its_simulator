import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, IndexLocator
from matplotlib.axes import Axes


def get_heatmap(data, links, start, end):

    """
    :param data: TrafficData object
    :param links: list of links to show. None will show all links
    :param start: start time
    :param end: end time
    :return: a heat map plot
    """

    if links:
        ind = [data.links.index(link) for link in links]
        Z = data.rel_speed[ind, start:end+1]
    else:
        links = data.links
        Z = data.rel_speed[:, start:end+1]

    num_links, t = Z.shape
    # x = np.arange(start, t*2 + 1, 2)  # len = 7
    x = np.arange(start, end+1, 2)

    fig, ax = plt.subplots(figsize=[17, len(links)*0.2+5])
    ax.pcolormesh(Z, cmap='RdYlGn')
    ax.xaxis.set_major_locator(IndexLocator(base=2, offset=0.5))
    ax.yaxis.set_major_locator(IndexLocator(base=1, offset=0.5))

    ax.set_xticklabels(labels=x)
    ax.set_yticklabels(labels=links)
    plt.xlabel('Time Intervals')
    plt.ylabel('Link IDs')
