from pathlib import Path

import itertools
import numpy as np
import logging
import matplotlib.pyplot as plt
import collections

from wolf.utils.miscelanous import pretty_format_list
from wolf.utils.os import makedirs

logger = logging.getLogger(__name__)

EPSILON = 1e-6    # A very small number (not related to the exploration parameter)

def to_onehot(vector, max_value):
    rez = [0] * (max_value + 1)
    for i in range(len(vector)):
        index = vector[i] if vector[i] < max_value else max_value
        index = i * (max_value + 1) + index
        rez[int(index)] = 1.0
    return rez

def epsilon_decay(start=1.0, decay=0.01, N=100, savepath=None):
    makedirs(savepath)
    if decay == 0:
        decays = np.full(N, start)
    elif decay > 0:
        decays = np.exp(-np.arange(N) / (1.0 / decay)) * start
    else:
        raise Exception("Decay must be positive")
    str_decay = pretty_format_list(decays)
    logger.info("Epsilons (decayed) : [{}]".format(str_decay))
    if logger.getEffectiveLevel() <= logging.DEBUG:
        plt.plot(range(len(decays)), decays)
        plt.title("epsilon decays")
        plt.show()
        if savepath is not None:
            plt.savefig(Path(savepath) / "epsilon_decay")
        plt.close()
    return decays


def normalized(a):
    sum = 0.0
    for v in a:
        sum += v
    rez = np.array([v / sum for v in a])
    return rez


def update_lims(lims, values):
    return min(lims[0], np.amin(values)), max(lims[1], np.amax(values))


# TODO check if ok
def create_arrangements(nb_elements, size_arr, current_size_arr=0, arrs=None):
    new_arrs = []
    if not arrs:
        arrs = [[]]
    for arr in arrs:
        for i in range(0, nb_elements):
            new_arr = list(arr)
            new_arr.append(i)
            new_arrs.append(new_arr)
    if current_size_arr >= size_arr - 1:
        return new_arrs
    else:
        return create_arrangements(
            nb_elements=nb_elements,
            size_arr=size_arr,
            current_size_arr=current_size_arr + 1,
            arrs=new_arrs,
        )


def near_split(x, num_bins=None, size_bins=None):
    """
        Split a number into several bins with near-even distribution.

        You can either set the number of bins, or their size.
        The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def zip_with_singletons(*args):
    """
        Zip lists and singletons by repeating singletons

        Behaves usually for lists and repeat other arguments (including other iterables such as tuples np.array!)
    :param args: arguments to zip x1, x2, .. xn
    :return: zipped tuples (x11, x21, ..., xn1), ... (x1m, x2m, ..., xnm)
    """
    return zip(*(arg if isinstance(arg, list) else itertools.repeat(arg) for arg in args))


def recursive_dict_update(d, u):
    for k, v in u.items():
        if k == '*':
            recursive_dict_update(d, {k: v for k in d.keys()})
            continue
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def sim2sec_fl(sim, sim_step):
    return sim * sim_step

def sec2sim_fl(sec, sim_step):
    return sec / sim_step

def sim2sec(sim, sim_step):
    return int(sim2sec_fl(sim, sim_step))

def sec2sim(sec, sim_step):
    return int(sec2sim_fl(sec, sim_step))

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    epsilon_decay(1.0, 0.001, 5000, savepath="tmp")
    epsilon_decay(1.0, 0.005, 1000, savepath="tmp")
    epsilon_decay(1.0, 0.0005, 10000, savepath="tmp")
