import numpy as np
import pickle
import random
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
import pandas as pd

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

class hashabledict(dict):
    """
    Hashable dictionaries allow a dictionary to be a key in a nested dictionary
    """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class TargetDistribution:
    def __init__(self, samples, variables, evidence, T, thinning, burn_in, chains):
        self.samples = samples
        self.variables = variables
        self.evidence = evidence
        self.T = T
        self.thinning = thinning
        self.burn_in = burn_in
        self.chains = chains

    def save_file(self, path, name):
        filename = path + name
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class GibbsSampling:
    def __init__(self, model, nodes=None):
        self.model = model
        if nodes:
            self.nodes = nodes
        else:
            self.nodes = model.nodes
        self.cpds = model.cpds
        self.rf = self.relevant_factors()
        self.chains = 1

    def run_chains(self, T, evidence, thinning, burn_in, chains):
        self.chains = chains

        N = T * thinning + burn_in
        variables = self.nodes.copy()
        K = len(variables)

        samples = np.zeros((chains, N, K))
        for i in range(self.chains):
            samples[i, :, :] = self.MCMC(T, evidence, thinning, burn_in)

        return TargetDistribution(samples, variables, evidence, T, thinning, burn_in, self.chains)

    def relevant_factors(self):
        factors = defaultdict(list)
        for cpd in self.cpds:
            for var in cpd.scope:
                factors[var].append(cpd)

        # dictionary of the factors for each variable
        relevant_factors = defaultdict(list)
        for var in factors:
            relevant_factors[var] = factors[var]

        return relevant_factors

    def MCMC(self, T, evidence, thinning, burn_in):

        N = T * thinning + burn_in
        variables = self.nodes.copy()
        K = len(variables)
        samples = np.zeros((N, K))
        state = np.zeros(K)

        e_indices = []
        if evidence:
            for var, val in evidence.items():
                i = variables.index(var)
                e_indices.append(i)
                state[i] = val

        for n in tqdm(range(1, N)):
            for x in range(len(variables)):
                if x in e_indices:
                    pass
                else:
                    var = variables[x]
                    prob_x0 = 1.0
                    prob_x1 = 1.0
                    for cpd in self.rf[var]:
                        state[x] = 0
                        search = {}
                        for v in cpd.scope:
                            ind = variables.index(v)
                            search[v] = state[ind]
                        prob_x0 *= cpd.distribution(search)
                        search[var] = 1
                        prob_x1 *= cpd.distribution(search)
                    p = prob_x0 / (prob_x0 + prob_x1)
                    s = np.random.choice([0, 1], p=[p, 1 - p])
                    state[x] = s
            samples[n, :] = state

        if self.chains == 1:
            return TargetDistribution(samples, variables, evidence, T, thinning, burn_in, self.chains)
        else:
            return samples

class Query:
    def __init__(self, q_var, q_state, evidence, distribution):

        self.q_var = q_var
        self.q_state = q_state
        self.evidence = evidence
        self.variables = distribution.variables
        self.burn_in = distribution.burn_in
        self.thinning = distribution.thinning
        self.T = distribution.T
        # only running one chain
        self.chains = 1

        assert self.evidence == distribution.evidence
        self.samples = distribution.samples

        self.x = np.array([i for i in range(self.burn_in, self.T, self.thinning)])

        if self.chains == 1:
            self.y = np.array(self.run_query(self.samples))
        else:
            y_t = []
            for i in range(self.N):
                y = self.run_query(self.samples[i])
                y_t.append(y)
            y_t = np.array(y_t).T

            self.y_ave = []
            self.error = []
            for y in y_t:
                mean, err = self.mean_confidence_interval(y)
                self.y_ave.append(mean)
                self.error.append(err)
            self.y_ave = np.array(self.y_ave)

    def run_query(self, sample):

        var = self.q_var[0]
        ind = self.variables.index(var)

        y = []
        for i in range(self.burn_in+1, len(sample)+1, self.thinning):
            count0 = 0
            count1 = 0
            for s in sample[:i, ind]:
                if s == 0:
                    count0 += 1
                else:
                    count1 += 1

            total = count0 + count1

            if self.q_state == 0:
                p_y = count0 / total
            else:
                p_y = count1 / total
            y.append(p_y)

        return y

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n - 1)

        return m, h

    @staticmethod
    def plot(x, y, actual=None, error=None):

        plt.figure(figsize=[15, 7])
        plt.plot(x, y, color='b')
        plt.axis(xmin=-50, xmax=2550, ymin=0, ymax=1)
        plt.minorticks_on()
        plt.grid(True, which='both', alpha=0.5, ls='--')
        legend = ['MCMC']

        if actual:
            plt.axhline(y=actual, color='r', linestyle='--')
            legend.append('MLE value')

        plt.xlabel('Number of Samples')
        plt.ylabel('Query Probability Value')
        plt.legend(legend)

        if error:
            plt.errorbar(x, y, yerr=error)
