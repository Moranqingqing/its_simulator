from itertools import chain, product
from tabulate import tabulate
import numpy as np
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from sklearn.linear_model import LogisticRegression
from pgmpy.factors.discrete import TabularCPD
from copy import deepcopy


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
    t = node[i + 1:]

    return t

class hashabledict(dict):
    """
  Hashable dictionaries allow a dictionary to be a key in a nested dictionary
  """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class CPD(TabularCPD):
    def __init__(self, variable, variable_card, values, evidence=None, evidence_card=None, state_names={}):
        super().__init__(variable, variable_card, values, evidence, evidence_card, state_names)

        self.evidence = evidence
        self.evidence_card = evidence_card
        self.state_names = state_names
        self.state_counts = values
        self.linkID, self.t = get_att(self.variable)


class DynamicMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def __init__(self, model, data, **kwargs):
        super(DynamicMaximumLikelihoodEstimator, self).__init__(model, data, **kwargs)

    def estimate_cpd(self, node):
        state_counts = self.state_counts(node)

        # if a column contains only `0`s (no states observed for some configuration
        # of parents' states) fill that column uniformly instead
        state_counts.loc[:, (state_counts == 0).all()] = 1

        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
        node_cardinality = len(self.state_names[node])

        # Get the state names for the CPD
        state_names = {node: list(state_counts.index)}
        if parents:
            state_names.update(
                {
                    state_counts.columns.names[i]: list(state_counts.columns.levels[i])
                    for i in range(len(parents))
                }
            )

        cpd = CPD(
            node,
            node_cardinality,
            np.array(state_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: self.state_names[var] for var in chain([node], parents)}
        )
        cpd.normalize()
        return cpd

    @staticmethod
    def estimate_cpd_t(model, node, cpd):
        evidence = model.get_parents(node)

        cpd_t = CPD(
            node,
            cpd.variable_card,
            cpd.state_counts,
            evidence,
            cpd.evidence_card
        )
        cpd_t.normalize()

        return cpd_t

class LogisticRegressionEstimator(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        super(LogisticRegressionEstimator, self).__init__(model, data, **kwargs)

    def get_parameters(self):
        parameters = []

        for node in sorted(self.model.nodes()):
            cpd = self.estimate_cpd(node)
            parameters.append(cpd)

        return parameters

    def estimate_cpd(self, node):
        parents = sorted(self.model.get_parents(node))

        # variable has to have at least 1 parent and more than one class label in the data
        if parents and len(self.data[node].unique()) > 1:
            X = self.data.filter(parents)
            y = self.data[node]
            clf = LogisticRegression()
            clf.fit(X, y)

            evidence_var = parents
            assert len(evidence_var) == len(clf.coef_.flatten())

            # get coefficients and intercept to define the sigmoid function
            coefficients = {}
            for var, c in zip(evidence_var, clf.coef_.flatten()):
                coefficients[var] = c
            intercept = clf.intercept_[0]

            return LogisticCPD(node=node, parents=parents, intercept=intercept, coefficients=coefficients)

        # nodes that only have fully congested or fully uncongested data will have a deterministic probability of 100%
        # for their respective classes
        elif len(self.data[node].unique()) == 1:
            if self.data[node].unique() == [1]:
                prob0 = 0.01
                prob1 = 0.99
            else:
                prob0 = 0.99
                prob1 = 0.01
            return LogisticCPD(node=node, parents=parents, probs=[prob0, prob1])

        # variables with no parents can just use state counts
        else:
            MLE = DynamicMaximumLikelihoodEstimator(self.model, self.data)
            tab_cpd = MLE.estimate_cpd(node)
            return LogisticCPD(node=node, parents=parents, probs=tab_cpd.values)


class LogisticCPD:
    def __init__(self, node, parents, intercept=None, coefficients=None, probs=None):

        self.variable = node
        self.parents = parents
        self.scope = [self.variable] + self.parents
        self.linkID, self.t = get_att(self.variable)
        self.intercept = intercept
        self.coefficients = coefficients
        self.probs = probs

        if self.probs is not None:
            self.distribution = self.bernoulli
            self.values = self.get_vals()
        else:
            self.distribution = self.sigmoid
            self.values = self.get_vals()

        self.values = self.values.reshape([2 for _ in range(0, len(self.scope))])
        self.val_dict = self._get_dict()

    def bernoulli(self, evidence=None, var_state=None):

        prob0, prob1 = self.probs

        if evidence:
            for variable, state in evidence.items():
                if variable == self.variable:
                    var_state = state

        # which probability to return based on the state of the variable
        if var_state == 0:
            return prob0
        elif var_state == 1:
            return prob1
        else:
            return prob0, prob1

    def sigmoid(self, evidence, var_state=None):

        func = self.intercept
        for variable, state in evidence.items():
            if variable == self.variable:
                var_state = state
            else:
                func += self.coefficients[variable] * state

        prob1 = 1. / (1. + np.exp(-func))
        prob0 = 1 - prob1

        # which probability to return based on the state of the variable
        if var_state == 0:
            return prob0
        elif var_state == 1:
            return prob1
        else:
            return prob0, prob1

    def _binary_combinations(self):

        """
        This function finds all possible combinations of binary values for the variables
        """
        return [tuple(i) for i in product([0, 1], repeat=len(self.scope))]

    def get_vals(self):

        evidence = {}
        values = []
        for e in self._binary_combinations():
            for variable, val in zip(self.scope, e):
                evidence[variable] = val

            values.append(self.distribution(evidence))

        return np.array(values)

    def _get_dict(self):

        selfcopy = deepcopy(self)

        # produces a dictionary where each key is a hashable dictionary of variable
        # binary values pairs and the dictionary value is it's conditional probability
        val_dict = hashabledict({})
        for index in selfcopy._binary_combinations():
            index_dict = hashabledict({})
            for i in range(len(selfcopy.scope)):
                # building the keys where scope[i] is the variable and index[i] is its binary value
                index_dict[selfcopy.scope[i]] = index[i]
            # the final dictionary where the binary coordinates are used to search for the
            # correct Pr value in the cpd table
            val_dict[index_dict] = selfcopy.values[index]

        return val_dict

    def __str__(self):
        header = [variable for variable in self.scope]
        header.append('Probability')
        data = np.array(list(product([0, 1], repeat=len(self.scope))))
        data = np.column_stack((data, self.values.flatten('C').reshape(-1, 1)))

        return tabulate(data, headers=header, tablefmt="grid")