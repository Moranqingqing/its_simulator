import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prediction_models.neural_nets.nn_predictor import NNPredictor


class LocalFCNModel(nn.Module):
    def __init__(self, predictor):
        super(LocalFCNModel, self).__init__()

        self.predictor = predictor

        self.num_links = len(self.predictor.traffic_data.links)
        self.num_features = len(self.predictor.traffic_data.features) * self.predictor.seq_len
        self.adj_matrix = self.predictor.build_adj_matrix()

        # set the activation function
        if self.predictor.activation == 'relu':
            self.activation = F.relu
        elif self.predictor.activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif self.predictor.activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError

        input_size = [int(np.sum(self.adj_matrix, axis=1)[i] * self.num_features) for i in range(self.num_links)]
        self.max_input = max(input_size)
        if isinstance(self.predictor.horizon, int):
            output_size = len(self.predictor.target_feature) * self.predictor.horizon
        elif isinstance(self.predictor.horizon, list):
            output_size = len(self.predictor.target_feature) * len(self.predictor.horizon)
        else:
            raise ValueError

        self.fc1_weight = nn.Parameter(torch.empty(self.num_links, self.max_input, self.predictor.hidden_size), requires_grad=True)
        self.fc1_bias = nn.Parameter(torch.empty(self.num_links, self.predictor.hidden_size), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.max_input)
        nn.init.uniform_(self.fc1_weight, -stdv, stdv)
        nn.init.uniform_(self.fc1_bias, -stdv, stdv)

        self.dropout = nn.Dropout(p=self.predictor.dropout)
        self.fc2_weight = nn.Parameter(torch.empty(self.num_links, self.predictor.hidden_size, output_size), requires_grad=True)
        self.fc2_bias = nn.Parameter(torch.empty(self.num_links, output_size), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.predictor.hidden_size)
        nn.init.uniform_(self.fc2_weight, -stdv, stdv)
        nn.init.uniform_(self.fc2_bias, -stdv, stdv)

    def forward(self, x):
        horizon = self.predictor.horizon if isinstance(self.predictor.horizon, int) else len(self.predictor.horizon)

        x = x.view(x.shape[0], self.num_features, self.num_links)

        output = torch.zeros(x.shape[0], self.num_links, self.max_input)
        output = output.to(self.predictor.device)

        for i in range(self.num_links):
            neighbors = np.where(self.adj_matrix[i, :] == 1)[0]
            in_i = x[:, :, neighbors].view(x.shape[0], -1)
            output[:, i, :in_i.shape[1]] = in_i

        output = torch.einsum('bij, ijk->bik', output, self.fc1_weight) + self.fc1_bias
        output = self.activation(output)
        output = self.dropout(output)
        output = torch.einsum('bij, ijk->bik', output, self.fc2_weight) + self.fc2_bias
        output = output.view(x.shape[0], self.num_links, horizon, len(self.predictor.target_feature))
        output = torch.transpose(output, 1, 2)
        output = output.view(x.shape[0], horizon, -1)
        return output


class LocalFCNPredictor(NNPredictor):
    def __init__(self, traffic_data, hidden_size, batch_size, n_hops=0, past_steps=0, max_epoch=5, learning_rate=1e-4, dropout=0, activation='relu',
                 horizon=1, target_feature=None, save=False, save_path='../models/local_fcn/', **optimizer_args):
        super().__init__(traffic_data, batch_size=batch_size, seq_len=past_steps + 1, max_epoch=max_epoch, learning_rate=learning_rate,
                         target_feature=target_feature, horizon=horizon, save=save, save_path=save_path, **optimizer_args)

        self.n_hops = n_hops
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation

        # create model
        self.model = LocalFCNModel(self)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "nnr-h{}".format(self.hidden_size)

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)

    def build_adj_matrix(self):
        num_links = len(self.traffic_data.links)
        adj_matrix = np.eye(num_links)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = self.get_neighbors(link)
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_matrix[i, link_indices] = 1
        return adj_matrix

    def get_neighbors(self, target_link):
        """
        Get all the links within n hops from target by breadth first search with n iterations

        Parameters
        ----------
        target_link : str
            The target link to start the search from

        Returns
        -------
        list of str
            The links within the neighborhood of target, excluding target itself
        """
        neighbors = [target_link]
        # breadth first search with n iterations
        for _ in range(self.n_hops):
            new_neighbors = [n for n in neighbors]
            for neighbor in neighbors:
                for direction in ['left', 'straight', 'right']:
                    new_neighbors.extend(self.traffic_data.network.downstream[neighbor][direction])
                    new_neighbors.extend(self.traffic_data.network.upstream[neighbor][direction])
            new_neighbors = list(set(new_neighbors))
            neighbors = new_neighbors
        # remove target_link itself from neighbors
        neighbors.remove(target_link)
        return neighbors
