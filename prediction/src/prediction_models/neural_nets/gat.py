import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prediction_models.neural_nets.nn_predictor import NNPredictor

# Graph Attention Model.

class GATModel(nn.Module):
    def __init__(self, predictor):
        super(GATModel, self).__init__()

        self.predictor = predictor

        self.num_links = len(self.predictor.traffic_data.links)
        self.num_input_features = len(self.predictor.traffic_data.features)
        self.adj_matrix = self.predictor.build_adj_matrix()

        if isinstance(self.predictor.horizon, int):
            self.output_size = len(self.predictor.target_feature) * self.predictor.horizon
        elif isinstance(self.predictor.horizon, list):
            self.output_size = len(self.predictor.target_feature) * len(self.predictor.horizon)
        else:
            raise ValueError

        if not self.predictor.hidden_sizes:
            num_features = [len(self.predictor.traffic_data.features), self.output_size * self.predictor.K[-1]]
            hidden_sizes = [self.num_input_features, self.output_size]
        else:
            num_features = [self.predictor.hidden_sizes[i] * self.predictor.K[i] for i in range(self.predictor.num_layers - 1)]
            num_features.append(self.output_size)
            num_features.insert(0, self.num_input_features)
            hidden_sizes = [i for i in self.predictor.hidden_sizes]
            hidden_sizes.append(self.output_size)
            hidden_sizes.insert(0, self.num_input_features)

        self.layers = nn.ModuleList([GATLayer(self.predictor, num_features[i], hidden_sizes[i + 1], self.predictor.K[i])
                                     for i in range(self.predictor.num_layers - 1)])
        self.output_layer = GATLayer(self.predictor, num_features[-2], num_features[-1], self.predictor.K[-1], output_layer=True)

    def forward(self, x):
        for i in range(self.predictor.num_layers - 1):
            x = self.layers[i](x)
        x = self.output_layer(x)  # [batch_size, num_links, output_size]

        horizon = self.predictor.horizon if isinstance(self.predictor.horizon, int) else len(self.predictor.horizon)
        x = x.view(x.shape[0], self.num_links, len(self.predictor.target_feature), horizon)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.shape[0], horizon, -1)
        return x


class GATLayer(nn.Module):
    def __init__(self, predictor, input_size, output_size, K, output_layer=False):
        super(GATLayer, self).__init__()

        self.predictor = predictor
        self.input_size = input_size
        self.output_size = output_size
        self.K = K # multi-head attention. K == number of heads.
        self.output_layer = output_layer

        self.num_links = len(self.predictor.traffic_data.links)
        self.adj_matrix = self.predictor.build_adj_matrix()

        # multi-head attention with K attention modules
        self.fc = nn.ModuleList([nn.Linear(self.input_size, self.output_size) for _ in range(self.K)])
        self.attention = nn.ModuleList([nn.Linear(2 * self.output_size, 1) for _ in range(self.K)])

    def forward(self, x):
        output = torch.empty(x.shape[0], self.num_links, self.output_size, self.K)
        output = output.to(self.predictor.device)

        x = x.view(x.shape[0], self.input_size, self.num_links)
        x = torch.transpose(x, 1, 2)  # [batch_size, num_links, input_size]

        for i in range(self.num_links):
            for k in range(self.K):
                # feed forward neighbors
                neighbors = np.where(self.adj_matrix[i, :] == 1)[0]
                neighbors_i = x[:, neighbors, :]  # [batch_size, num_neighbors, input_size]
                neighbors_i = self.fc[k](neighbors_i)  # [batch_size, num_neighbors, output_size]

                # feed forward target node
                target = x[:, i, :]  # [batch_size, num_features]
                target = target[:, None, :]  # [batch_size, 1, num_features]
                target = target.repeat(1, len(neighbors), 1)  # [batch_size, num_neighbors, input_size]
                target = self.fc[k](target)  # [batch_size, num_neighbors, output_size]

                # concat the two and calculate attention
                att = torch.cat((target, neighbors_i), dim=2)  # [batch_size, num_neighbors, 2*output_size]
                att = self.attention[k](att).view(x.shape[0], len(neighbors))  # [batch_size, num_neighbors]
                att = F.leaky_relu(att, negative_slope=self.predictor.negative_slope)  # 0.2 is used in the GAT paper
                att = F.softmax(att, dim=1)
                att = att[:, :, None]  # [batch_size, num_neighbors, 1]
                att = att.repeat(1, 1, self.output_size)  # [batch_size, num_neighbors, output_size]

                out_i = torch.mul(neighbors_i, att)  # [batch_size, num_neighbors, output_size]
                out_i = torch.sum(out_i, dim=1)  # [batch_size, output_size]
                output[:, i, :, k] = out_i
        if self.output_layer:
            output = torch.sum(output, dim=3)  # [batch_size, num_links, output_size]
        else:
            output = output.view(x.shape[0], self.num_links, -1)  # [batch_size, num_links, output_size * K]
            output = torch.transpose(output, 1, 2)  # [batch_size, output_size * K, num_links]
        return output


class GATPredictor(NNPredictor):
    """
    Predictor based on Graph Attention Networks (Velickovic, P. at al., 2018)
    """

    def __init__(self, traffic_data, batch_size, num_layers=1, hidden_sizes=None, K=None, negative_slope=0.2, max_epoch=5, learning_rate=1e-4,
                 horizon=1, target_feature=None, save=False, save_path='../models/fcn/', **optimizer_args):
        super().__init__(traffic_data, batch_size=batch_size, seq_len=1, max_epoch=max_epoch, learning_rate=learning_rate,
                         target_feature=target_feature, horizon=horizon, save=save, save_path=save_path, **optimizer_args)

        self.K = K
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        if not K:
            self.K = [1]
        else:
            self.K = K
        self.negative_slope = negative_slope

        # checks
        if len(self.K) != self.num_layers:
            raise ValueError("number of layers does not match K")
        if self.num_layers < 1:
            raise ValueError("there must be at least 1 layer")
        if self.num_layers > 1 and not self.hidden_sizes:
            raise ValueError("hidden size not defined")
        if self.num_layers > 1 and len(self.hidden_sizes) != self.num_layers - 1:
            raise ValueError("number of hidden sizes does not match layers")

        # create model
        self.model = GATModel(self)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "gat-k{}".format(self.K)

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)

    def build_adj_matrix(self):
        num_links = len(self.traffic_data.links)
        adj_matrix = np.eye(num_links)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = []
            for direction in ['left', 'straight', 'right']:
                neighbors.extend(self.traffic_data.network.downstream[link][direction])
                neighbors.extend(self.traffic_data.network.upstream[link][direction])
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_matrix[i, link_indices] = 1
        return adj_matrix
