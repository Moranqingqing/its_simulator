import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prediction_models.neural_nets.nn_predictor import NNPredictor


# modified from Semi-Supervised Classification with Graph Convolutional Networks. Kipf et al. 2016.
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.init_parameters()

    def init_parameters(self):
        std = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input, adj):
        """
        Parameters
        ----------
        input: matrix. B x seq_len x N x F
        adj: matrix. N x N

        Returns
        -------
        output: matrix. N x out_features
        """
        support = torch.matmul(input, self.weight)  # XW. (B, seq_len, N, F) x (F, out_features) = (B, seq_len, N, out_features)
        output = torch.matmul(adj, support)  # AXW. (N, N) x (B, seq_len, N, out_features) = (B, seq_len, N, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """
    Parameters
    ----------
    predictor : an instance of class GCNPredictor
    num_hidden : int
        The size of hidden representation of each node
    num_out_features : int
        output dimension of the last GCN layer (gcn_outdim), which will be connected to the FC layer.
    dropout_prob : float
        a float between 0.0 and 1.0, the dropout probability.
    """
    def __init__(self, predictor, num_hidden, num_out_features, dropout_prob):
        super(GCN, self).__init__()
        self.predictor = predictor

        self.adj = self.predictor.build_adj_matrix()

        self.num_in_features = len(predictor.source_feature)
        self.num_target_features = len(predictor.target_feature)
        self.layer1 = GraphConvolutionLayer(self.num_in_features, num_hidden)
        self.layer2 = GraphConvolutionLayer(num_hidden, num_out_features)
        '''layer3, a FC layer, is defined below.'''

        self.dropout_prob = dropout_prob

        self.num_links = self.predictor.traffic_data.data.shape[1]  # (F, N, T) -> N
        self.input_size = self.predictor.seq_len * self.num_links * num_out_features
        if isinstance(self.predictor.horizon, int):
            self.output_size = self.num_links * self.num_target_features * self.predictor.horizon
        elif isinstance(self.predictor.horizon, list):
            self.output_size = self.num_links * self.num_target_features * len(self.predictor.horizon)
        else:
            raise ValueError
        print(self.input_size, self.output_size)
        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # x: B x seq_len x NF
        # reshape to B x seq_len x N x F
        x = x.view(x.shape[0], x.shape[1], -1, self.num_in_features)
        # 2 GCN layers + 1 FC layer
        x = F.relu(self.layer1(x, self.adj))  # (B, seq_len, N, H)
        x = F.dropout(x, self.dropout_prob, training=self.training)  # or use nn.Dropout(self.dropout_prob) # different from F.dropout
        x = self.layer2(x, self.adj)  # (B, seq_len, N, F_O), F_O==gcn_outdim
        x = x.view(x.shape[0], -1)  # (B, seq_len*N*F_O)
        x = self.fc(x)  # (B, horizon*N*F_out), F_out == target_feature
        horizon = self.predictor.horizon if isinstance(self.predictor.horizon, int) else len(self.predictor.horizon)
        x = x.view(x.shape[0], horizon, self.num_links, len(self.predictor.target_feature))  # (B, horizon, N, F_out)
        x = x.squeeze(-1)
        return x


class GCNPredictor(NNPredictor):

    def __init__(self, traffic_data, hidden_size, batch_size, is_pems, adj_type='connect', max_epoch=5, learning_rate=1e-4, past_steps=0, horizon=1,
                 dropout=0,
                 gcn_outdim=16, target_feature=None, save=False, save_path='../models/GCN/', n_hops=1, broken_prob=0, source_feature=None,
                 **optimizer_args):
        super().__init__(traffic_data, batch_size, seq_len=past_steps + 1, max_epoch=max_epoch, learning_rate=learning_rate,
                         target_feature=target_feature, horizon=horizon, save=save, save_path=save_path, broken_prob=broken_prob,
                         source_feature=source_feature, is_pems=is_pems, **optimizer_args)
        assert adj_type in ['connect', 'distance'], "adjacency matrix type must be either 'connect' or 'distance'"
        self.adj_type = adj_type  # for pems dataset only.
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_hops = n_hops
        self.gcn_outdim = gcn_outdim
        self.model = GCN(self, hidden_size, self.gcn_outdim, self.dropout)  # self == GCNPredictor
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            # save instance of class
            self.save(save_path)

    def __str__(self):
        return f"GCN-i{self.model.num_in_features}-k{self.n_hops}-h{self.hidden_size}-o{self.gcn_outdim}-b{self.batch_size}-" \
               f"lr{self.learning_rate}-e{self.max_epoch}-p{self.seq_len - 1}-hor{self.horizon}"

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)

    def build_adj_matrix(self):
        if self.is_pems:
            adj_matrix = self.traffic_data.adjacency_matrix[self.adj_type]
            return torch.from_numpy(adj_matrix).float().to(self.device)

        num_links = len(self.traffic_data.links)
        adj_matrix = np.eye(num_links)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = self.get_neighbors(link)
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_matrix[i, link_indices] = 1
        adj_matrix = torch.from_numpy(adj_matrix).float().to(self.device)
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
