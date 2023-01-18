import torch
import torch.nn as nn

import math
import numpy as np
import torch.nn.functional as F
from src.prediction_models.neural_nets.nn_predictor import NNPredictor

from model.AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num # N
        self.input_dim = dim_in # F_I
        self.num_layers = num_layers # Number of GNN layers
        self.dcrnn_cells = nn.ModuleList() # Create an empty module list to contain the layers
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True) # E: N*d

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):

        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden;
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) # B, T, C, N
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output




class AGCRNPredictor(NNPredictor):
    """
    (NeurIPS 2020) Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
    """

    def __init__(self, traffic_data, batch_size, num_layers=1, hidden_sizes=None, K=None, negative_slope=0.2,
                 max_epoch=5, learning_rate=1e-4,
                 horizon=1, target_feature=None, save=False, save_path='../models/agcrn/',
                 args,
                 **optimizer_args):
        super().__init__(traffic_data, batch_size=batch_size, seq_len=1, max_epoch=max_epoch,
                         learning_rate=learning_rate,
                         target_feature=target_feature, horizon=horizon, save=save, save_path=save_path,
                         **optimizer_args)

        self.K = K # Chebyshev K
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
        self.model = AGCRN(args)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "agcrn-k{}".format(self.K)

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


