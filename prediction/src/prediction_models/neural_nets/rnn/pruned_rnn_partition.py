import math

import numpy as np
import torch
import torch.nn as nn

from src.prediction_models.neural_nets.rnn.base_rnn import BaseRNNPredictor


class PrunedRNNModel(nn.Module):
    """
    PyTorch model for Pruned RNN Predictor. Using a custom GRU implementation

    Parameters
    ----------
    predictor : PrunedRNNPredictor
        The PrunedRNNPredictor instance that is the parent of this model

    Attributes
    ----------
    predictor : PrunedRNNPredictor
        The PrunedRNNPredictor instance that is the parent of this model
    num_links : int
        The number of links in the dataset
    num_features : int
        The number of features in the dataset
    attention
        The attention matrix of the road network graph, with the diagonal initialized to 1 and immediate neighbors initialized to alpha
    reset_hidden_weight
        The hidden-hidden weights in the reset gate of the GRU implementation
    reset_hidden_bias
        The hidden-hidden bias in the reset gate of the GRU implementation
    reset_input_weight
        The input-hidden weights in the reset gate of the GRU implementation
    reset_input_bias
        The input-hidden bias in the reset gate of the GRU implementation
    update_hidden_weight
        The hidden-hidden weights in the update gate of the GRU implementation
    update_hidden_bias
        The hidden-hidden bias in the update gate of the GRU implementation
    update_input_weight
        The input-hidden weights in the update gate of the GRU implementation
    update_input_bias
        The input-hidden bias in the update gate of the GRU implementation
    new_hidden_weight
        The hidden-hidden weights in the output unit of the GRU implementation
    new_hidden_bias
        The hidden-hidden bias in the output unit of the GRU implementation
    new_input_weight
        The input-hidden weights in the output unit of the GRU implementation
    new_input_bias
        The input-hidden bias in the output unit of the GRU implementation
    fc_weight
        The fully connected layer weight used to generate output from hidden representations
    fc_bias
        The fully connected layer bias used to generate output from hidden representations
    """

    def __init__(self, predictor):
        super(PrunedRNNModel, self).__init__()

        self.predictor = predictor

        self.num_links = len(self.predictor.traffic_data.links)
        self.num_features = len(self.predictor.traffic_data.features)
        self.hidden_size = self.predictor.hidden_size
        self.output_size = 0
        self.num_partitions = self.predictor.num_partitions
        self.partition_size = self.predictor.partition_size
        self.partition_sizes = self.predictor.partition_sizes

        self.adj_matrix = self.predictor.build_adj_matrix()
        self.attention = nn.ParameterList(
            [nn.Parameter(self.adj_matrix[i * self.partition_size:(i + 1) * self.partition_size, :], requires_grad=False)
             for i in range(self.num_partitions)])

        self.broken_prob = 0

        self.reset_hidden_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.hidden_size, self.hidden_size), requires_grad=False)
                                                     for size in self.partition_sizes])
        self.reset_hidden_bias = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_size, size), requires_grad=False)
                                                   for size in self.partition_sizes])
        self.reset_input_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.num_features, self.hidden_size), requires_grad=False)
                                                    for size in self.partition_sizes])
        self.reset_input_bias = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_size, size), requires_grad=False)
                                                  for size in self.partition_sizes])

        self.update_hidden_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.hidden_size, self.hidden_size), requires_grad=False)
                                                      for size in self.partition_sizes])
        self.update_hidden_bias = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_size, size), requires_grad=False)
                                                    for size in self.partition_sizes])
        self.update_input_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.num_features, self.hidden_size), requires_grad=False)
                                                     for size in self.partition_sizes])
        self.update_input_bias = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_size, size), requires_grad=False)
                                                   for size in self.partition_sizes])

        self.new_hidden_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.hidden_size, self.hidden_size), requires_grad=False)
                                                   for size in self.partition_sizes])
        self.new_hidden_bias = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_size, size), requires_grad=False)
                                                 for size in self.partition_sizes])
        self.new_input_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.num_features, self.hidden_size), requires_grad=False)
                                                  for size in self.partition_sizes])
        self.new_input_bias = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_size, size), requires_grad=False)
                                                for size in self.partition_sizes])

        self.fc_weight = nn.ParameterList([nn.Parameter(torch.empty(size, self.hidden_size, len(self.predictor.target_feature)), requires_grad=False)
                                           for size in self.partition_sizes])
        self.fc_bias = nn.ParameterList([nn.Parameter(torch.empty(len(self.predictor.target_feature), size), requires_grad=False)
                                         for size in self.partition_sizes])

        self.trained_partition = None

        self.initialize()

        # self.gru = nn.ModuleList([nn.GRUCell(self.num_features, self.hidden_size) for _ in range(self.num_links)])
        # self.fc = nn.ModuleList([nn.Linear(self.hidden_size, len(self.predictor.target_feature)) for _ in range(self.num_links)])

    def initialize(self):
        """
        Initializes the network weights using the same initialization scheme as PyTorch GRU
        https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight_list in [self.reset_hidden_weight, self.reset_hidden_bias, self.reset_input_weight, self.reset_input_bias,
                            self.update_hidden_weight, self.update_hidden_bias, self.update_input_weight, self.update_input_bias,
                            self.new_hidden_weight, self.new_hidden_bias, self.new_input_weight, self.new_input_bias, self.fc_weight, self.fc_bias]:
            for weight in weight_list:
                nn.init.uniform_(weight, -stdv, stdv)

    def set_trained_links(self, partition):
        """
        Train the model only on the links provided by setting the requires_grad property for those links to True.
        Note: link_indices is a list of indices rather than list of link names

        Parameters
        ----------
        partition : int
            The partition to train
        """
        self.trained_partition = partition
        self.output_size = self.partition_sizes[partition] * len(self.predictor.target_feature)
        for i in range(self.num_partitions):
            if i == partition:
                self.attention[i].requires_grad = True
                self.reset_hidden_weight[i].requires_grad = True
                self.reset_hidden_bias[i].requires_grad = True
                self.reset_input_weight[i].requires_grad = True
                self.reset_input_bias[i].requires_grad = True
                self.update_hidden_weight[i].requires_grad = True
                self.update_hidden_bias[i].requires_grad = True
                self.update_input_weight[i].requires_grad = True
                self.update_input_bias[i].requires_grad = True
                self.new_hidden_weight[i].requires_grad = True
                self.new_hidden_bias[i].requires_grad = True
                self.new_input_weight[i].requires_grad = True
                self.new_input_bias[i].requires_grad = True
                self.fc_weight[i].requires_grad = True
                self.fc_bias[i].requires_grad = True

            else:
                self.attention[i].requires_grad = False
                self.reset_hidden_weight[i].requires_grad = False
                self.reset_hidden_bias[i].requires_grad = False
                self.reset_input_weight[i].requires_grad = False
                self.reset_input_bias[i].requires_grad = False
                self.update_hidden_weight[i].requires_grad = False
                self.update_hidden_bias[i].requires_grad = False
                self.update_input_weight[i].requires_grad = False
                self.update_input_bias[i].requires_grad = False
                self.new_hidden_weight[i].requires_grad = False
                self.new_hidden_bias[i].requires_grad = False
                self.new_input_weight[i].requires_grad = False
                self.new_input_bias[i].requires_grad = False
                self.fc_weight[i].requires_grad = False
                self.fc_bias[i].requires_grad = False

    def horizon_length(self):
        """
        Get the length of prediction horizon

        Returns
        -------
        horizon
            the length of prediction horizon
        """
        if isinstance(self.predictor.horizon, int):
            horizon = self.predictor.horizon
        elif self.predictor.predictor_method == 'rolling':
            horizon = max(self.predictor.horizon)
        elif self.predictor.predictor_method == 'separate':
            horizon = len(self.predictor.horizon)
        else:
            raise ValueError

        return horizon

    # def build_weights(self):
    #     trained_gru_weights = []
    #     trained_fc_weights = []
    #
    #     trained_gru_weights.append(torch.stack([self.reset_hidden_weight[i] for i in self.trained_indices], dim=0))
    #     trained_gru_weights.append(torch.stack([self.reset_hidden_bias[i] for i in self.trained_indices], dim=1))
    #     trained_gru_weights.append(torch.stack([self.reset_input_weight[i] for i in self.trained_indices], dim=0))
    #     trained_gru_weights.append(torch.stack([self.reset_input_bias[i] for i in self.trained_indices], dim=1))
    #     trained_gru_weights.append(torch.stack([self.update_hidden_weight[i] for i in self.trained_indices], dim=0))
    #     trained_gru_weights.append(torch.stack([self.update_hidden_bias[i] for i in self.trained_indices], dim=1))
    #     trained_gru_weights.append(torch.stack([self.update_input_weight[i] for i in self.trained_indices], dim=0))
    #     trained_gru_weights.append(torch.stack([self.update_input_bias[i] for i in self.trained_indices], dim=1))
    #     trained_gru_weights.append(torch.stack([self.new_hidden_weight[i] for i in self.trained_indices], dim=0))
    #     trained_gru_weights.append(torch.stack([self.new_hidden_bias[i] for i in self.trained_indices], dim=1))
    #     trained_gru_weights.append(torch.stack([self.new_input_weight[i] for i in self.trained_indices], dim=0))
    #     trained_gru_weights.append(torch.stack([self.new_input_bias[i] for i in self.trained_indices], dim=1))
    #
    #     trained_fc_weights.append(torch.stack([self.fc_weight[i] for i in self.trained_indices], dim=0))
    #     trained_fc_weights.append(torch.stack([self.fc_bias[i] for i in self.trained_indices], dim=1))
    #
    #     return trained_gru_weights, trained_fc_weights

    def forward_propagate(self, hidden, in_i):
        """
        Forward propagate the GRU by a single time step

        Parameters
        ----------
        hidden
            The previous hidden state (shape [batch_size, hidden_size, num_links])
        in_i
            The current input (shape [batch_size, num_features, num_links])

        Returns
        -------
        hidden
            The updated hidden state (shape [batch_size, hidden_size, num_links])
        """
        if self.training:
            reset_hidden_weight = self.reset_hidden_weight[self.trained_partition]
            reset_hidden_bias = self.reset_hidden_bias[self.trained_partition]
            reset_input_weight = self.reset_input_weight[self.trained_partition]
            reset_input_bias = self.reset_input_bias[self.trained_partition]

            update_hidden_weight = self.update_hidden_weight[self.trained_partition]
            update_hidden_bias = self.update_hidden_bias[self.trained_partition]
            update_input_weight = self.update_input_weight[self.trained_partition]
            update_input_bias = self.update_input_bias[self.trained_partition]

            new_hidden_weight = self.new_hidden_weight[self.trained_partition]
            new_hidden_bias = self.new_hidden_bias[self.trained_partition]
            new_input_weight = self.new_input_weight[self.trained_partition]
            new_input_bias = self.new_input_bias[self.trained_partition]
        else:
            reset_hidden_weight = torch.cat([i for i in self.reset_hidden_weight], dim=0)
            reset_hidden_bias = torch.cat([i for i in self.reset_hidden_bias], dim=1)
            reset_input_weight = torch.cat([i for i in self.reset_input_weight], dim=0)
            reset_input_bias = torch.cat([i for i in self.reset_input_bias], dim=1)

            update_hidden_weight = torch.cat([i for i in self.update_hidden_weight], dim=0)
            update_hidden_bias = torch.cat([i for i in self.update_hidden_bias], dim=1)
            update_input_weight = torch.cat([i for i in self.update_input_weight], dim=0)
            update_input_bias = torch.cat([i for i in self.update_input_bias], dim=1)

            new_hidden_weight = torch.cat([i for i in self.new_hidden_weight], dim=0)
            new_hidden_bias = torch.cat([i for i in self.new_hidden_bias], dim=1)
            new_input_weight = torch.cat([i for i in self.new_input_weight], dim=0)
            new_input_bias = torch.cat([i for i in self.new_input_bias], dim=1)

        reset = torch.sigmoid(torch.einsum('bij, jik->bkj', hidden, reset_hidden_weight) + reset_hidden_bias
                              + torch.einsum('bij, jik->bkj', in_i, reset_input_weight) + reset_input_bias)
        update = torch.sigmoid(torch.einsum('bij, jik->bkj', hidden, update_hidden_weight) + update_hidden_bias
                               + torch.einsum('bij, jik->bkj', in_i, update_input_weight) + update_input_bias)
        new = torch.tanh(reset * (torch.einsum('bij, jik->bkj', hidden, new_hidden_weight) + new_hidden_bias)
                         + torch.einsum('bij, jik->bkj', in_i, new_input_weight) + new_input_bias)
        hidden = (torch.ones_like(update) - update) * new + update * hidden

        return hidden

    def forward(self, x):
        # Set an initial hidden state
        if self.training:
            # [batch_size, hidden_size, partition_size]
            hidden_trained = torch.randn(x.shape[0], self.predictor.hidden_size, self.partition_sizes[self.trained_partition])
            hidden_trained = hidden_trained.to(self.predictor.device)

            attn_trained = self.attention[self.trained_partition].T
        else:
            hidden_trained = torch.randn(x.shape[0], self.predictor.hidden_size, self.num_links)
            hidden_trained = hidden_trained.to(self.predictor.device)

            attn_trained = torch.cat([i for i in self.attention], dim=0)
            attn_trained = attn_trained.T

        # Forward propagate the RNN
        for i in range(self.predictor.seq_len):
            # input at time i
            in_i = x[:, i, :].view((x.shape[0], self.num_features, self.num_links))  # [batch_size, num_features, num_links]

            # propagate hidden representations to neighbors using the attention matrix
            in_trained = torch.matmul(in_i, attn_trained)  # [batch_size, num_features, partition_size]

            # forward propagate
            hidden_trained = self.forward_propagate(hidden_trained, in_trained)

        # the length of prediction horizon
        horizon = self.horizon_length()

        # create an empty tensor for the output
        if self.training:
            output_size = self.output_size
            output = torch.empty(x.shape[0], horizon, self.output_size).to(self.predictor.device)
            fc_weight = self.fc_weight[self.trained_partition]
            fc_bias = self.fc_bias[self.trained_partition]
        else:
            output_size = len(self.predictor.traffic_data.links) * len(self.predictor.target_feature)
            output = torch.empty(x.shape[0], horizon, output_size).to(self.predictor.device)
            fc_weight = torch.cat([i for i in self.fc_weight], dim=0)
            fc_bias = torch.cat([i for i in self.fc_bias], dim=1)

        # unroll the prediction horizon
        for i in range(horizon):
            # Pass the hidden representations to the fully connected layer to get prediction
            # [batch_size, num_out_features, partition_size]
            out_i = torch.einsum('bij, jik->bkj', hidden_trained, fc_weight) + fc_bias
            output[:, i, :] = out_i.view((x.shape[0], output_size))

            # the input to the next time step is the current prediction
            # prediction does not necessarily contain all input features, so the non-predicted features are masked to 0
            in_i = torch.zeros((x.shape[0], self.num_features, self.num_links)).to(self.predictor.device)
            feature_indices = [self.predictor.traffic_data.features.index(i) for i in self.predictor.target_feature]
            if self.training:
                in_i[:, feature_indices, self.trained_partition * self.partition_size:(self.trained_partition + 1) * self.partition_size] = out_i
            else:
                in_i[:, feature_indices, :] = out_i

            # propagate hidden representations to neighbors using the attention matrix
            in_trained = torch.matmul(in_i, attn_trained)  # [batch_size, hidden_size, num_links]

            # forward propagate
            hidden_trained = self.forward_propagate(hidden_trained, in_trained)

        if isinstance(self.predictor.horizon, list) and self.predictor.predictor_method == 'rolling':
            horizon = [i - 1 for i in self.predictor.horizon]
            output = output[:, horizon, :]

        return output


class PrunedRNNPredictor(BaseRNNPredictor):
    """
    A predictor with no tied parameters across links. Hidden states are created using a fully-connected network pruned according to the graph
    structure and updated with the GRU update rules

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    hidden_size : int
        The size of hidden representation of each node
    batch_size : int
        The size of each training batch
    seq_len : int
        The length of each sequence
    max_epoch : int
        when to stop training
    learning_rate : float
    target_feature : list of str
        what features to predict, default: ['speed']
    horizon : int or list of int
        which horizons to predict, default 1
        if int is provided, prediction is generated for every horizon up to the number provided
        if list is provided, only prediction for the specified horizons are generated
    predictor_method : {'rolling', 'separate'}
        if horizon is a list, specify the method for the prediction side of RNN
        'rolling' means every horizon is predicted internally but only the specified horizons are used
        'separate' means that only the specified horizons are predicted, and they are predicted independently
    save : bool
        whether to save model
    save_path : str
        where to save the model
    **optimizer_args
        Other keyword arguments to the PyTorch optimizer

    Attributes
    ----------
    alpha : float
        The alpha parameter for neighbor propagation in the GRNN framework
    predictor_method : {'rolling', 'separate'}
    model
        The pruned RNN propagation model
    """

    def __init__(self, traffic_data, hidden_size, batch_size, seq_len, alpha=0.2, max_epoch=5, learning_rate=1e-4, target_feature=None, horizon=1,
                 predictor_method='rolling', partition_size=200, save=False, save_path='../models/gc_rnn/', **optimizer_args):
        super().__init__(traffic_data=traffic_data, hidden_size=hidden_size, batch_size=batch_size, seq_len=seq_len, model_type=None,
                         max_epoch=max_epoch, learning_rate=learning_rate, target_feature=target_feature, horizon=horizon, save=save,
                         save_path=save_path, **optimizer_args)

        self.alpha = alpha
        self.predictor_method = predictor_method
        self.partition_size = partition_size
        self.num_partitions = math.ceil(len(self.traffic_data.links) / self.partition_size)
        self.partition_sizes = [self.partition_size] * math.floor(len(self.traffic_data.links) / self.partition_size)
        self.partition_sizes.append(len(self.traffic_data.links) % self.partition_size)

        # create model and send to gpu
        self.model = PrunedRNNModel(self)
        self.model.to(self.device)

        # build gradient mask to limit propagation
        self.mask = self.build_mask()

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "gc_rnn-{}-h{}-s{}-b{}-lr{}-w{}".format(self.predictor_method, self.hidden_size, self.seq_len,
                                                       self.batch_size, self.learning_rate, self.optimizer_args['weight_decay'])

    def train(self):
        """
        Trains the neural network with MSE Loss and Adam optimizer
        """
        data, label = self.get_data('train')

        train_loader = self.to_data_loader(data, label, self.batch_size, shuffle=True)

        self.model.train()

        for partition in range(self.num_partitions):
            self.model.set_trained_links(partition)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam([i for i in self.model.parameters() if i.requires_grad], lr=self.learning_rate, **self.optimizer_args)

            for epoch in range(1, self.max_epoch + 1):
                for train, target in train_loader:
                    train = train.to(self.device)
                    target = target.to(self.device)
                    target = target[:, :, partition * self.partition_size:(partition + 1) * self.partition_size]
                    optimizer.zero_grad()
                    prediction = self.model(train)
                    loss = criterion(target, prediction)
                    loss.backward()

                    self.model.attention[partition].grad *= self.mask[partition]

                    optimizer.step()

                # if epoch % 50 == 0 and self.save_model:
                #     self.save_pytorch_model(self.save_path + 'e{}.pt'.format(epoch))

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)

    def build_adj_matrix(self):
        num_links = len(self.traffic_data.links)
        adj_matrix = torch.eye(num_links, device=self.device)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = []
            for direction in ['left', 'straight', 'right']:
                neighbors.extend(self.traffic_data.network.downstream[link][direction])
                neighbors.extend(self.traffic_data.network.upstream[link][direction])
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_matrix[i, link_indices] = self.alpha
        return adj_matrix

    def build_adj_list(self):
        adj_list = []
        for link in self.traffic_data.links:
            neighbors = []
            for direction in ['left', 'straight', 'right']:
                neighbors.extend(self.traffic_data.network.downstream[link][direction])
                neighbors.extend(self.traffic_data.network.upstream[link][direction])
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_list.append(link_indices)
        return adj_list

    def build_mask(self):
        num_links = len(self.traffic_data.links)
        mask = torch.eye(num_links, device=self.device)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = self.get_neighbors(link)
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            mask[i, link_indices] = 1

        mask = [mask[i * self.partition_size:(i + 1) * self.partition_size, :] for i, _ in enumerate(self.partition_sizes)]
        return mask

    def build_neighbors_list(self):
        neighbors_list = []
        for link in self.traffic_data.links:
            neighbors = self.get_neighbors(link)
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            neighbors_list.append(link_indices)
        return neighbors_list

    def get_neighbors(self, target_link, n=5):
        """
        Get all the links within n hops from target by breadth first search with n iterations

        Parameters
        ----------
        target_link : str
            The target link to start the search from
        n : int
            Number of iterations of breadth first search

        Returns
        -------
        list of str
            The links within the neighborhood of target, excluding target itself
        """
        neighbors = [target_link]
        # breadth first search with n iterations
        for _ in range(n):
            new_neighbors = [n for n in neighbors]
            for neighbor in neighbors:
                for direction in ['left', 'straight', 'right']:
                    new_neighbors.extend(self.traffic_data.network.downstream[neighbor][direction])
                    new_neighbors.extend(self.traffic_data.network.upstream[neighbor][direction])
            new_neighbors = list(set(new_neighbors))
            neighbors = new_neighbors

        return neighbors
