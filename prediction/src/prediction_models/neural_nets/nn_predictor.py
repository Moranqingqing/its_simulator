import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from src.metrics import compute_errors_all
from src.prediction_models.prediction_model import PredictionModel


class NNPredictor(PredictionModel):
    """
    Base class for all PyTorch deep learning based predictors

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    batch_size : int
        The size of each training batch
    seq_len : int
        The length of each sequence
    max_epoch : int
        when to stop training
    learning_rate : float
    source_feature : list of str
        what features to input, default: ['flow', 'speed']
    target_feature : list of str
        what features to predict, default: ['speed']
    is_pems : bool
        whether the dataset is PeMS or not.
    horizon : int or list of int
        which horizons to predict, default 1
        if int is provided, prediction is generated for every horizon up to the number provided
        if list is provided, only prediction for the specified horizons are generated
    save : bool
        whether to save model
    save_path : str
        where to save the model
    **optimizer_args
        Other keyword arguments to the PyTorch optimizer

    Attributes
    ----------
    device
        The device that PyTorch will use in computation, GPU if available
    batch_size : int
        The size of each training batch
    seq_len : int
        The sequence length for each data point, used when retrieving data
    max_epoch : int
        when to stop training
    learning_rate : float
    save_model : bool
        whether to save model
    save_path : str
        where to save the model
    optimizer_args : dict
        Other keyword arguments to the PyTorch optimizer
    """

    def __init__(self, traffic_data, batch_size, seq_len, max_epoch, learning_rate, target_feature, source_feature=None,
                 is_pems=False, horizon=1, save=False, save_path=None, save_interval=50, broken_prob=0.0, **optimizer_args):
        super().__init__(traffic_data, source_feature=source_feature, target_feature=target_feature, horizon=horizon, broken_prob=broken_prob)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.is_pems = is_pems
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate

        self.save_model = save
        self.save_path = save_path
        self.save_interval = save_interval  # save every save_interval epochs
        self.optimizer_args = optimizer_args

    def train(self):
        """
        Trains the neural network with MSE Loss and Adam optimizer
        """
        data, label = self.get_data('train')

        train_loader = self.to_data_loader(data, label, self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, **self.optimizer_args)

        self.model.train()  # set to train mode.

        for epoch in range(1, self.max_epoch + 1):
            for train, target in train_loader:
                train = train.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                prediction = self.model(train)
                loss = criterion(target, prediction)
                loss.backward()
                optimizer.step()

            if epoch % self.save_interval == 0 and self.save_model:
                self.save_pytorch_model(self.save_path + 'e{}.pt'.format(epoch))
            errors = compute_errors_all(self)
            print("epoch {}: train loss: {}".format(epoch, loss.item()))
            print(errors)

    def evaluate(self, mode, links=None):
        data, label = self.get_data(mode)

        eval_loader = self.to_data_loader(data, label, self.batch_size)

        x_true = []
        x_pred = []

        self.model.eval()

        with torch.no_grad():
            for data, target in eval_loader:
                data = data.to(self.device)
                target = target.cpu().numpy()

                prediction = self.model(data)
                prediction = prediction.cpu().detach().numpy()

                x_true.append(target)
                x_pred.append(prediction)

        x_true = np.concatenate(x_true, axis=0)
        x_pred = np.concatenate(x_pred, axis=0)

        x_true = np.transpose(x_true, (2, 0, 1))
        x_pred = np.transpose(x_pred, (2, 0, 1))

        if self.traffic_data.normalizer:
            if len(self.target_feature) != 1:
                raise NotImplementedError("Multiple target features not supported")

            x_true = self.traffic_data.normalizer[self.target_feature[0]].inverse_transform(x_true)
            x_pred = self.traffic_data.normalizer[self.target_feature[0]].inverse_transform(x_pred)

        return x_true, x_pred

    def get_data(self, mode):
        data, label = self.get_network_data(mode)  # (P, seq_len, F*N)
        if isinstance(self.horizon, list):
            horizon = [i - 1 for i in self.horizon]
            label = np.take(label, horizon, axis=1)

        return data, label

    def save(self, path):
        """
        Saves the class attributes and model architecture

        Parameters
        ----------
        path : str
            The folder path that this model will be saved in

        Returns
        -------
        None
        """
        self.save_path = path + self.__str__() + '/'
        os.mkdir(self.save_path)
        with open(self.save_path + 'model.pickle', 'wb') as f:
            pickle.dump(self, f)

    def save_pytorch_model(self, path):
        """
        Saves the PyTorch model parameters

        Parameters
        ----------
        path : str
        """
        torch.save(self.model.state_dict(), path)
        print("PyTorch model dict saved to {}.".format(path))

    def load_pytorch_model(self, path):
        """
        Loads the PyTorch model parameters

        Parameters
        ----------
        path : str
            The pickled PyTorch model to load from
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_network_data(self, mode):
        """
        Get the traffic_data for the entire network

        Parameters
        ----------
        mode : {'train', 'val', 'test'}
            which dataset to use

        Returns
        -------
        data : numpy.array
            shape [time, sequence_length, links*features]
        label : numpy.array
        """
        dataset = self.get_dataset(mode)

        horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)
        truncate = self.seq_len + horizon - 1  # e.g. 10+5-1=14. seq_len=past_steps+1.

        # there total number of sample S is T - truncate. e.g. 9000-14=8986.
        # the last minute of horizon is sliced simultaneously for all samples.
        # e.g. choose speed, truncate=15. [(1, N, S)] # S=T-14=8986.
        label = [dataset.sel(feature=self.target_feature).isel(time=slice(truncate, None)).data]
        # slide window back. slice every minute of horizon simultaneously for all samples,
        # then stack 5 minutes to form 5-minute length samples (labels).
        for i in range(1, horizon):  # e.g. horizon=5.
            label.insert(0, dataset.sel(feature=self.target_feature).isel(time=slice(truncate - i, -i)).data)  # [5, (1, N, S)]

        label = np.concatenate(label, axis=0)  # (5, N, S)
        label = np.transpose(label, (2, 0, 1))  # (S, 5, N)

        # sliding window moving backwards
        # slice every minute of horizon simultaneously for all samples,
        # then stack 10 minutes to form 10-minute length samples (input).
        data = []
        for i in range(self.seq_len):
            # source feature, all links, 1 timestamps. (F, N, S)
            data_i = dataset.sel(feature=self.source_feature).isel(time=slice(i, -truncate + i)).data
            data_i = np.reshape(data_i, (-1, data_i.shape[2]))  # (F*N, S)
            data.append(data_i)  # [seq_len,  (F*N, S)]
        data = np.array(data)  # (seq_len, F*N, S)
        data = np.transpose(data, (2, 0, 1))  # (S, seq_len, F*N)

        if self.traffic_data.run_len:
            data, label = self.remove_invalid_data(data, label)
        if self.broken_prob:
            data = self.add_broken(data)

        if data.ndim == 3:
            # extend (S, seq_len, N) to (S, seq_len, N, 1)
            data = np.expand_dims(data, axis=-1)

        return data, label

    def remove_invalid_data(self, data, label):
        """
        Removes data points to get rid of data, label combo that span 2 different runs

        Parameters
        ----------
        data : numpy.array
            The data to be processed
        label : numpy.array
            The label to be processed

        Returns
        -------
        data : numpy.array
            The processed data
        label : numpy.array
            The process label
        """
        horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)
        truncate = self.seq_len + horizon - 1
        remove = []
        for i in range(truncate):
            remove.extend(list(range(self.traffic_data.run_len - i - 1, len(data), self.traffic_data.run_len)))
        remove = sorted(remove)

        data = np.delete(data, remove, axis=0)
        label = np.delete(label, remove, axis=0)

        return data, label

    def to_data_loader(self, data, label, batch_size, shuffle=False):
        """
        Converts the labeled numpy data into PyTorch DataLoader

        Parameters
        ----------
        data : numpy.array
            The data to be converted
        label : numpy.array
            The label to be converted
        batch_size : int
            The size of each batch in
        shuffle : bool
            Whether the data should be shuffled

        Returns
        -------
        loader : torch.utils.data.DataLoader
        """
        data = torch.from_numpy(data).float().to(self.device)
        label = torch.from_numpy(label).float().to(self.device)
        dataset = torch.utils.data.TensorDataset(data, label)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader
