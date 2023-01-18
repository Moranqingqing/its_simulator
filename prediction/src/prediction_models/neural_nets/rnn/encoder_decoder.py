import numpy as np
import torch
import torch.nn as nn

from src.prediction_models.neural_nets.rnn.base_rnn import BaseRNNPredictor


class EncoderDecoderModel(nn.Module):
    """
    The RNN model used in EncoderDecoderPredictor

    Parameters
    ----------
    predictor : PredictionModel
        The predictor that contains this model

    Attributes
    ----------
    predictor : PredictionModel
        The predictor that contains this model
    encoder
        The encoder RNN model
    decoder
        The decoder RNN model
    encoder_fc
        The fully-connected layer that generates the intermediate vector to pass to the decoder
    encoder_fc_cell
        The fully-connected layer that generates the initial cell state to pass to the LSTM decoder
    decoder_fc
        The fully-connected layer that generates the prediction from decoder hidden states

    """

    def __init__(self, predictor):
        super(EncoderDecoderModel, self).__init__()

        self.predictor = predictor

        if self.predictor.model_type == 'lstm':
            model = nn.LSTMCell
        elif self.predictor.model_type == 'gru':
            model = nn.GRUCell
        else:
            raise ValueError

        # the RNN model for the encoder and decoder
        self.encoder = model(self.predictor.input_size, self.predictor.encoder_hidden_size)
        self.decoder = model(self.predictor.output_size, self.predictor.decoder_hidden_size)

        # the fully connected layer(s) that converts the final encoder hidden state to the initial state(s) of the decoder
        self.encoder_fc = nn.Linear(self.predictor.encoder_hidden_size, self.predictor.decoder_hidden_size)
        if self.predictor.model_type == 'lstm':
            self.encoder_fc_cell = nn.Linear(self.predictor.encoder_hidden_size, self.predictor.decoder_hidden_size)

        # the fully conencted layer that converts decoder hidden states to predictions
        self.decoder_fc = nn.Linear(self.predictor.decoder_hidden_size, self.predictor.output_size)

    def forward(self, x):
        # Set an initial hidden state
        encoder_hidden = torch.zeros(x.shape[0], self.predictor.encoder_hidden_size)
        encoder_hidden = encoder_hidden.to(self.predictor.device)

        # initial LSTM cell state
        if self.predictor.model_type == 'lstm':
            encoder_cell = torch.zeros(x.shape[0], self.predictor.encoder_hidden_size)
            encoder_cell = encoder_cell.to(self.predictor.device)

        # Forward propagate the encoder
        for i in range(self.predictor.seq_len):
            if self.predictor.model_type == 'lstm':
                encoder_hidden, encoder_cell = self.encoder(x[:, i, :], (encoder_hidden, encoder_cell))
            else:
                encoder_hidden = self.encoder(x[:, i, :], encoder_hidden)

        # Pass the encoder output to the fully connected layer to get intermediate vector and cell state
        decoder_hidden = self.encoder_fc(encoder_hidden)
        if self.predictor.model_type == 'lstm':
            decoder_cell = self.encoder_fc_cell(encoder_hidden)

        # initialize the output and the initial input of the decoder
        horizon = self.predictor.horizon if isinstance(self.predictor.horizon, int) else max(self.predictor.horizon)
        output = torch.empty(x.shape[0], horizon, self.predictor.output_size).to(self.predictor.device)
        in_i = torch.zeros(x.shape[0], self.predictor.output_size).to(self.predictor.device)

        # forward propagate the decoder
        for i in range(horizon):
            if self.predictor.model_type == 'lstm':
                decoder_hidden, decoder_cell = self.decoder(in_i, (decoder_hidden, decoder_cell))
            else:
                decoder_hidden = self.decoder(in_i, decoder_hidden)

            # generate prediction from hidden state of decoder, then add to the output
            out_i = self.decoder_fc(decoder_hidden)
            output[:, i, :] = out_i
            in_i = out_i

        if isinstance(self.predictor.horizon, list):
            horizon = [i - 1 for i in self.predictor.horizon]
            output = output[:, horizon, :]

        return output


class EncoderDecoderPredictor(BaseRNNPredictor):
    """
    Predictor using encoder-decoder with RNN cells

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    encoder_hidden_size : int
        The size of hidden layer of the encoder RNN
    decoder_hidden_size : int
        The size of hidden layer of the encoder RNN
    batch_size : int
        The size of each training batch
    seq_len : int
        The length of each sequence
    model_type : {'gru', 'lstm'}
        The type of model
    max_epoch : int
        when to stop training
    learning_rate : float
    target_feature : list of str
        what features to predict, default: ['speed']
    save : bool
        whether to save model
    save_path : str
        where to save the model
    **optimizer_args
        Other keyword arguments to the PyTorch optimizer
    """

    def __init__(self, traffic_data, encoder_hidden_size, decoder_hidden_size, batch_size, seq_len, model_type, max_epoch=5, learning_rate=1e-4,
                 target_feature=None, horizon=1, save=False, save_path='../models/gru/', **optimizer_args):
        super().__init__(traffic_data=traffic_data, hidden_size=None, batch_size=batch_size, seq_len=seq_len, model_type=model_type,
                         max_epoch=max_epoch, learning_rate=learning_rate, target_feature=target_feature, horizon=horizon, save=save,
                         save_path=save_path, **optimizer_args)

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # number of input/output features (int)
        self.input_size = len(self.traffic_data.links) * len(self.traffic_data.features)
        self.output_size = len(self.traffic_data.links) * len(self.target_feature)

        # create model
        self.model = EncoderDecoderModel(self)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "{}-h_enc{}-h_dec{}-s{}-b{}-lr{}".format(self.model_type, self.encoder_hidden_size, self.decoder_hidden_size,
                                                        self.seq_len, self.batch_size, self.learning_rate)

    def train(self):
        return super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)
