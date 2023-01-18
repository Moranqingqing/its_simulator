'''
Created on Sep. 14, 2020

@author: user
'''
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import activations


class DenseVariational(Layer):

    def __init__(self, units,
                 activation=None,
                 mean_initializer='glorot_uniform',
                 variance_initializer=tf.initializers.RandomUniform(-10., -5.),
                 kl_weight=1e-3,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.mean_initializer = tf.initializers.get(mean_initializer)
        self.variance_initializer = variance_initializer
        self.kl_weight = kl_weight
        self._activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=self.mean_initializer)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=self.mean_initializer)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=self.variance_initializer)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=self.variance_initializer)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))
        return self._activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        gaussian_mix = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=[self.prior_pi_1, self.prior_pi_2]
            ),
            components_distribution=tfp.distributions.Normal(
                loc=[0.0, 0.0],
                scale=[self.prior_sigma_1, self.prior_sigma_2]
            )
        )
        return gaussian_mix.log_prob(w)

