import tensorflow as tf  # TF 2.0

"""
From https://github.com/thisisiron/spectral_normalization-tf2
"""


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(
            shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_v',
            dtype=tf.float32)

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs, **kwargs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class SpectralNormalization2(layers.Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization2, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

        if not hasattr(self.layer, 'kernel'):
            raise ValueError(
                '`SpectralNormalization` must wrap a layer that'
                ' contains a `kernel` for weights')

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_variable(
            shape=tuple([1, self.w_shape[-1]]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            name='sn_u',
            trainable=False,
            dtype=dtypes.float32)

        super(SpectralNormalization, self).build()

    @def_function.function
    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = K.learning_phase()

        if training == True:
            # Recompute weights for each forward pass
            self._compute_weights()

        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = array_ops.identity(self.u)
        _v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
        _v = _v / math_ops.maximum(math_ops.reduce_sum(_v ** 2) ** 0.5, eps)
        _u = math_ops.matmul(_v, w_reshaped)
        _u = _u / math_ops.maximum(math_ops.reduce_sum(_u ** 2) ** 0.5, eps)

        self.u.assign(_u)
        sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

        self.layer.kernel.assign(self.w / sigma)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
