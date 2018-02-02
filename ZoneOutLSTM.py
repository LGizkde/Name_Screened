import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


class ZoneoutLSTMCell(RNNCell):
    def __init__(self, input_size, hidden_size, scope="ZoneoutLSTM"):
        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.activation = tf.tanh

        with tf.variable_scope(scope):
            self.linearBias = tf.get_variable(name="ZoneoutLSTM_Bias", shape=[self.hiddenSize * 4],
                                        initializer=tf.constant_initializer(0.0))
            self.linearWeights = tf.get_variable(name="ZoneoutLSTM_Weights",
                                                 shape=[self.inputSize + self.hiddenSize, self.hiddenSize * 4],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.Parameters = [self.linearWeights, self.linearBias]

    def __call__(self, inputs, state, scope="ZoneoutLSTM"):
        h_prev, c_prev = state
        input_pre_h = tf.concat([inputs, h_prev], 1)
        info_gates = tf.matmul(input_pre_h, self.linearWeights) + self.linearBias
        input_gate, new_input, forget_gate, output_gate = tf.split(info_gates, 4, 1)

        binary_mask_cell = self.get_random_mask(tf.shape(c_prev))
        c_temp = c_prev * tf.sigmoid(forget_gate) + tf.sigmoid(input_gate) * self.activation(new_input)
        c = binary_mask_cell * c_prev + (tf.ones(tf.shape(c_prev)) - binary_mask_cell) * c_temp

        binary_mask_output = self.get_random_mask(tf.shape(h_prev))
        h_temp = tf.sigmoid(output_gate) * self.activation(c)
        h = binary_mask_output * h_prev + (tf.ones(tf.shape(h_prev)) - binary_mask_output) * h_temp
        new_state = tf.nn.rnn_cell.LSTMStateTuple(h, c)
        return h, new_state

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.hiddenSize, self.hiddenSize)

    @property
    def output_size(self):
        return self.hiddenSize

    @staticmethod
    def get_random_mask(shape):
      random_tensor_cell = tf.random_uniform(shape, 0.0, 2.0, seed=None)
      binary_mask_cell = tf.floor(random_tensor_cell)
      return binary_mask_cell