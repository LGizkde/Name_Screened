import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, \
    AttentionWrapperState, _compute_attention
import Config

class MyAttention(_BaseAttentionMechanism):

    def __init__(self, num_units, memory, memory_sequence_length=None, scope="MyAttention"):
        self._name = scope + "_MyAttention"
        self._num_units = num_units

        with tf.variable_scope(scope):
            query_layer = tf.layers.Dense(num_units, name="query_layer", use_bias=False)
            memory_layer = tf.layers.Dense(num_units, name="memory_layer", use_bias=False)
            self.v = tf.get_variable("attention_v", [num_units], dtype=tf.float32)
            self.pre_location_filter = tf.get_variable("pre_location_filter",
                                                       [Config.AttentionConvKernelSize, 1, Config.AttentionConvFilterSize],
                                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.pre_Location_weight = tf.get_variable("prelocation_w",
                                                       [Config.AttentionConvFilterSize, 128],
                                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
        wrapped_probability_fn = lambda score, _: tf.nn.softmax(score)

        super(MyAttention, self).__init__(
            query_layer=query_layer,
            memory_layer=memory_layer,
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=float("-inf"),
            name=self._name)

    def __call__(self, query, previous_alignments):
        previous_alignments = tf.expand_dims(previous_alignments, -1)
        conv_output = tf.nn.conv1d(previous_alignments, self.pre_location_filter, stride=1, padding="SAME")
        conv_output = tf.reshape(conv_output, shape=[-1, 32])
        linear_output = tf.matmul(conv_output, self.pre_Location_weight)
        linear_output = tf.reshape(linear_output, shape=[Config.BatchSize, -1, 128])
        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)
        score = tf.reduce_sum(self.v * tf.tanh(self._keys + processed_query + linear_output), [2])
        alignments = self._probability_fn(score, previous_alignments)
        return alignments


class MyAttentionWrapperState(
    namedtuple("MyAttentionWrapperState",
                           ("cell_state", "attention", "attention_history", "time", "alignments",
                            "alignment_history"))):

  def clone(self, **kwargs):
    return super(MyAttentionWrapperState, self)._replace(**kwargs)

class MyAttentionWrapper(RNNCell):
    def __init__(self, cell, attention_mechanism,  attention_layer_size=None, alignment_history=True, scope="MyAttentionWrapper"):

        super(MyAttentionWrapper, self).__init__(name=scope+"_AttentionWrapper")
        with tf.variable_scope(scope):
            self.attention_layer = tf.layers.Dense(attention_layer_size, name="attention_layer", use_bias=False)

        self.attention_layer_size = attention_layer_size
        self.cell = cell
        self.attention_mechanism = attention_mechanism
        self.cell_input_fn = (lambda inputs, attention: tf.concat([inputs, attention], -1))
        self.use_alignment_history = alignment_history

    @property
    def output_size(self):
        return self.attention_layer_size

    @property
    def state_size(self):
        return MyAttentionWrapperState(
            cell_state=self.cell.state_size,
            time=tf.TensorShape([]),
            attention=self.attention_layer_size,
            attention_history=(),
            alignments=self.attention_mechanism.alignments_size,
            alignment_history=())  # sometimes a TensorArray


    def zero_state(self, batch_size):
        cell_state = self.cell.zero_state(batch_size, tf.float32)
        return self.init_state(batch_size, cell_state)

    def init_state(self, batch_size, init_cell_state):
        cell_state = tf.contrib.framework.nest.map_structure(lambda s: tf.identity(s, name="checked_cell_state"), init_cell_state)

        return MyAttentionWrapperState(
            cell_state=cell_state,
            time=tf.zeros([], dtype=tf.int32),
            attention=tf.zeros(shape=[batch_size, self.attention_layer_size], dtype=tf.float32),
            attention_history=tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
            alignments=self.attention_mechanism.initial_alignments(batch_size, tf.float32),
            alignment_history=tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))

    def call(self, inputs, state):
        cell_inputs = self.cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self.cell(cell_inputs, cell_state)
        cell_output = tf.identity(cell_output, name="checked_cell_output")

        previous_alignment = state.alignments
        previous_alignment_history = state.alignment_history
        previous_attention_history = state.attention_history

        attention_mechanism = self.attention_mechanism
        attention, alignments = _compute_attention(attention_mechanism, cell_output, previous_alignment, self.attention_layer)
        alignment_history = previous_alignment_history.write(state.time, alignments) if self.use_alignment_history else ()
        attention_history = previous_attention_history.write(state.time, attention)

        next_state = MyAttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_history=attention_history,
            alignments=alignments,
            alignment_history=alignment_history)

        return cell_output, next_state

