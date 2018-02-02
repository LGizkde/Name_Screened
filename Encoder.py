import tensorflow as tf
import Config
import ZoneOutLSTM


class Encoder:

    def __init__(self, scope = "TacotronEncoder"):
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.EncoderL2R = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.EncoderHiddenSize,
                                                          scope=self.scope + "_EncoderL2R")
            self.EncoderR2L = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.EncoderHiddenSize,
                                                          scope=self.scope + "_EncoderR2L")
            self.wordEmbedding = tf.get_variable("wordEmbedding", [Config.VocabSize, Config.EmbeddingSize],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.preNetFilters = tf.get_variable("preNetFilters",  [Config.EncoderPreNetConvKernelSize, Config.EmbeddingSize,
                                                   Config.EncoderPreNetConvFilterSize * Config.EncoderPreNetStackSize],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.DropoutRate = Config.EncoderDropoutRate
        self.Parameters = [self.wordEmbedding, self.preNetFilters]
        self.Parameters.extend(self.EncoderL2R.Parameters)
        self.Parameters.extend(self.EncoderR2L.Parameters)

    def buildEncoderStates(self, inputs, inputLength):
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(self.EncoderL2R, self.EncoderR2L, inputs, dtype=tf.float32,
                                                               sequence_length=inputLength, time_major=True)
        outputs = tf.concat(bi_outputs, -1)
        return outputs, bi_state[1][0]


    def prenet(self, inputs, isTraining):
        prenet_filters = tf.split(self.preNetFilters, Config.EncoderPreNetStackSize, -1)

        for i in range(0, Config.EncoderPreNetStackSize, 1):
            with tf.variable_scope('encoder_prenet_layer_{0}'.format(i + 1)):
                conv_output = tf.nn.conv1d(inputs, prenet_filters[i], stride=1, padding="SAME")
                batch_norm = tf.layers.batch_normalization(conv_output, training=isTraining)
                relu_ouput = tf.nn.relu(batch_norm)
                output = tf.nn.dropout(relu_ouput, self.DropoutRate)
                inputs = output

        return output


    def buildEncoder(self, inputs, inputLength, isTraining=True):
        embedded_inputs = tf.nn.embedding_lookup(self.wordEmbedding, inputs)
        self.embedded_inputs = embedded_inputs
        enc_conv_outputs = self.prenet(embedded_inputs, isTraining)
        self.enc_conv_outputs = enc_conv_outputs
        outputs, bw_state = self.buildEncoderStates(enc_conv_outputs, inputLength)
        self.bw_state = bw_state
        self.outputs = outputs
        return outputs, bw_state