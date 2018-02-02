import tensorflow as tf
import Config
import ZoneOutLSTM
import Attention

class Decoder:

    def __init__(self, scope = "TacotronDecoder"):

        self.scope = scope
        self.attentionSize = Config.EncoderHiddenSize*2
        with tf.variable_scope(self.scope):

            self.preNetWeight0 = tf.get_variable("DecoderPrenetWeight0", [Config.NumMels, Config.DecoderPreNetHiddenSize],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.preNetBias0 = tf.get_variable("DecoderPrenetBias0", [Config.DecoderPreNetHiddenSize],
                                              initializer=tf.constant_initializer(0.0))
            self.preNetWeight1 = tf.get_variable("DecoderPrenetWeight1", [Config.DecoderPreNetHiddenSize, Config.DecoderPreNetHiddenSize],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.preNetBias1 = tf.get_variable("DecoderPrenetBias1", [Config.DecoderPreNetHiddenSize],
                                    initializer=tf.constant_initializer(0.0))

            self.linearWeights = tf.get_variable("DecoderLinearWeight", [Config.DecoderHiddenSize + self.attentionSize , Config.MelVectorSize],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.linearBias = tf.get_variable("DecoderLinearBias", [Config.MelVectorSize], initializer=tf.constant_initializer(0.0))

            self.postNetLinearWeight = tf.get_variable("PostNetLinearWeight", [Config.MelVectorSize, Config.DecoderPostNetConvFilterSize],
                                                       initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.postNetFilters = tf.get_variable("postNetFilters", [Config.DecoderPostNetConvKernelSize, Config.DecoderPostNetConvFilterSize,
                                                           Config.DecoderPostNetConvFilterSize * Config.DecoderPostNetStackSize],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.postLinearWeights = tf.get_variable("DecoderPostNetLinearWeight", [Config.DecoderPostNetConvFilterSize, Config.MelVectorSize],
                                                     initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.postLinearBias = tf.get_variable("DecoderPostNetLinearBias", [Config.MelVectorSize],
                                    initializer=tf.constant_initializer(0.0))
            self.WI = tf.get_variable("DecoderInitWeight", shape=(Config.EncoderHiddenSize, Config.DecoderHiddenSize * 2* Config.DecoderRnnStackSize),
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.WIb = tf.get_variable("DecoderInitBias", shape=Config.DecoderHiddenSize * 2 * Config.DecoderRnnStackSize,
                                       initializer=tf.constant_initializer(0.0))


        self.dropoutRate = Config.DecoderDropoutRate

    def prenet(self, inputs):
        inputs = tf.reshape(inputs, shape=[-1, Config.NumMels])
        linear_output = tf.matmul(inputs, self.preNetWeight0) + self.preNetBias0
        output = tf.nn.dropout(linear_output, self.dropoutRate)
        linear_output = tf.matmul(output, self.preNetWeight1) + self.preNetBias1
        output = tf.nn.dropout(linear_output, self.dropoutRate)
        output = tf.reshape(output, shape=[-1, Config.BatchSize, Config.DecoderPreNetHiddenSize])
        return output

    def buildAttentionCell(self, encoderOutputs, sourceSequenceLength):
        memory = tf.transpose(encoderOutputs, [1, 0, 2])
        with tf.variable_scope(self.scope):
            attention_mechanism = Attention.MyAttention(num_units=Config.AttentionSize, memory=memory,
                                                            memory_sequence_length=sourceSequenceLength, scope=self.scope + "_Attention")
            cells = []

            cells.append(ZoneOutLSTM.ZoneoutLSTMCell(input_size=self.attentionSize + Config.DecoderPreNetHiddenSize,
                                                     hidden_size=Config.DecoderHiddenSize, scope=self.scope + "DecoderLSTM0"))
            for i in range(1, Config.DecoderRnnStackSize, 1):
                cell=ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.DecoderHiddenSize,
                                                     hidden_size=Config.DecoderHiddenSize,scope=self.scope + 'DecoderLSTM{0}'.format(i))
                cells.append(cell)
            stacked_Cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            attn_cell = Attention.MyAttentionWrapper(stacked_Cell, attention_mechanism, attention_layer_size=self.attentionSize)

        return attn_cell

    def buildDecoderInitState(self, srcSentEmb):
        initTuple = ()
        initWeights = tf.split(self.WI, Config.DecoderRnnStackSize, -1)
        initBiases = tf.split(self.WIb, Config.DecoderRnnStackSize, -1)
        for i in range(0, Config.DecoderRnnStackSize, 1):
            WIS = tf.matmul(srcSentEmb,initWeights[i]) + initBiases[i]
            initHiddenMem = tf.tanh(WIS)
            (initHiddden, initMem) = tf.split(initHiddenMem, 2, -1)
            initTuple = initTuple + (tf.nn.rnn_cell.LSTMStateTuple(initMem, initHiddden),)
        return initTuple

    def buildDecoderStates(self, srcrHiddens, srcSentEmb, decoderLSTMInput, srcLength, trgLength):
        decoderInitState = self.buildDecoderInitState(srcSentEmb)
        decoderCell = self.buildAttentionCell(srcrHiddens, srcLength)
        #decoderInitState = decoderCell.zero_state(Config.BatchSize)
        decoderInitState = decoderCell.init_state(Config.BatchSize, decoderInitState)
        helper = tf.contrib.seq2seq.TrainingHelper(decoderLSTMInput, trgLength, time_major=True)
        my_decoder = tf.contrib.seq2seq.BasicDecoder(decoderCell, helper, decoderInitState)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=True,
                                                                            swap_memory=True, scope=self.scope)
        return outputs, final_context_state

    def postnet(self, inputs, is_training):
        postnet_filters = tf.split(self.postNetFilters, Config.DecoderPostNetStackSize, -1)

        for i in range(0, Config.DecoderPostNetStackSize, 1):
            with tf.variable_scope('decoder_postnet_layer_{0}'.format(i + 1)):
                conv_output = tf.nn.conv1d(inputs, postnet_filters[i], stride=1, padding="SAME")
                batch_norm = tf.layers.batch_normalization(conv_output, training=is_training)
                relu_ouput = tf.nn.relu(batch_norm)
                output = tf.nn.dropout(relu_ouput, self.dropoutRate)
                inputs = output

        return output

    def buildDecoder(self, srcrHiddens, srcSentEmb, outputs, srcLength, trgLength):
        prenet_output = self.prenet(outputs)
        lstm_outputs, final_state = self.buildDecoderStates(srcrHiddens, srcSentEmb, prenet_output, srcLength, trgLength)
        linear_input = lstm_outputs.rnn_output
        attentions = tf.reshape(final_state.attention_history.concat(), shape=[-1, Config.BatchSize, self.attentionSize])
        linear_input = tf.concat([attentions, linear_input], axis=-1)
        linear_input = tf.reshape(linear_input, shape=[-1, Config.DecoderHiddenSize + self.attentionSize])
        mel_pre_output = tf.matmul(linear_input, self.linearWeights) + self.linearBias
        mel_postnet_input = tf.matmul(mel_pre_output, self.postNetLinearWeight)
        mel_pre_output = tf.reshape(mel_pre_output, shape=[-1, Config.BatchSize, Config.MelVectorSize])
        mel_postnet_input = tf.reshape(mel_postnet_input, shape=[-1, Config.BatchSize, Config.DecoderPostNetConvFilterSize])
        postnet_output = self.postnet(mel_postnet_input, is_training=True)
        postnet_output = tf.reshape(postnet_output, shape=[-1, Config.DecoderPostNetConvFilterSize])
        mel_after_output = tf.matmul(postnet_output, self.postLinearWeights) + self.postLinearBias + tf.reshape(mel_pre_output, shape=[-1, Config.MelVectorSize])
        mel_after_output = tf.reshape(mel_after_output, shape=[-1, Config.BatchSize, Config.MelVectorSize])
        return (mel_pre_output, mel_after_output)