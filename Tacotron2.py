import tensorflow as tf
import Config
import Encoder
import Decoder

class Tacotron2:

    def __init__(self, global_step, inputs, input_length, mel_targets, target_lengths, scope = "Tacotron2"):

        self.encoder = Encoder.Encoder(scope + "_Encoder")
        self.decoder = Decoder.Decoder(scope + "_Decoder")
        self.global_step = global_step
        self.inputs = inputs
        self.input_lengths = input_length
        self.mel_targets = mel_targets
        self.target_lengths = target_lengths
        self.loss = tf.placeholder(tf.float32, shape=(None,), name='loss')
        '''
        self.inputs = tf.placeholder(tf.int32, shape=[None, Config.BatchSize], name='inputs')
        self.input_lengths = tf.placeholder(tf.int32, shape=[Config.BatchSize], name='input_lengths')
        self.mel_targets = tf.placeholder(tf.float32, shape=[None,  Config.BatchSize, Config.MelVectorSize], name='mel_targets')
        self.target_lengths = tf.placeholder(tf.int32, shape=[Config.BatchSize], name='target_lengths')
        '''
        self.optimizer = tf.train.AdamOptimizer()

    def buildTacotron2(self):        
        encoderStates, encoderEmb = self.encoder.buildEncoder(self.inputs, self.input_lengths, True)
        self.encoderEmb = encoderEmb
        self.melPreOutput, self.melAfterOutput = self.decoder.buildDecoder(encoderStates, encoderEmb, self.mel_targets,
                                                                 self.input_lengths, self.target_lengths)
    
    def addLoss(self, masks):
        self.before = tf.losses.mean_squared_error(self.mel_targets, self.melPreOutput, weights=masks)
        self.after = tf.losses.mean_squared_error(self.mel_targets, self.melAfterOutput, weights=masks)
        self.loss = self.before + self.after
        self.optim = self.optimizer.minimize(self.loss, global_step=self.global_step)

if __name__ == '__main__':
    trainer = Tacotron2()
    min_loss, before, after = trainer.buildTacotron2()