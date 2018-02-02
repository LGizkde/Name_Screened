import numpy as np 
import pandas as pd
from datetime import datetime
import os
import subprocess
import time
import tensorflow as tf 
import traceback
import Config
import Tacotron2

from DataLoader import Feeder
from utils.text import sequence_to_text
from utils import ValueWindow

def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')

def train():
	checkpoint_path = os.path.join(Config.LogDir, 'model.ckpt')
	save_dir = os.path.join(Config.LogDir, 'pretrained/')
	input_path = Config.DataDir

	#Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path)

	#Set up model:
	step_count = 0
	try:
		#simple text file to keep count of global step
		with open('step_counter.txt', 'r') as file:
			step_count = int(file.read())
	except:
		print('no step_counter file found, assuming there is no saved checkpoint')

	global_step = tf.Variable(step_count, name='global_step', trainable=False)

	model = Tacotron2.Tacotron2(global_step, feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.target_lengths)
	model.buildTacotron2()
	model.addLoss(feeder.masks)

	#Book keeping
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	#Train
	with tf.Session() as sess:
		try:
			sess.run(tf.global_variables_initializer())

			#saver.restore(sess, checkpoint_state.model_checkpoint_path)

			#initiating feeder
			feeder.start_in_session(sess)

			#Training loop
			while not coord.should_stop():
				start_time = time.time()
				step, loss, _ = sess.run([model.global_step, model.loss, model.optim])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				if step % 1 == 0:
					message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
						step, time_window.average, loss, loss_window.average)
					print (message)
				'''
				if loss > 100 or np.isnan(loss):
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % Config.CheckpointInterval == 0:
					with open('step_counter.txt', 'w') as file:
						file.write(str(step))
					log('Saving checkpoint to: {}-{}'.format(checkpoint_path, step))
					saver.save(sess, checkpoint_path, global_step=step)
					# Unlike the original tacotron, we won't save audio
					# because we yet have to use wavenet as vocoder
					log('Saving alignement..')
					input_seq, prediction, alignment = sess.run([model.inputs[0],
																 model.mel_outputs[0],
																 model.alignments[0],
																 ])
					#save predicted spectrogram to disk (for plot and manual evaluation purposes)
					mel_filename = 'ljspeech-mel-prediction-step-{}.npy'.format(step)
					np.save(os.path.join(log_dir, mel_filename), prediction.T, allow_pickle=False)

					#save alignment plot to disk (evaluation purposes)
					plot.plot_alignment(alignment, os.path.join(log_dir, 'step-{}-align.png'.format(step)),
						info='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss))
					log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))
				'''

		except Exception as e:
			#log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = Config.GPUIndex
	train()

if __name__ == '__main__':
	main()
