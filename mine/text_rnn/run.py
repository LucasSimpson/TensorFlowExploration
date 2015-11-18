import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


class RNN_Model (object):
	def __init__ (self):
		# ******* PARAMS *********

		# total vocabulary size
		vocab_size = 50

		# one character at a time
		lstm_size = 2

		# will feed 50 chars sequentially
		num_steps = 40

		# only 1 batch for simplicity
		batch_size = 3

		# define is training
		is_training = True


		# ********* SET UP *********

		# make the lstm cell, with size lstm_size
		self.lstm_cell = rnn_cell.BasicLSTMCell (lstm_size, forget_bias=0.0)

		# if in training mode, add a dropout layer
		if is_training:
			self.lstm_cell = rnn_cell.DropoutWrapper (self.lstm_cell, output_keep_prob=0.5)

		# set initial state to zeroes
		self.initial_state = self.lstm_cell.zero_state(batch_size, tf.float32)

		# define inputs. has size batch_size X num_steps
		self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

		# define the embedding tensor
		initial = tf.truncated_normal ([batch_size, lstm_size], stddev=0.1)
		self.embedding = tf.Variable (initial)

		# get the inputs from embedded data
		self.inputs = tf.split (1, num_steps, tf.nn.embedding_lookup (self.embedding, self.input_data))
		self.inputs = [tf.squeeze (input_, [1]) for input_ in self.inputs]

		# define outputs
		self.outputs, self.states = rnn.rnn(self.lstm_cell, self.inputs, initial_state=self.initial_state)

		print self.outputs [0]
		print self.states [0]

		# reshape input into [batch_size * num_steps, lstm_size]
		output = tf.reshape(tf.concat(1, self.outputs), [-1, lstm_size])

		print output

		# ********* TRAINING **********

		# quit if not training
		if not is_training:
			return


		



rnn = RNN_Model ()