import input_data

import tensorflow as tf
import numpy as np

# creates a fully connected layer
def fc_layer (prev_layer, output_length, input_length_=None):

	if not input_length_:
		input_length = prev_layer.get_shape ().as_list () [1]
	else:
		input_length = input_length_

	w_initial = tf.truncated_normal ([input_length, output_length])
	
	fc_w = tf.Variable (w_initial)
	fc_b = tf.Variable (tf.constant ([0.1] * output_length))
	
	return tf.matmul (prev_layer, fc_w) + fc_b

# creates a softmax layer
def softmax_layer (prev_layer):
	return tf.nn.softmax (prev_layer)

# creates a cross entropy layer
def cross_entropy_layer (prev_layer, comparison_logits):
	return -tf.reduce_sum (comparison_logits * tf.log (prev_layer))


# paramaters

batch_size = 200
num_tests = 3000


# build network

mnist = input_data.read_data_sets ("MNIST_data/", one_hot=True)

inputs = tf.placeholder ("float", [None, 28 * 28])

h1 = fc_layer (inputs, 10)

outputs = softmax_layer (h1)

actual = tf.placeholder ("float", [None, 10])

cross_entropy = cross_entropy_layer (outputs, actual)
train_step = tf.train.GradientDescentOptimizer (0.01).minimize (cross_entropy)

is_right = tf.equal (tf.argmax (outputs, 1), tf.argmax (actual, 1))
accuracy = tf.reduce_mean (tf.cast (is_right, "float"))


# train
sess = tf.Session ()
init = tf.initialize_all_variables ()
sess.run (init)

for a in range (num_tests):
	batch_inputs, batch_logits = mnist.train.next_batch (batch_size)

	if a % 100 == 0:
		acc = sess.run (accuracy, feed_dict={
				inputs: mnist.test.images,
				actual: mnist.test.labels,
			})
		print '%s / %s complete. Accuracy=%s' % (a, num_tests, acc)

	feed_dict = {
		inputs: batch_inputs,
		actual: batch_logits,
	}

	sess.run (train_step, feed_dict=feed_dict)

# final evaluation
print 'Final accuracy: %s' % sess.run (accuracy, feed_dict={
		inputs: mnist.test.images,
		actual:	mnist.test.labels,
	})
