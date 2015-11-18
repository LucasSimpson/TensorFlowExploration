import input_data

import tensorflow as tf
import numpy as np

# shortcut to make a weight variable with truncated normal distribution
def weight_variable (shape):
	initial = tf.truncated_normal (shape, stddev=0.1)
	return tf.Variable (initial)

# shortcut for making bias variables with 0.1 starting constant
def bias_variable (shape):
	initial = tf.constant (0.1, shape=shape)
	return tf.Variable (initial)

# creates a fully connected layer
def fc_layer (prev_layer, output_length, input_length_=None):
	if not input_length_:
		input_length = prev_layer.get_shape ().as_list () [1]
	else:
		input_length = input_length_
	
	fc_w = weight_variable ([input_length, output_length])
	fc_b = bias_variable ([output_length])
	
	return tf.matmul (prev_layer, fc_w) + fc_b

# creates a softmax layer
def softmax_layer (prev_layer):
	return tf.nn.softmax (prev_layer)

# creates a cross entropy layer
def cross_entropy_layer (prev_layer, comparison_logits):
	return -tf.reduce_sum (comparison_logits * tf.log (prev_layer))

# creates a convolution layer
# PADDING: SAME -> only full sized windows vs VALID -> partial windows allowed
def conv2d_layer (prev_layer, kernel_width, kernel_height, stride_x, stride_y, num_kernels, padding='SAME'):
	z_depth = prev_layer.get_shape ().as_list () [3]
	filter_ = weight_variable ([kernel_height, kernel_width, z_depth, num_kernels])

	return tf.nn.conv2d (
		input = prev_layer, 
		filter = filter_, 
		strides = [1, stride_x, stride_y, 1],
		padding = padding,
	) + bias_variable ([num_kernels])

# creates a max pooling layer
# PADDING: SAME -> only full sized windows vs VALID -> partial windows allowed
def max_pool_layer (prev_layer, pool_width, pool_height, stride_x, stride_y, padding='SAME'):
	return tf.nn.max_pool (
		value = prev_layer,
		ksize = [1, pool_width, pool_height, 1],
		strides = [1, stride_x, stride_y, 1],
		padding = padding,
	)

# creates a rectified linear layer
def relu_layer (prev_layer):
	return tf.nn.relu (prev_layer)

# creates a dropout layer
def dropout_layer (prev_layer, keep_prob):
	return tf.nn.dropout (prev_layer, keep_prob)

# some params
num_tests = 400
batch_size = 50

# start interactive session
sess = tf.InteractiveSession ()

# define data source
mnist = input_data.read_data_sets ("/tmp/MNIST_data/", one_hot=True)

# define variables
inputs = tf.placeholder ("float", shape=[None, 28 * 28])
actual = tf.placeholder ("float", shape=[None, 10])
keep_prob = tf.placeholder ("float")

# reshape into grid for convolution
grid_inputs = tf.reshape (inputs, [-1, 28, 28, 1])

# first layer
conv1 = conv2d_layer (grid_inputs, 5, 5, 1, 1, 32)
print conv1
relu1 = relu_layer (conv1)
mxp1 = max_pool_layer (relu1, 2, 2, 2, 2)

# second layer
conv2 = conv2d_layer (mxp1, 5, 5, 1, 1, 64)
relu2 = relu_layer (conv2)
mxp2 = max_pool_layer (relu2, 2, 2, 2, 2)

# flatten
mxp2_flat = tf.reshape (mxp2, [-1, 7 * 7 * 64])

# fully connected layer
fc = fc_layer (mxp2_flat, 1024)

# dropout for training
dropout = dropout_layer (fc, keep_prob)

# get results
outputs = softmax_layer (fc_layer (dropout, 10, 1024))

# training
cross_entropy = cross_entropy_layer (outputs, actual)
train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)

# get accuracy
is_right = tf.equal (tf.argmax (outputs, 1), tf.argmax (actual, 1))
accuracy = tf.reduce_mean (tf.cast (is_right, "float"))

# loggin and visualization
accuracy_sum = tf.scalar_summary ("Accuracy", accuracy)
cross_entropy_hist = tf.scalar_summary ("Cross Entropy", cross_entropy)
conv1_k1 = tf.image_summary ("Conv1_k1", tf.split (3, 32, conv1) [0])

merged_summary_op = tf.merge_all_summaries ()
writer = tf.train.SummaryWriter ("./logs", accuracy.graph.as_graph_def ())

# train setup
init = tf.initialize_all_variables ()
sess.run (init)

# training
for a in range (num_tests + 1):
	batch = mnist.train.next_batch (batch_size)

	if a % 25 == 0:
		summary, acc = sess.run ([merged_summary_op, accuracy], feed_dict={
				inputs: batch [0],
				actual: batch [1],
				keep_prob: 1.0,
			})
		print '%s / %s complete. Accuracy=%s' % (a, num_tests, acc)
		writer.add_summary (summary, a)

	feed_dict = {
		inputs: batch [0],
		actual: batch [1],
		keep_prob: 0.5,
	}

	sess.run (train_step, feed_dict=feed_dict)


# final evaluation
batch = mnist.train.next_batch (400)
print 'Final accuracy: %s' % sess.run (accuracy, feed_dict={
		inputs: batch [0],
		actual:	batch [1],
		keep_prob: 1.0,
	})


