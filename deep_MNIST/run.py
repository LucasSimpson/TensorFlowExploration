import input_data
import tensorflow as tf

def weight_variable (shape):
	# some sort of distribution for starting values
	initial = tf.truncated_normal (shape, stddev=0.1)
	return tf.Variable (initial)

def bias_variable (shape):
	# constant not at 0 to avoid it dropping out or being ignored
	initial = tf.constant (0.1, shape=shape)
	return tf.Variable (initial)
	
def conv2d (x, W):
	# x is input, w is in this case shape
	return tf.nn.conv2d (x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2 (x):
	# 2x2 max pooling, so strides are 2x2 to avoid overlap
	return tf.nn.max_pool (x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets ("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession ()



x = tf.placeholder ("float", shape=[None, 784]) # input image data
y_ = tf.placeholder ("float", shape=[None, 10]) # input correct result label

# reshaping the single row vector int 28x28 grid
x_image = tf.reshape (x, [-1, 28, 28, 1]) 


# ********** FIRST LAYER **********

# x, y, channels, depth. so 32 kernels here, each 5x5x1 conv area
W_conv1 = weight_variable ([5, 5, 1, 32]) 

# 32 kernels so 32 biases
b_conv1 = bias_variable ([32]) 

# define convultion -> adding bias -> RElu 
h_conv1 = tf.nn.relu (conv2d (x_image, W_conv1) + b_conv1)

# define into max pool
h_pool1 = max_pool_2x2 (h_conv1)


# *********** SECOND LAYER **********

# 5x5x32 convolution, 64 kernels. 32 depth comes from prev 32 kernels
W_conv2 = weight_variable ([5, 5, 32, 64])

# 64 kernels this time, so again 64 biases
b_conv2 = bias_variable ([64])

# convolution -> adding bias -> Relu
h_conv2 = tf.nn.relu (conv2d (h_pool1, W_conv2) + b_conv2)

# max pool over convolution
h_pool2 = max_pool_2x2 (h_conv2)


# ********** DENSELY CONNECTED LAYER ********

# 7 * 7 * 64 input values, 1024 exit neurons
W_fc1 = weight_variable ([7 * 7 * 64, 1024])

# 1024 neurons, so 1024 biases
b_fc1 = bias_variable ([1024])

# reshape the result of pooling layer into single vector
h_pool2_flat = tf.reshape (h_pool2, [-1, 7 * 7 * 64])

# define pooling layer -> 1024 fully connected layer -> 1024 biases -> RElu
h_fc1 = tf.nn.relu (tf.matmul (h_pool2_flat, W_fc1) + b_fc1)


# ************ DROPOUT LAYER ***************

# define keep probability as placeholder to enable/disable for testing/using
keep_prob = tf.placeholder ("float")

# connect result -> dropout layer
h_fc1_drop = tf.nn.dropout (h_fc1, keep_prob)


# ************ READOUT LAYER **************

# 1024 input neurons, 10 output probalistic vector
W_fc2 = weight_variable ([1024, 10])

# final 10 biases
b_fc2 = bias_variable ([10])

# softmax probability final readout
y_conv = tf.nn.softmax (tf.matmul (h_fc1_drop, W_fc2) + b_fc2)


# ************** TRAINING *****************

# compute difference between ideal and actual
cross_entropy = -tf.reduce_sum (y_ * tf.log (y_conv))

# define a training step with adam optimizer. Minimize cross_entropy
train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)

# compare max experimental vs max actual
correct_prediction = tf.equal (tf.argmax (y_conv, 1), tf.argmax (y_, 1))

# calculate percentage accuracy
accuracy = tf.reduce_mean (tf.cast (correct_prediction, "float"))

# initialize all variables to starting values
sess.run (tf.initialize_all_variables ())

# 20000 iterations
for a in range (2000):

	# grab 50 pictures as 50x784 array
	batch = mnist.train.next_batch (50)

	# every 100 iteration evaluate how our model is doing. note 1.0 keep_prob
	if a % 100 == 0:
		train_accuracy = accuracy.eval (feed_dict={
				x: batch [0],
				y_: batch [1],
				keep_prob: 1.0,
			})
		print "step %d, training accuracy %g" % (a, train_accuracy)

	# do training step
	train_step.run (feed_dict={
			x: batch [0],
			y_: batch [1],
			keep_prob: 0.5,
		})

# final output (~99.2%)ls

print "Test accuracy %g" % accuracy.eval (feed_dict={
											x: batch [0],
											y_: batch [1],
											keep_prob: 1.0,
										})