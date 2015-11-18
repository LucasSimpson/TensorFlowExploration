import tensorflow as tf

import numpy as np

# creates a fully connected layer
def fc_layer (prev_layer, output_length):
	input_length = prev_layer.get_shape ().as_list () [1]

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


# define parameters

test_vec_length = 300
num_classes = 300
batch_size = 1
test_casses = 1000
h1_layer_size = num_classes


# create some bogus data

classes = []
for a in range (num_classes):
	data = np.random.random ([test_vec_length])
	classes += [[data]]

correct_outputs = []
for a in range (num_classes):
	data = [0.0 for a in range (num_classes)]
	data [a] = 1.0
	correct_outputs += [[data]]


# test inputs

test_inputs = [class_ [0] for class_ in classes]
test_outputs = [correct_output [0] for correct_output in correct_outputs]


# nn structure

inputs = tf.placeholder (tf.float32, shape=(None, test_vec_length))

outputs = softmax_layer (fc_layer (inputs, h1_layer_size))


# train structure

output_comparison = tf.placeholder (tf.float32, shape=(None, num_classes))

cross_entropy = cross_entropy_layer (outputs, output_comparison)

train_step = tf.train.GradientDescentOptimizer (0.01).minimize (cross_entropy)


# accuracy structure

predicted_index = tf.argmax (outputs, 1)
correct_index = tf.argmax (output_comparison, 1)

is_right = tf.equal (predicted_index, correct_index)

accuracy = tf.reduce_mean (tf.cast (is_right, "float"))


# perform training

print 'Training to reconize %s different vectors each of %s length with %s test casses' % (num_classes, test_vec_length, test_casses)

init = tf.initialize_all_variables ()
sess = tf.Session ()
sess.run (init)

for a in range (test_casses):
	if a % 100 == 0:
		acc = sess.run (accuracy, feed_dict={inputs: test_inputs, output_comparison: test_outputs})
		print '%s / %s complete. Accuracy=%s' % (a, test_casses, acc)

	index_ = np.random.randint (0, num_classes)

	feed_dict = {
		inputs: classes [index_],
		output_comparison: correct_outputs [index_],
	}

	sess.run (train_step, feed_dict=feed_dict)







