from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.python.platform
import tensorflow as tf

import cifar10_input

#Constants describing features of the CIFAR10 dataset
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
BATCH_SIZE = 128

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
		Args:
			name: name of the variable
			shape: list of ints
			initializer: initializer for Variable
		Returns:
			Variable Tensor
	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var


def _generate_image_and_label_batch(image, label, min_queue_examples):
	"""Construct a queued batch of images and labels.
	Args:
		image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
		label: 1-D Tensor of type.int32
		min_queue_examples: int32, minimum number of samples to retain
			in the queue that provides of batches of examples.
	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
	"""
	# Create a queue that shuffles the examples, and then
	# read 'FLAGS.batch_size' images + labels from the example queue.
	num_preprocess_threads = 16
	images, label_batch = tf.train.shuffle_batch(
		[image, label],
		batch_size=BATCH_SIZE,
		num_threads=num_preprocess_threads,
		capacity=min_queue_examples + 3 * BATCH_SIZE,
		min_after_dequeue=min_queue_examples)
	# Display the training images in the visualizer.
	tf.image_summary('images', images)
	return images, tf.reshape(label_batch, [BATCH_SIZE])


def distorted_inputs():
	"""Construct distorted input for CIFAR training using the Reader ops.
	Raises:
		ValueError: if no data_dir
	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
	"""
	filename = os.path.join('data', 'train.tfrecords')
	# Create a queue that produces the filenames to read.
	filename_queue = tf.train.string_input_producer([filename])

	# Read examples from files in the filename queue.
	image, label = cifar10_input.read_and_decode(filename_queue)
	reshaped_image = tf.cast(image, tf.float32)
	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for training the network. Note the many random
	# distortions applied to the image.
	# Randomly crop a [height, width] section of the image.
	distorted_image = tf.image.random_crop(reshaped_image, [height, width])

	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	# Because these operations are not commutative, consider randomizing
	# randomize the order their operation.
	distorted_image = tf.image.random_brightness(distorted_image,
											   max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image,
											 lower=0.2, upper=1.8)
	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(distorted_image)
	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
						   min_fraction_of_examples_in_queue)

	print ('Filling queue with %d CIFAR images before starting to train. '
		 'This will take a few minutes.' % min_queue_examples)
	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, label,
										 min_queue_examples)


def inputs(eval_data):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
		eval_data: bool, indicating if one should use the train or eval data set.
	Raises:
		ValueError: if no data_dir
	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
	"""
	filename = os.path.join('data', 'test.tfrecords')

	num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	# Create a queue that produces the filenames to read.
	filename_queue = tf.train.string_input_producer([filenames])
	# Read examples from files in the filename queue.
	image, label = cifar10_input.read_and_decode(filename_queue)
	reshaped_image = tf.cast(image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE
	# Image processing for evaluation.
	# Crop the central [height, width] of the image.
	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
														 width, height)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(resized_image)
	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
	# Generate a batch of images and labels by building up a queue of examples.

	return _generate_image_and_label_batch(float_image, label,
										 min_queue_examples)


def conv2d(name, l_input, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], 
													padding='SAME'),b),name=name)


def max_pool(name, l_input, k):
	return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], 
							padding='SAME', name=name)


def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def inference(images):
	#Conv1 Layer
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_on_cpu('weights',[5, 5, 3, 64], 
								tf.truncated_normal_initializer(stddev=1e-4))

		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))

		conv1 = conv2d(scope.name, images, kernel, biases)

	pool1 = max_pool('pool1', conv1, 3)

	norm1 = norm('norm1', pool1)

	#Conv2 Layer
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_on_cpu('weights', [5, 5, 3, 64], 
								tf.truncated_normal_initializer(stddev=1e-4))

		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))

		conv2 = conv2d(scope.name, images, kernel, biases)

	pool2 = max_pool('pool2', conv2, 3)

	norm2 = norm('norm2', pool2)

	# local3
	with tf.variable_scope('local3') as scope:
	# Move everything into depth so we can perform a single matrix multiply.
		dim = 1
		for d in pool2.get_shape()[1:].as_list():
	  		dim *= d
		reshape = tf.reshape(pool2, [BATCH_SIZE, dim])
		weights = _variable_on_cpu('weights', [dim, 384], tf.truncated_normal_initializer(stddev=0.04))
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
	

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_on_cpu('weights', [384,192],
										  tf.truncated_normal_initializer(stddev=0.04))

		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))

		local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)

	# softmax, i.e. softmax(WX + b)
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_on_cpu('weights', [192, NUM_CLASSES],
										  tf.truncated_normal_initializer(stddev=1/192.0))

		biases = _variable_on_cpu('biases', [NUM_CLASSES],
							  tf.constant_initializer(0.0))

		softmax_linear = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

	return softmax_linear

def loss(logits, labels):
	# Reshape the labels into a dense Tensor of
	# shape [batch_size, NUM_CLASSES].
	sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
	indices = tf.reshape(tf.range(0, BATCH_SIZE, 1), [BATCH_SIZE, 1])
	concated = tf.concat(1, [indices, sparse_labels])
	dense_labels = tf.sparse_to_dense(concated, [BATCH_SIZE, NUM_CLASSES], 1.0, 0.0)
	# Calculate the average cross entropy loss across the batch.
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		logits, dense_labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _average_losses(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
	total_loss: Total loss from loss().
  Returns:
	loss_averages_op: op for generating moving averages of losses.
  """
 # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  return loss_averages_op

def train(total_loss, global_step):
	"""Train CIFAR-10 model.
	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.
	Args:
		total_loss: Total loss from loss().
		global_step: Integer Variable counting the number of training steps
		processed.
	Returns:
		train_op: op for training.
	"""
	# Variables that affect learning rate.
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.scalar_summary('learning_rate', lr)
	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _average_losses(total_loss)
	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)
	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
	  MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')
	return train_op 
