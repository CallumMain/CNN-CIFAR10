"""Functions for downloading and reading cifar-10 data."""
import os
import tensorflow.python.platform
import tensorflow as tf

IMAGE_PIXELS = 32*32

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		dense_keys=['image_raw', 'label'],
		# Defaults are not specified since both keys are required.
		dense_types=[tf.string, tf.int64])

	# Convert from a scalar string tensor (whose single string has
	# length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
	# [mnist.IMAGE_PIXELS].
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image.set_shape([IMAGE_PIXELS])

	# OPTIONAL: Could reshape into a 28x28 image and apply distortions
	# here.  Since we are not applying any distortions in this
	# example, and the next step expects the image to be flattened
	# into a vector, we don't bother.

	# Convert from [0, 255] -> [-0.5, 0.5] floats.
	img = tf.reshape(image, [32,32,3])

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	label = tf.cast(features['label'], tf.int32)

	return img, label