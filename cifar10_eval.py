from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10

eval_dir = '/eval/'
checkpoint_dir = '/model/'
eval_interval_secs = 60*5
num_examples = 10000
run_once = False
batch_size = 128 


def eval_once(saver, top_k_op):
	"""Run Eval once.
	Args:
		saver: Saver.
		top_k_op: Top K op.
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return
		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
																				 start=True))
			num_iter = int(math.ceil(num_examples / batch_size))
			true_count = 0  # Counts the number of correct predictions.
			total_sample_count = num_iter * batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1
			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)


def evaluate():
	"""Eval CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		# Get images and labels for CIFAR-10.
		eval_data = eval_data == 'test'
		images, labels = cifar10.inputs(eval_data=eval_data)
		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = cifar10.inference(images)
		# Calculate predictions.
		top_k_op = tf.nn.in_top_k(logits, labels, 1)
		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
				cifar10.MOVING_AVERAGE_DECAY)
		variables_to_restore = {}
		for v in tf.all_variables():
			if v in tf.trainable_variables():
				restore_name = variable_averages.average_name(v)
			else:
				restore_name = v.op.name
			variables_to_restore[restore_name] = v
		saver = tf.train.Saver(variables_to_restore)
		while True:
			eval_once(saver, top_k_op)
			if run_once:
				break
			time.sleep(eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
	evaluate()

if __name__ == '__main__':
	main()