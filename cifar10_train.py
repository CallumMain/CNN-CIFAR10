from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import os.path
import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
import cifar_10

MAX_STEPS = 100000
BATCH_SIZE = 128

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        # Get images and labels for CIFAR-10.
        images, labels = cifar_10.distorted_inputs()
        # Build a Graph that computes the logits predictions from the

        # inference model.
        logits = cifar_10.inference(images)
        # Calculate loss.
        loss = cifar_10.loss(logits, labels)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar_10.train(loss, global_step)
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_path = os.path.join('model', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    train()

if __name__ == '__main__':
    main()
