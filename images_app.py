from __future__ import division

import flask
from werkzeug import secure_filename
import os
from skimage import io, transform

import tensorflow as tf
import cifar_10

app = flask.Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = 'uploads/'

def create_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
	return

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


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
			prediction = sess.run([top_k_op])

		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)
		return prediction

def predict_label(image):
	 with tf.Graph().as_default():
		# Build a Graph that computes the logits predictions from the
		# inference model.
		input_image = tf.constant(image)

		logits = cifar10.inference(input_image)
		# Calculate predictions.
		top_k_op = tf.nn.top_k(logits, 1)
		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
			cifar_10.MOVING_AVERAGE_DECAY)
		variables_to_restore = {}
		for v in tf.all_variables():
			if v in tf.trainable_variables():
				restore_name = variable_averages.average_name(v)
			else:
				restore_name = v.op.name
			variables_to_restore[restore_name] = v
		saver = tf.train.Saver(variables_to_restore)

		label = eval_once(saver, top_k_op)

		return label


def resize_dims(image):
	height, width, _ = image.shape
	if height > width:
		r = 32 / width
		dim = (32, int(height * r))
	else:
		r = 32 / height
		dim = (int(width * r), 32)

	im = transform.resize(image, dim)
	return image[0:32,0:32]
# Homepage


@app.route("/")
def viz_page():
	"""
	Homepage: serve our visualization page, awesome.html
	"""
	with open("index.html", 'r') as viz_file:
		return viz_file.read()


@app.route('/upload', methods=["POST"])
def classify_image():
	files = flask.request.files['file']
	if files and allowed_file(files.filename):

		filename = secure_filename(files.filename)

		create_directory(app.config['UPLOAD_FOLDER'])

		files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		return flask.redirect(flask.url_for('uploaded_file', filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
	filepath = './uploads/' + filename
	im = io.imread(filepath)
	scaled = resize_dims(im)
	label = predict_label(scaled)
	results = {"label": label}
	os.remove(filepath)
	return flask.jsonify(results)

if __name__ == '__main__':
	app.run(debug=True)