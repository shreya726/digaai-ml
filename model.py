#!/usr/bin/env python

import csv
import json

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.metrics as metrics
from tensorflow.contrib import lookup
from tensorflow.python import debug as tf_debug

from mapping import mapping as MAPPING

with open('grams.json') as data:
	GRAMS = json.load(data)

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_STEPS = 20

BATCH_SIZE = 1000

MAX_NAME_LENGTH = 10
PADDING = 'ZYXW'

CSV_COLUMNS = ['first1','first2','first3','first4','first5','first6','first7','first8','first9','first10','last1', 'last2', 'last3', 'last4', 'last5', 'last6', 'last7', 'last8', 'last9', 'last10','source']
LABEL_COLUMN = 'source'

# 1 for brazilian and 0 for non-brazilian 
CLASSES = ['1', '0']  

def read_dataset(mode):
	filename = "./data/"	#PATH OF FILE"


	if mode == "train":
		mode = tf.contrib.learn.ModeKeys.TRAIN
		filename += "training_processed.csv"
	else:
		mode = tf.contrib.learn.ModeKeys.EVAL
		filename += "eval_processed.csv" # FIXME for testing?

	def _input_fn():   # gets passed to tensorflow
		
		# Get file names
		input_file_names = tf.train.match_filenames_once(filename)
		filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)
		
		# load the data with given batch size (constant above)
		reader = tf.TextLineReader(skip_header_lines=False)
		_, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
		value_column = tf.expand_dims(value, -1)

		# Read CSV files
		columns = tf.decode_csv(value_column, record_defaults=[['first1'],['first2'],['first3'],['first4'],['first5'],['first6'],['first7'],['first8'],['first9'],['first10'],['last1'], ['last2'], ['last3'], ['last4'], ['last5'], ['last6'], ['last7'], ['last8'], ['last9'], ['last10'],['source']], field_delim=',')

		tf.logging.info('after decode_csv')

		# Converting features and labels to numbers (decoded from CSV as string)
		features = tf.string_to_number(columns[0:20], tf.float32)
		label = tf.string_to_number(columns[20:21], tf.float32)

		# Reshaping features and labels
		features = tf.reshape(features,shape=[-1,20])
		label = tf.reshape(label, shape=[-1,1])
	
		return features, label
	
	return _input_fn

def cnn_model(features, target, mode):
	tf.logging.info('inside cnn_model')
	first_name = features

	logits = tf.contrib.layers.fully_connected(first_name, num_outputs=1, activation_fn=None)

	predictions_dict = {
			'ethnicity' : tf.gather(CLASSES, tf.argmax(logits, 1)),
			'class' : tf.argmax(logits, 1),
			'prob' : tf.nn.softmax(logits)
	}

	if mode == tf.contrib.learn.ModeKeys.TRAIN or \
			mode == tf.contrib.learn.ModeKeys.EVAL:
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
		loss = tf.reduce_mean(loss)
		train_op = tf.contrib.layers.optimize_loss(
				loss,
				tf.contrib.framework.get_global_step(),
				optimizer='Adam',
				learning_rate=0.01)
	else:
		loss = None
		train_op = None

	return tflearn.ModelFnOps(
			mode=mode,
			predictions=predictions_dict,
			loss=loss,
			train_op=train_op)

def serving_input_fn():
	feature_placeholders = {
			'first' : tf.placeholder(tf.string, [None]),
	}
	features = {
			key : tf.expand_dims(tensor, -1)
			#for key, tensor in feature_placeholders.items()
	}

	return tflearn.utils.input_fn_utils.InputFnOps(
			features,
			None,
			feature_placeholders)

def get_train():
	return read_dataset('train')

def get_validate():
	return read_dataset('eval')

from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
def train_fn(output_dir):
	tf.logging.info('inside train')
	logs_path = 'logs'
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	return tflearn.Experiment(
			tflearn.Estimator(model_fn=cnn_model, model_dir=output_dir),
			train_input_fn=get_train(),
			eval_input_fn=get_validate(),
			eval_metrics={
				'acc': tflearn.MetricSpec(
					metric_fn=metrics.streaming_accuracy, prediction_key='class'
					)
			},
			export_strategies=[saved_model_export_utils.make_export_strategy(
				serving_input_fn,
				default_output_alternative_key=None,
				exports_to_keep=1
				)
			],
			train_steps=TRAIN_STEPS
		)

