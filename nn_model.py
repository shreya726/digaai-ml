#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.metrics as metrics

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_STEPS = 2000
BATCH_SIZE = 1000

# 1 for brazilian and 0 for non-brazilian 
CLASSES = ['1', '0']  

def read_dataset(mode):
	# Path of data file
	filename = "./data/"

	if mode == "train":
		mode = tf.contrib.learn.ModeKeys.TRAIN
		filename += "training_onehot.csv"
	else:
		mode = tf.contrib.learn.ModeKeys.EVAL
		filename += "eval_onehot.csv" # FIXME for testing?

	def _input_fn(): 
		# gets the file and parses it		
		input_file_names = tf.train.match_filenames_once(filename)
		filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)
		
		# Parse using pandas
		df = pd.read_csv(filename,header=None)
		tf.logging.info('Finished reading CSV')
		
		# Labels is last column
		labels = tf.convert_to_tensor(df.iloc[:,520].as_matrix(),np.float32)
		
		features = df.iloc[:,0:520].as_matrix()
		features = tf.convert_to_tensor(features,np.float32)
		tf.logging.info('Converted dataframe to tensor of shape ' + str(features.shape))
		
		# Reshaping features, labels
		features = tf.reshape(features, shape=[-1,520])
		labels = tf.reshape(labels, shape=[-1,1])
		tf.logging.info('Reshaped features and labels')
			
		return features, labels

	return _input_fn

def nn_model(features, target, mode):
	tf.logging.info('Inside nn_model')

	logits = tf.contrib.layers.fully_connected(features, num_outputs=1, activation_fn=tf.sigmoid)
	 
	predictions_dict = {
			'ethnicity' : tf.gather(CLASSES, tf.argmax(logits, 1)),
			'class' : tf.argmax(logits, 1),
			'prob' : tf.nn.softmax(logits)
	}
	
	if mode == tf.contrib.learn.ModeKeys.TRAIN or \
			mode == tf.contrib.learn.ModeKeys.EVAL:
		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
		loss = tf.reduce_mean(loss)
		train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
	else:
		loss = None
		train_op = None

	return tflearn.ModelFnOps(
			mode=mode,
			predictions=predictions_dict,
			loss=loss,
			train_op=train_op)

def serving_input_fn():

	feature_placeholders = {str(i): tf.placeholder(tf.string, [None]) for i in range(0,20)}
	features = {key: tf.expand_dims(tensor, -1) for key, tensor in feature_placeholders.items()}

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
	tf.logging.info('Inside train_fn')
	
	logs_path = 'logs'
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	
	return tflearn.Experiment(
			tflearn.Estimator(model_fn=nn_model, model_dir=output_dir),
			train_input_fn=get_train(),
			eval_input_fn=get_validate(),
			eval_metrics={
				'acc': tflearn.MetricSpec(
					metric_fn=metrics.streaming_accuracy, prediction_key='class'
					)
			},
			export_strategies=None,
			train_steps=TRAIN_STEPS
	)