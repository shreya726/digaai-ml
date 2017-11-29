#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.metrics as metrics
from tensorflow.contrib import lookup
import json
import csv

from mapping import mapping as MAPPING
with open('grams.json') as data:
	GRAMS = json.load(data)

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_STEPS = 1000

# TODO @Shreya
BATCH_SIZE = 0

MAX_NAME_LENGTH = 10
PADDING = 'ZYXW'

# TODO @Ben all column names
CSV_COLUMNS = ['first', 'last', 'source' ]
LABEL_COLUMN = 'source'

# 1 for brazilian and 0 for non-brazilian 
CLASSES = ['1', '0']  

def read_dataset(mode):
	# TODO @Ben use mode to create filename so that we can 
	# pass mode as 'train' or 'eval'
	filename = "./data/"	#PATH OF FILE"


	if mode == "train":
		mode = tf.contrib.learn.ModeKeys.TRAIN
		filename += "training_processed.csv"
	else:
		mode = tf.contrib.learn.ModeKeys.EVAL
		filename += "eval_processed.csv" # FIXME for testing?

	def _input_fn():   # gets passed to tensorflow
		# gets the file and parses it
		# inputs_as tensors = []
		# with open(filename, 'r') as data:
		# 	csv_reader = csv.reader(data, delimiter=',')
		# 	for row in csv_reader:
		# 		first = [convert_to_tensor(g) for g in convert_to_numerical(get_3grams(row[0]))]
		# 		#last = convert_to_numerical(get_3grams(row[1]))
		# 		source = int(row[2])
		# 		inputs_as_tensors += [[first,source]]
		input_file_names = tf.train.match_filenames_once(filename)
		# print('FILE NAMES')
		# print(filename)
		# print(input_file_names.initial_value)
		# print('RIP')
		filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)

		# load the data with given batch size (constant above)
		reader = tf.TextLineReader()
		_, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
		value_column = tf.expand_dims(value, -1)

		# TODO @Ben add record_defaults and field_delim values
		# read https://www.tensorflow.org/api_docs/python/tf/decode_csv
		columns = tf.decode_csv(value_column, record_defaults=[['first'],['last'],['source']], field_delim=',')
		features = dict(zip(CSV_COLUMNS, columns))

		# source name, removes it from the features dictionary (table)
		label = features.pop(LABEL_COLUMN)

		# this code initializes a table and keeps it open for other iterations
		# read here: https://www.tensorflow.org/api_docs/python/tf/tables_initializer
		with tf.Session() as sess:
			sess.run(features)  # to print a tensor must start a sessions
			
		return features, label

	return _input_fn

def cnn_model(features, target, mode):
	# load the 3-gram mappings 
	# TODO @Shreya for filename 
	#table = lookup.index_table_from_file(vocabulary_file="grams.json",
	#		num_oov_buckets=1, default_value=-1)
	
	# where CSV_COLUMNS[0] is 'first'
	# I'm not sure if the axis argument is useful in our case 
	# sice our input has only one value in it. So I remove axis=[1]
	# read https://www.tensorflow.org/api_docs/python/tf/squeeze
	# but I am also converned that tf.squeeze() will remove the name itself
	# so I will mark this FIXME in case we run into problems later
	first_name = features[CSV_COLUMNS[0]]
	#last_name = tf.squeeze(input=features[CSV_COLUMNS[1]])
	
	#tf.log()
	print(first_name)
	logits = tf.contrib.layers.fully_connected(tf.to_float(first_name), num_outputs=len(CLASSES))

	# TODO figure out the ethniticity (source) part 
	predictions_dict = {
			'ethnicity' : tf.gather(CLASSES, tf.argmax(logits, 1)),
			'class' : tf.argmax(logits, 1),
			'prob' : tf.nn.softmax(logits)
	}

	if mode == tf.contrib.learn.ModeKeys.TRAIN or \
			mode == tf.contrib.learn.ModeKeys.EVAL:
		loss = tf.losses.sparse_softmax_cross_entropy(tf.to_int32(target), logits)
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
	#'last'  : tf.placeholder(tf.string, [None]),
	feature_placeholders = {
			'first' : tf.placeholder(tf.string, [None]),
	}
	features = {
			key : tf.expand_dims(tensor, -1)
			for key, tensor in feature_placeholders.items()
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

