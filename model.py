#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.metrics as metrics
from tensorflow.contrib import lookup
from tensorflow.python import debug as tf_debug
import json
import csv
import pandas as pd

from mapping import mapping as MAPPING
with open('grams.json') as data:
	GRAMS = json.load(data)

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_STEPS = 20

# TODO @Shreya
BATCH_SIZE = 1000

MAX_NAME_LENGTH = 10
PADDING = 'ZYXW'

# TODO @Ben all column names
CSV_COLUMNS = ['first1','first2','first3','first4','first5','first6','first7','first8','first9','first10','last1', 'last2', 'last3', 'last4', 'last5', 'last6', 'last7', 'last8', 'last9', 'last10','source']
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
		
		input_file_names = tf.train.match_filenames_once(filename)
		filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)
		#df = pd.read_csv(filename,header=None)
		
		# load the data with given batch size (constant above)
		reader = tf.TextLineReader(skip_header_lines=False)
		_, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
		value_column = tf.expand_dims(value, -1)

		# TODO @Ben add record_defaults and field_delim values
		# read https://www.tensorflow.org/api_docs/python/tf/decode_csv
		columns = tf.decode_csv(value_column, record_defaults=[['first1'],['first2'],['first3'],['first4'],['first5'],['first6'],['first7'],['first8'],['first9'],['first10'],['last1'], ['last2'], ['last3'], ['last4'], ['last5'], ['last6'], ['last7'], ['last8'], ['last9'], ['last10'],['source']], field_delim=',')
		#features = dict(zip(CSV_COLUMNS, columns))
		#print(columns)
		tf.logging.info('after decode_csv')
		features = tf.string_to_number(tf.slice(columns, [0,0,0], [20,900,1]), tf.float32)
		label = tf.string_to_number(tf.slice(columns,[0,21,0],[1,900,1]), tf.float32)
		tf.logging.info('after slicing columns')
		#with tf.Session() as sess:
		#	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		#	sess.run(features)  # to print a tensor must start a sessions
		tf.logging.info(features)	
		return features, label
	tf.logging.info('right before returning _input_fn')
	#x, y = _input_fn()
	#tf.logging.info(x)
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
	#print(features)
	#print(target)
	tf.logging.info('inside cnn_model')
	first_name = features
	#last_name = tf.squeeze(input=features[CSV_COLUMNS[1]])
	#with tf.Session() as sess:
	#	tf.logging.info(first_name.eval())	
	#	sess.close()
	#tf.logging.info(str(len(first_name)))
	logits = tf.contrib.layers.fully_connected(first_name, num_outputs=2, activation_fn=None)
	tf.logging.info(type(logits))
	# TODO figure out the ethniticity (source) part 
	predictions_dict = {
			'ethnicity' : tf.gather(CLASSES, tf.argmax(logits, 1)),
			'class' : tf.argmax(logits, 1),
			'prob' : tf.nn.softmax(logits)
	}
	tf.logging.info(str(predictions_dict['class']))
	#tf.logging.info(len(target.value))
	if mode == tf.contrib.learn.ModeKeys.TRAIN or \
			mode == tf.contrib.learn.ModeKeys.EVAL:
		loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(target, tf.int32), logits)
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
			#for key, tensor in feature_placeholders.items()
	}

	return tflearn.utils.input_fn_utils.InputFnOps(
			features,
			None,
			feature_placeholders)

def get_train():
	return read_dataset('train')
	#tf.logging.info('inside get_train')
	#return x,y

def get_validate():
	return read_dataset('eval')
	#tf.logging.info('inside get_validate')
	#return x,y

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

