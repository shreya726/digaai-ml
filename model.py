#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import json

from mapping import mapping as MAPPING
with open('grams.json') as data:
    GRAMS = json.load(data)

tf.logging.set_verbosity(tf.logging.INFO)

TRAN_STEPS = 1000

# TODO @Shreya
BATCH_SIZE = 0

MAX_NAME_LENGTH = 10
PADDING = 'ZYXW'

# TODO @Ben all column names
CSV_COLUMNS = ['first', 'last', 'source' ]
LABEL_COLUMN = 'source'

# 1 for brazilian and 0 for non-brazilian 
CLASSES = ['1', '0']  

def get_3grams(name):
    """ Get 3-grams of names 
    """
    if len(name) < MAX_NAME_LENGTH:
        grams = [name[i:i+3] for i in range(0, len(name) - 2)]

        # Padding names that are shorter than average
        overflow = MAX_NAME_LENGTH - len(name)
        if overflow > 0:
            grams += [name[len(name) - 2:] + PADDING]
        if overflow > 1:
            grams += [name[-1:] + PADDING*2]
        if overflow > 2:
            for i in range(len(name), MAX_NAME_LENGTH):
                grams += [PADDING*3]
        return grams
    
    else:
        # truncating names that are longer than average
        return [name[i:i+3] for i in range(0, len(name) - 2)]

def convert_to_numerical(grams):
    result = []
    for gram in grams:
        try:
            result += [int(GRAMS[gram])]
        except KeyError:
            converted_gram = []
            for letter in gram:
                converted_gram += [MAPPING[letter.encode('utf-8')]]
            num_val = '0'.join(converted_gram)
            result += [int(num_val)]

            # Saving grams to dictionary
            GRAMS[gram] = num_val
    return result

def read_dataset(mode):
    # TODO @Ben use mode to create filename so that we can 
    # pass mode as 'train' or 'eval'
    filename = "./data/"	#PATH OF FILE"

    if prefix == "train":
	mode = tf.contrib.learn.ModeKeys.TRAIN
	filename += "training"
    else:
	mode = tf.contrib.learn.ModeKeys.EVAL
	filename += "eval" # FIXME for testing?

    def _input_fn():   # gets passed to tensorflow
	# gets the file and parses it
	input_file_names = tf.train.match_filenames_once(filename)
	filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)

	# load the data with given batch size (constant above)
	reader = tf.TextLineReader()
	_, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
	value_column = tf.expand_dims(value, -1)

	# TODO @Ben add record_defaults and field_delim values
	# read https://www.tensorflow.org/api_docs/python/tf/decode_csv
	columns = tf.decode_csv(value_column, record_defaults=NONE, field_delim=',')
	features = dict(zip(CSV_COLUMNS, columns))

	# source name
	label = features.pop(LABEL_COLUMN)

        # First name into grams
        first = features.pop('first')
        first_3grams = get_3grams(first)

        # Last name into grams
        last = features.pop('last')
        last_3grams = get_3grams(last)

        # convert input to numeric value
        first_vector = convert_to_numerical(first_3grams)
        last_vector = convert_to_numerical(last_3grams)

        # Saving updated grams to json file
        with open('grams.json') as outfile:
            json.dump(GRAMS, outfile)
       
        #table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.contant(TARGETS))
        #target = table.lookup(label)

        # NOT NEEDED
        # first it creates a table with required params
        # read https://www.tensorflow.org/api_docs/python/tf/contrib/lookup/index_table_from_file
        #from sys import maxsize as NULL
        #table = tf.contrib.lookup.index_table_from_file(
        #        vocabulary_file='DATA FILENAME', num_oov_buckets=NULL,
        #        vocab_size=NULL, default_value=NULL)

        # tf.constant just converts the name split (3-grams) into a "constant" type for lookup
        # read https://www.tensorflow.org/api_docs/python/tf/constant
        # replace NULL with 3-grams of current input
        # numbers = table.lookup(label)

        # this code initializes a table and keeps it open for other iterations
        # read here: https://www.tensorflow.org/api_docs/python/tf/tables_initializer
        with tf.Session() as sess:
            tf.tables_initializer().run()
            print("{} --> {}".format(lines[0], numbers.eval()))
            
        return features, label

    return _input_fn

def cnn_model(features, target, mode):
    # load the 3-gram mappings 
    # TODO @Shreya for filename 
    table = lookup.index_table_from_file(vocabulary_file="grams.json",
            num_oov_buckets=1, default_value=-1)
    
    # where CSV_COLUMNS[0] is 'first'
    # I'm not sure if the axis argument is useful in our case 
    # sice our input has only one value in it. So I remove axis=[1]
    # read https://www.tensorflow.org/api_docs/python/tf/squeeze
    # but I am also converned that tf.squeeze() will remove the name itself
    # so I will mark this FIXME in case we run into problems later
    first_name = features[CSV_COLUMNS[0]]
    #last_name = tf.squeeze(input=features[CSV_COLUMNS[1]])

    logits = tf.contrib.layers.fully_connected(input=words, num_outputs=len(CLASSES), activation_fn=None)

    # TODO figure out the ethniticity (source) part 
    predictions_dict = {
            'ethnicity' : tf.gather(CLASSES, tf.argmax(logits, 1)),
            'class' : tf.argmax(logits, 1),
            'prob' : tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or \
            mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(first_name, logits)
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
def train_fn(output_path):
    return tflearn.Experiment(
            tflearn.Estimator(model_fn=cnn_model, model_dir=output_path),
            train_input_fn=get_train(),
            eval_input_fn=get_valid(),
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

