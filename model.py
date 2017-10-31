#!/usr/bin/env python

import tensorflow as tf

from mapping import mapping as MAPPING
from grams import grams as GRAMS

tf.logging.set_verbosity(tf.logging.INFO)

TRAN_STEPS = 1000

# TODO @Shreya
BATCH_SIZE = 0

MAX_NAME_LENGTH = 10

# TODO @Ben all column names
CSV_COLUMNS = ['first?', 'last?']
LABEL_COLUMN = 'first?'

def save_gram(gram, num_val):
    pass

def get_3grams(name):
    """ Get 3-grams of names 
    """
    if len(name) < BATCH_SIZE:
        grams = [name[i:i+3] for i in range(0, len(name) - 2)]

        # Padding names that are shorter than average
        padding = 'ZYXW'
        if BATCH_SIZE - len(name) > 0:
            grams += [name[len(name) - 2:] + padding]
        if BATCH_SIZE - len(name) > 1:
            grams += [name[-1:] + padding*2]
        if BATCH_SIZE - len(name) > 2:
            for i in range(len(name), BATCH_SIZE):
                grams += [padding*3]
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
                converted_gram += [MAPPING[letter]]
            num_val = '0'.join(converted_gram)
            result += [int(num_val)]
            save_gram(gram, num_val)
    return result

def read_dataset(mode):
    # TODO @Ben use mode to create filename so that we can 
    # pass mode as 'train' or 'eval'
    filename = "PATH OF FILE"
    if prefix == "train":
        mode = tf.contrib.learn.ModeKeys.TRAIN
    else:
        mode = tf.contrib.learn.ModeKeys.EVAL

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

        # First name
        first = features.pop('first')
        first_3grams = get_3grams(first)

        # Last name
        last = features.pop('last')
        last_3grams = get_3grams(last)


        # convert input to numeric value
        # TODO @Duaa and @Shreya: adjust the code below so it does sth like this

        first_vector = convert_to_numerical(first_3grams)
        last_vector = convert_to_numerical(last_3grams)
       

        #table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.contant(TARGETS))
        #target = table.lookup(label)

        # first it creates a table with required params
        # read https://www.tensorflow.org/api_docs/python/tf/contrib/lookup/index_table_from_file
        from sys import maxsize as NULL
        table = tf.contrib.lookup.index_table_from_file(
                vocabulary_file='DATA FILENAME', num_oov_buckets=NULL,
                vocab_size=NULL, default_value=NULL)

        # tf.constant just converts the name split (3-grams) into a "constant" type for lookup
        # read https://www.tensorflow.org/api_docs/python/tf/constant
        # replace NULL with 3-grams of current input
        numbers = table.lookup(tf.constant(NULL))

        # this code initializes a table and keeps it open for other iterations
        # read here: https://www.tensorflow.org/api_docs/python/tf/tables_initializer
        with tf.Session() as sess:
            tf.tables_initializer().run()
            print("{} --> {}".format(lines[0], numbers.eval()))
            
        return features, target

    return _input_fn

def cnn_model(features, target, mode):
    
    # load the 3-gram mappings 
    # TODO @Shreya for filename 
    table = lookup.index_table_from_file(vocabulary_file="FILENAME", num_oov_buckets=1, default_value=-1)
    
    # where CSV_COLUMNS[1] is 'first'
    first_name = tf.squeeze(features[CSV_COLUMNS[0]], [1])
    
    # TODO @Wjdan
    # 0- make sure it doesn't exceed MAX_NAME_LENGTH
    # 1- generate 3-grams of first_name
    # 2- look its numerical value in the table 
    # 3- WHAT TO PUT IF NOT FOUND??????
    

def get_train():
    #reaturn read_dataset('train')
    pass

def get_validate():
    #return read_dataset('eval')
    pass

def train_fn():
    pass




