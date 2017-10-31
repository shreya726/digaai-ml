#!/usr/bin/env python

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

TRAN_STEPS = 1000

# TODO @Shreya
BATCH_SIZE = 0

MAX_NAME_LENGTH = 10

# TODO @Ben all column names
CSV_COLUMNS = ['first?', 'last?']
LABEL_COLUMN = 'first?'

def save_grams():
    """
    saves new 3-grams with their numerical values to a mapping
    """
    pass

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
        columns = tf.decode_csv(value_column, record_defaults=NONE, field_delim=NONE)
        features = dict(zip(CSV_COLUMNS, columns))
        label = feature.pop(LABEL_COLUMN)

        # convert input to numeric value
        # TODO @Duaa and @Shreya: adjust the code below so it does sth like this
        # 1- looks up the 3-gram's value in the 3-gram's mapping
        # 2- if doesn't exist, construct one using the letter mapping
        # 3- and adds it to the table, anything else?

        #table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.contant(TARGETS))
        #target = table.lookup(label)

        # first it creates a table with required params
        # read https://www.tensorflow.org/api_docs/python/tf/contrib/lookup/index_table_from_file
        from sys import maxsize as NULL
        table = tf.contrib.lookup.index_table_from_file(
                vocabulary_file='DATA FILENAME', num_oov_buckets=NULL,
                vocab_size=NULL default_value=NULL)

        # tf.constant just converts the name split (3-grams) into a "constant" type for lookup
        # read https://www.tensorflow.org/api_docs/python/tf/constant
        # replace NULL with 3-grams of current input
        numbers = table.lookup(tf.constant(NULL))

        # this code initializes a table and keeps it open for other iterations
        # read here: https://www.tensorflow.org/api_docs/python/tf/tables_initializer
        with tf.Session() as sess:
            tf.tables_initializer().run()
            print "{} --> {}".format(lines[0], numbers.eval())
            
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




