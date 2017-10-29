#!/usr/bin/env python

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

TRAN_STEPS = 1000

# TODO @Shreya
BATCH_SIZE = 0

MAX_NAME_LENGTH = 10
CSV_COLUMNS = ['first', 'last']

def save_grams():
    """
    saves new 3-grams with their numerical values to a mapping
    """
    pass

def read_dataset(mode):

    def _input_fn()   # gets passed to tensorflow
        pass

    return _input_fn

def cnn_model(features, target, mode):
    pass

def get_train():
    #reaturn read_dataset('train')
    pass

def get_validate():
    #return read_dataset('eval')
    pass

def train_fn():
    pass




