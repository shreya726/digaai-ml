import argparse
import json
import os

import nn_model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
			'--output_dir',
			help='GCS location to write checkpoints and export models',
			default="./output"
 )
	parser.add_argument(
			'--train_steps',
			help='How many batches to run training job for',
			type=int,
			default=1000
	)

	args = parser.parse_args()
	arguments = args.__dict__
	
	# unused args provided by service
	arguments.pop('job_dir', None)
	arguments.pop('job-dir', None)

	output_dir = arguments.pop('output_dir')

	output_dir = os.path.join(
			output_dir,
			json.loads(
					os.environ.get('TF_CONFIG', '{}')
			).get('task', {}).get('trail', '')
	)

	# Run the training job
	learn_runner.run(nn_model.train_fn, output_dir)
