#!/bin/bash -l

echo "deleting old output dir and creating a new one with 777 permissions"
rm -rf output
mkdir output
chmod 777 output

#load appropriate envornment
module load cuda/8.0
module load cudnn/5.1
module load python/3.6.2
module load tensorflow/r1.3_cpu

#execute the program
python task.py
