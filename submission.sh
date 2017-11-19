#!/bin/bash -l

#Specify project
#$ -P cs542

#Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
#$ -l h_rt=12:00:00

#Send an email when the job is done or aborted (by default no email is sent)
#$ -m e

# Give job a name
#$ -N hello

#$ Join output and error streams into one file
#$ -j y



#load appropriate envornment
module load python/2.7.13
module load cuda/8.0
module load cudnn/6.0
module load tensorflow/r1.3

#execute the program
python task.py