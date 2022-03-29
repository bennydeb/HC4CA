#!/bin/bash
# Experiment batch identifier
EXP_NAME=test_run

file='Dataset/pir_3S_train_235244130621.gzip'
application='../utils/run_experiment.py'
$application --pickle  $file --name ${file%.*}_${EXP_NAME}_ --locations --multiclass --test-size 0.2

#! application name
application="repos/HC4CA/src/run_experiment.py"
venv="env/HC4CA/bin/activate"
