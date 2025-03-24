#!/bin/bash

scripts=$(dirname "$0")
base=$(realpath "$scripts/..")

models=$base/models
logs=$base/logs
data=$base/data/alice
tools=$base/tools

mkdir -p $models
mkdir -p $logs

num_threads=4
device=""

for dropout in 0.0 0.2 0.3 0.5 0.6
do
  echo "Training model with dropout=$dropout"

  CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads \
  python $tools/pytorch-examples/word_language_model/main.py \
    --data $data \
    --epochs 40 \
    --log-interval 100 \
    --emsize 250 --nhid 250 \
    --dropout $dropout --tied \
    --save $models/model_dropout_$dropout.pt \
    --logfile $logs/dropout_$dropout.log
done