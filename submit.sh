#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e

rm -rf nohup.out

source .venv/bin/activate

nohup python train_trl.py --config playwright --wandb &
# nohup python generate.py --task_name screenplay_summarize &

pid=$!

echo "PID: $pid"
echo "to tail the logs: tail -f nohup.out"
echo "to kill the process: kill -9 $pid"
echo $pid > pid.txt