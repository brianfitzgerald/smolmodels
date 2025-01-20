#!/bin/bash

set -e

rm -rf nohup.out

source .venv/bin/activate

nohup python train_trl.py --config ultrafeedback --wandb &

pid=$!

echo "PID: $pid"
echo "to tail the logs: tail -f nohup.out"
echo "to kill the process: kill -9 $pid"
echo $pid > pid.txt