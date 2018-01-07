#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu --gres=gpu:1 -A comsm0018 -t 0-02:00 --mem=8G --pty bash
