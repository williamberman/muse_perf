#! /bin/bash

set -e
set -u

GPU="${1}"
DEVICE="${2}"

python muse_perf.py --batch_size 1 --component backbone --device $DEVICE --file backbone_$GPU.txt

python muse_perf.py --batch_size 2 --component backbone --device $DEVICE --file backbone_$GPU.txt

python muse_perf.py --batch_size 4 --component backbone --device $DEVICE --file backbone_$GPU.txt

python muse_perf.py --batch_size 8 --component backbone --device $DEVICE --file backbone_$GPU.txt

python muse_perf.py --batch_size 16 --component backbone --device $DEVICE --file backbone_$GPU.txt

python muse_perf.py --batch_size 32 --component backbone --device $DEVICE --file backbone_$GPU.txt
