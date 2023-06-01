#! /bin/bash

set -e
set -u

GPU="${1}"

python muse_perf.py --dtype float16 --batch_size 1 --model backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 2 --model backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 4 --model backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 8 --model backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 16 --model backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 32 --model backbone --device cuda --file backbone_$GPU.txt
