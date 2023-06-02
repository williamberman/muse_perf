#! /bin/bash

set -e
set -u

GPU=t4

python muse_perf.py --batch_size 1 --component backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --batch_size 2 --component backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --batch_size 4 --component backbone --device cuda --file backbone_$GPU.txt

python muse_perf.py --batch_size 8 --component backbone --device cuda --file backbone_$GPU.txt --compiled None
python muse_perf.py --batch_size 8 --component backbone --device cuda --file backbone_$GPU.txt --compiled default
python muse_perf.py --batch_size 8 --component backbone --device cuda --file backbone_$GPU.txt --compiled reduce-overhead --model muse_f16

python muse_perf.py --batch_size 16 --component backbone --device cuda --file backbone_$GPU.txt --compiled None
python muse_perf.py --batch_size 16 --component backbone --device cuda --file backbone_$GPU.txt --compiled default
python muse_perf.py --batch_size 16 --component backbone --device cuda --file backbone_$GPU.txt --compiled reduce-overhead --model muse_f16

python muse_perf.py --batch_size 32 --component backbone --device cuda --file backbone_$GPU.txt --compiled None
python muse_perf.py --batch_size 32 --component backbone --device cuda --file backbone_$GPU.txt --compiled default
python muse_perf.py --batch_size 32 --component backbone --device cuda --file backbone_$GPU.txt --compiled reduce-overhead --model muse_f16
