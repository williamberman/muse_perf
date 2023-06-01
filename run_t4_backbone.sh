#! /bin/bash

set -e

python muse_perf.py --dtype float16 --batch_size 1 --compiled False --model backbone --device cuda --file backbone_t4.txt
python muse_perf.py --dtype float16 --batch_size 1 --compiled True --model backbone --device cuda --file backbone_t4.txt

python muse_perf.py --dtype float16 --batch_size 2 --compiled False --model backbone --device cuda --file backbone_t4.txt
python muse_perf.py --dtype float16 --batch_size 2 --compiled True --model backbone --device cuda --file backbone_t4.txt

python muse_perf.py --dtype float16 --batch_size 4 --compiled False --model backbone --device cuda --file backbone_t4.txt
python muse_perf.py --dtype float16 --batch_size 4 --compiled True --model backbone --device cuda --file backbone_t4.txt

python muse_perf.py --dtype float16 --batch_size 8 --compiled False --model backbone --device cuda --file backbone_t4.txt
python muse_perf.py --dtype float16 --batch_size 8 --compiled True --model backbone --device cuda --file backbone_t4.txt

python muse_perf.py --dtype float16 --batch_size 16 --compiled False --model backbone --device cuda --file backbone_t4.txt
python muse_perf.py --dtype float16 --batch_size 16 --compiled True --model backbone --device cuda --file backbone_t4.txt

python muse_perf.py --dtype float16 --batch_size 32 --compiled False --model backbone --device cuda --file backbone_t4.txt
python muse_perf.py --dtype float16 --batch_size 32 --compiled True --model backbone --device cuda --file backbone_t4.txt
