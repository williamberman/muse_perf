#! /bin/bash

set -e
set -u

GPU="${1}"

python muse_perf.py --batch_size 1 --full --device cuda --file full_$GPU.txt

python muse_perf.py --batch_size 2 --full --device cuda --file full_$GPU.txt

python muse_perf.py --batch_size 4 --full --device cuda --file full_$GPU.txt

python muse_perf.py --batch_size 8 --full --device cuda --file full_$GPU.txt

python muse_perf.py --batch_size 16 --full --device cuda --file full_$GPU.txt

python muse_perf.py --batch_size 32 --full --device cuda --file full_$GPU.txt
