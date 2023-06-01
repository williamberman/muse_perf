#! /bin/bash

set -e

python muse_perf.py --dtype float16 --batch_size 1 --compiled False --full --device cuda --file full_t4.txt
python muse_perf.py --dtype float16 --batch_size 1 --compiled True --full --device cuda --file full_t4.txt

python muse_perf.py --dtype float16 --batch_size 2 --compiled False --full --device cuda --file full_t4.txt
python muse_perf.py --dtype float16 --batch_size 2 --compiled True --full --device cuda --file full_t4.txt

python muse_perf.py --dtype float16 --batch_size 4 --compiled False --full --device cuda --file full_t4.txt
python muse_perf.py --dtype float16 --batch_size 4 --compiled True --full --device cuda --file full_t4.txt

python muse_perf.py --dtype float16 --batch_size 8 --compiled False --full --device cuda --file full_t4.txt
python muse_perf.py --dtype float16 --batch_size 8 --compiled True --full --device cuda --file full_t4.txt

python muse_perf.py --dtype float16 --batch_size 16 --compiled False --full --device cuda --file full_t4.txt
python muse_perf.py --dtype float16 --batch_size 16 --compiled True --full --device cuda --file full_t4.txt

python muse_perf.py --dtype float16 --batch_size 32 --compiled False --full --device cuda --file full_t4.txt
python muse_perf.py --dtype float16 --batch_size 32 --compiled True --full --device cuda --file full_t4.txt
