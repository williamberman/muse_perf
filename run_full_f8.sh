#! /bin/bash

set -e
set -u

GPU="${1}"
FILE=full_f8_$GPU.txt

python muse_perf.py --batch_size 1 --full --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 2 --full --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 4 --full --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 8 --full --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 16 --full --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 32 --full --device cuda --file $FILE --model muse_f8
