#! /bin/bash

set -e
set -u

GPU="${1}"
FILE=vae_f8_$GPU.txt

python muse_perf.py --batch_size 1 --component vae --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 2 --component vae --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 4 --component vae --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 8 --component vae --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 16 --component vae --device cuda --file $FILE --model muse_f8
python muse_perf.py --batch_size 32 --component vae --device cuda --file $FILE --model muse_f8
