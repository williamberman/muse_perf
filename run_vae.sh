#! /bin/bash

set -e
set -u

GPU="${1}"

python muse_perf.py --dtype float16 --batch_size 1 --component vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 2 --component vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 4 --component vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 8 --component vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 16 --component vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 32 --component vae --device cuda --file vae_$GPU.txt
