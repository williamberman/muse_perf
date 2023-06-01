#! /bin/bash

set -e
set -u

GPU="${1}"

python muse_perf.py --dtype float16 --batch_size 1 --model vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 2 --model vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 4 --model vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 8 --model vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 16 --model vae --device cuda --file vae_$GPU.txt

python muse_perf.py --dtype float16 --batch_size 32 --model vae --device cuda --file vae_$GPU.txt