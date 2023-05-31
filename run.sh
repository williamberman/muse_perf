#! /bin/bash

set -e

GPU="${1}"

python muse_perf.py --model transformer --device cuda --file transformer_${GPU}.txt
python muse_perf.py --model vae --device cuda --file vae_${GPU}.txt
python muse_perf.py --model transformer --device cpu --file transformer_cpu.txt
python muse_perf.py --model vae --device cpu --file vae_cpu.txt
