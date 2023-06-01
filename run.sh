#! /bin/bash

set -e

GPU="${1}"

python muse_perf.py --model backbone --device cuda --file backbone_${GPU}.txt
python muse_perf.py --model vae --device cuda --file vae_${GPU}.txt
python muse_perf.py --full --device cuda --file full_${GPU}.txt

python muse_perf.py --model backbone --device cpu --file backbone_cpu.txt
python muse_perf.py --model vae --device cpu --file vae_cpu.txt
python muse_perf.py --full --device cpu --file full_cpu.txt
