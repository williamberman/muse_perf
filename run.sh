#! /bin/bash

set -e

GPU="${1}"

python muse_perf.py --model transformer --device cuda --file transformer_${GPU}
python muse_perf.py --model vae --device cuda --file vae_${GPU}
python muse_perf.py --model transformer --device cpu --file transformer_cpu
python muse_perf.py --model vae --device cpu --file vae_cpu
