#!/bin/sh
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -g gu14
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

n=5
max_len=512

mkdir -p data/sr$n

python3 gen_data.py data/sr$n/train.txt 100000 --min_n $n --max_n $n --max_len $max_len
python3 gen_data.py data/sr$n/test.txt 100 --min_n $n --max_n $n --max_len $max_len