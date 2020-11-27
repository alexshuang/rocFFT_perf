#!/bin/sh

set -e

BATCH=10000
N=10
N_COLD=1

./rocfft_perf.sh 4096  $BATCH 0 $N $N_COLD
./rocfft_perf.sh 3125 $BATCH 0 $N $N_COLD
./rocfft_perf.sh 512 $BATCH 0 $N $N_COLD
./rocfft_perf.sh 2187 $BATCH 0 $N $N_COLD
./rocfft_perf.sh 112 $BATCH 0 $N $N_COLD
