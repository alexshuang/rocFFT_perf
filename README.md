# rocFFT_perf
rocFFT performance testing tools.

# Usage
Usage: rocfft_perf.sh <LENGTH> [TRANSFORM_TYPE=0] [N=10] [COLD_N=1] [BATCH_COUNT=1] [OUT_DIR=out]
       e.g. rocfft_perf.sh 32-32-32 ...

# Results
<OUT_DIR>/perf_len<LENGTH>.log
