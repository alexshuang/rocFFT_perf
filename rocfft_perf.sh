#!/bin/bash

set -e

export PATH=$PATH:/opt/rocm/bin

LENGTH=$1
BATCH_COUNT=${2:-1}
TRANS_TYPE=${3:-0}
N=${4:-10}
COLD_N=${5:-1}
ISTRIDE=${6:-strided}
OSTRIDE=${7:-strided}
OUT_DIR=${8:-out}
OUT_DIR=$OUT_DIR/len${LENGTH}_b${BATCH_COUNT}_N${N}
RESULT_FILE=$OUT_DIR/perf_len$LENGTH.log
LENGTH=`echo $LENGTH | awk -F'-' '{ print($1, $2, $3, $4, $5) }'`
if [ $ISTRIDE != "strided" ]; then
	ISTRIDE=`echo $ISTRIDE | awk -F'-' '{ print($1, $2, $3, $4, $5) }'`
fi
if [ $OSTRIDE != "strided" ]; then
	OSTRIDE=`echo $OSTRIDE | awk -F'-' '{ print($1, $2, $3, $4, $5) }'`
fi
echo "ISTRIDE = $ISTRIDE, OSTRIDE = $OSTRIDE"
mkdir -p $OUT_DIR

if [ $# -lt 1 ]; then
    echo "ERROR: Required parameter missing."
    echo -e "Usage: rocfft_perf.sh <LENGTH> [BATCH_COUNT=1] [TRANSFORM_TYPE=0] [N=10] [COLD_N=1] [ISTRIDE=1] [OSTRIDE=1] [OUT_DIR=out]"
    echo -e "\te.g. rocfft_perf.sh 32-32-32 ..."
    exit 1
fi

CMD="./rocfft-rider --length $LENGTH -t $TRANS_TYPE -N $N -b $BATCH_COUNT"
if [ "$ISTRIDE" != "strided" ]; then
	CMD+=" --istride $ISTRIDE"
fi
if [ "$OSTRIDE" != "strided" ]; then
	CMD+=" --ostride $OSTRIDE"
fi

echo $CMD

# BASIC
BASIC_PMC="L2CacheHit"
echo "pmc: $BASIC_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --hsa-trace --timestamp on -o $OUT_DIR/basic_prof.csv $CMD

# INSTS
INSTS_PMC="Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts"
echo "pmc: $INSTS_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --hsa-trace --timestamp on -o $OUT_DIR/insts_prof.csv $CMD

# MEMORY CONFLICT
MEM_CONFLICT_PMC="SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT LDSBankConflict"
echo "pmc: $MEM_CONFLICT_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --hsa-trace --timestamp on -o $OUT_DIR/mem_conflict_prof.csv $CMD

# MEMORY STALLED
MEM_STALLED_PMC="VALUUtilization MemUnitStalled WriteUnitStalled"
echo "pmc: $MEM_STALLED_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --hsa-trace --timestamp on -o $OUT_DIR/mem_stalled_prof.csv $CMD

export ROCFFT_LAYER=7
export ROCFFT_LOG_PROFILE_PATH=$OUT_DIR/rocfft_prof_log.csv
$CMD 2>&1 | tee $OUT_DIR/rocfft_rider_results.txt

rm -f $OUT_DIR/*.json $OUT_DIR/*.db

# rocprof parse
python3 rocfft_utils.py \
    --basic_prof_file $OUT_DIR/basic_prof.csv \
    --insts_prof_file $OUT_DIR/insts_prof.csv \
    --mem_conflict_prof_file $OUT_DIR/mem_conflict_prof.csv \
    --mem_stalled_prof_file $OUT_DIR/mem_stalled_prof.csv \
    --basic_pmc "$BASIC_PMC" \
    --insts_pmc "$INSTS_PMC" \
    --mem_conflict_pmc "$MEM_CONFLICT_PMC" \
    --mem_stalled_pmc "$MEM_STALLED_PMC" \
    --rocfft_log_profile $OUT_DIR/rocfft_prof_log.csv \
    --rider_results $OUT_DIR/rocfft_rider_results.txt \
    --num_iter $N \
    --num_cold_iter $COLD_N \
    --batch_count $BATCH_COUNT \
    | tee $RESULT_FILE

