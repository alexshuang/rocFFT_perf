#!/bin/sh

set -e

export PATH=$PATH:/opt/rocm/bin

LENGTH='8192'
N=10
COLD_N=1
OUT_DIR=${1:-out}
mkdir -p $OUT_DIR

CMD="./rocfft-rider -t 2 --length $LENGTH -N ${N}"

# BASIC
BASIC_PMC="Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts"
echo "pmc: $BASIC_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --hsa-trace --timestamp on -o $OUT_DIR/basic_prof.csv $CMD

# MEMORY CONFLICT
MEM_CONFLICT_PMC="SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT LDSBankConflict"
echo "pmc: $MEM_CONFLICT_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --hsa-trace --timestamp on -o $OUT_DIR/mem_conflict_prof.csv $CMD

# MEMORY STALLED
MEM_STALLED_PMC="VALUUtilization MemUnitStalled WriteUnitStalled"
echo "pmc: $MEM_STALLED_PMC" > /tmp/input.txt
rocprof -i mem_stalled_input.txt --hsa-trace --timestamp on -o $OUT_DIR/mem_stalled_prof.csv $CMD

export ROCFFT_LAYER=7
export ROCFFT_LOG_PROFILE_PATH=$OUT_DIR/rocfft_prof_log.csv
$CMD 2>&1 | tee $OUT_DIR/rocfft_rider_results.txt

rm -f $OUT_DIR/*.json $OUT_DIR/*.db

# rocprof parse
python3 rocfft_utils.py \
    --basic_prof_file $OUT_DIR/basic_prof.csv \
    --mem_conflict_prof_file $OUT_DIR/mem_conflict_prof.csv \
    --mem_stalled_prof_file $OUT_DIR/mem_stalled_prof.csv \
    --basic_pmc "$BASIC_PMC" \
    --mem_conflict_pmc "$MEM_CONFLICT_PMC" \
    --mem_stalled_pmc "$MEM_STALLED_PMC" \
    --rocfft_log_profile $OUT_DIR/rocfft_prof_log.csv \
    --rider_results $OUT_DIR/rocfft_rider_results.txt \
    --num_iter $N \
    --num_cold_iter $COLD_N \
    | tee analysis_$LENGTH.log

