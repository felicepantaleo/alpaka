#!/bin/bash
# Runs benchmark executables in interleaved rounds on one GPU.
# Usage: time.sh OUT_DIR ROUNDS NAME=EXECUTABLE...
# Environment: CUDA_VISIBLE_DEVICES selects the GPU.
# Each run uses 2^20 threads, 1024 draws per thread, 21 launches and offset 6, and prints the median time per kernel.
# The order of the executables rotates by one position in each round. Three runs of the first executable before
# round 1 bring the GPU to its working temperature; they go to preheat.log.
# Output in OUT_DIR: NAME.rN.log, preheat.log, gpu-before.txt, gpu-after.txt.
set -eu
OUT=$1; ROUNDS=$2; shift 2
ARGS="1048576 1024 21 6"
mkdir -p "$OUT"
NAMES=(); EXES=()
for a in "$@"; do NAMES+=("${a%%=*}"); EXES+=("${a#*=}"); done
N=${#NAMES[@]}
Q="--query-gpu=index,name,temperature.gpu,clocks.gr --format=csv"
nvidia-smi $Q > "$OUT/gpu-before.txt"
: > "$OUT/preheat.log"
for i in 1 2 3; do "${EXES[0]}" $ARGS >> "$OUT/preheat.log" 2>&1; done
for r in $(seq 1 "$ROUNDS"); do
  for i in $(seq 0 $((N - 1))); do
    j=$(( (i + r - 1) % N ))
    "${EXES[$j]}" $ARGS > "$OUT/${NAMES[$j]}.r$r.log" 2>&1
  done
done
nvidia-smi $Q > "$OUT/gpu-after.txt"
