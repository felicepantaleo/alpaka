#!/bin/bash
# Compiles one benchmark source against one alpaka tree for one AMD architecture (compile only).
# Usage: build-hip.sh ALPAKA_SOURCE_DIR BENCH_SOURCE OUT_DIR GFX
# Environment: HIPCC (default hipcc).
# Output in OUT_DIR: compile.cmd and compile.log with the -Rpass-analysis=kernel-resource-usage remarks.
set -eu
SRC=$1; BENCH=$2; OUT=$3; GFX=$4
HIPCC=${HIPCC:-hipcc}
mkdir -p "$OUT"
FLAGS=(-std=c++20 -O3 -DNDEBUG "--offload-arch=$GFX" -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_DISABLE_VENDOR_RNG
       -DALPAKA_DEBUG=0 -D__HIP_PLATFORM_AMD__ -I"$SRC/include" -Rpass-analysis=kernel-resource-usage
       -x hip -c "$BENCH" -o "$OUT/bench.o")
echo "$HIPCC ${FLAGS[*]}" > "$OUT/compile.cmd"
"$HIPCC" "${FLAGS[@]}" > "$OUT/compile.log" 2>&1
