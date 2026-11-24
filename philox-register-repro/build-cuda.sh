#!/bin/bash
# Compiles one benchmark source against one alpaka tree for one CUDA architecture and dumps the resource usage.
# Usage: build-cuda.sh ALPAKA_SOURCE_DIR BENCH_SOURCE OUT_DIR SM [link]
# Environment: NVCC (default nvcc), CXX (host compiler, default g++).
# Output in OUT_DIR: compile.cmd, ptxas.log (-Xptxas -v), resusage.txt (cuobjdump -res-usage),
# sass.txt (cuobjdump -sass), and with "link" the executable bench.
set -eu
SRC=$1; BENCH=$2; OUT=$3; SM=$4; LINK=${5:-}
NVCC=${NVCC:-nvcc}; CXX=${CXX:-g++}
CUOBJDUMP=$(dirname "$(command -v "$NVCC")")/cuobjdump
mkdir -p "$OUT"
FLAGS=(-ccbin "$CXX" -std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr --extended-lambda
       -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEBUG=0 -DALPAKA_HAS_STD_ATOMIC_REF
       "--generate-code=arch=compute_$SM,code=[sm_$SM]" -I"$SRC/include" -Xptxas -v -x cu -c "$BENCH" -o "$OUT/bench.o")
echo "$NVCC ${FLAGS[*]}" > "$OUT/compile.cmd"
"$NVCC" "${FLAGS[@]}" > "$OUT/ptxas.log" 2>&1
"$CUOBJDUMP" -res-usage "$OUT/bench.o" > "$OUT/resusage.txt"
"$CUOBJDUMP" -sass "$OUT/bench.o" > "$OUT/sass.txt"
if [ "$LINK" = link ]; then
  "$NVCC" -ccbin "$CXX" "$OUT/bench.o" -o "$OUT/bench"
fi
