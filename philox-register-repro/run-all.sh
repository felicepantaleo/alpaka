#!/bin/bash
# Runs the full measurement: builds, resource usage and timing, for four alpaka commits.
# Usage: run-all.sh ALPAKA_CLONE RESULT_DIR [ROUNDS]
# ALPAKA_CLONE is a clone of alpaka that contains the four commits below (fetch the PR branch
# https://github.com/felicepantaleo/alpaka/tree/fix-philox-random123 into it).
# Environment: NVCC, CXX (see build-cuda.sh); HIPCC (optional; without it the HIP builds are skipped);
# CUDA_VISIBLE_DEVICES for the timing; CUDA_ARCHS (default "75 80 90"); TIME_SM (default 75, the GPU of the timing).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
A=$1; OUT=$2; ROUNDS=${3:-5}
CUDA_ARCHS=${CUDA_ARCHS:-75 80 90}; TIME_SM=${TIME_SM:-75}
TREES=(develop=61c5a7113a1b4b3ac2648ceaed88874635d05c7d a83bc065f=a83bc065ffbc1e58db87dab7f8e070968a8a85be
       0dee57950=0dee57950378dd887a6b02c04942170389ff1868 head=48155b7b27fe2f4252d6248f060e44fd2e67f089)
mkdir -p "$OUT/src"
for t in "${TREES[@]}"; do
  n=${t%%=*}; c=${t#*=}
  [ -d "$OUT/src/$n" ] || git -C "$A" worktree add --detach "$OUT/src/$n" "$c"
  echo "$n $(git -C "$OUT/src/$n" rev-parse HEAD)" >> "$OUT/trees.txt"
done
{ "${NVCC:-nvcc}" --version; "${CXX:-g++}" --version | head -1; [ -n "${HIPCC:-}" ] && "$HIPCC" --version | head -3;
  nvidia-smi --query-gpu=index,name,driver_version --format=csv; } > "$OUT/versions.txt" 2>&1 || true
for bench in bench bench2; do
  for t in "${TREES[@]}"; do
    n=${t%%=*}
    for sm in $CUDA_ARCHS; do
      link=""; [ "$sm" = "$TIME_SM" ] && link=link
      "$HERE/build-cuda.sh" "$OUT/src/$n" "$HERE/$bench.cpp" "$OUT/$bench/$n/sm$sm" "$sm" $link
    done
    if [ -n "${HIPCC:-}" ] && [ "$bench" = bench ]; then
      for gfx in gfx90a gfx1100; do "$HERE/build-hip.sh" "$OUT/src/$n" "$HERE/$bench.cpp" "$OUT/$bench-hip/$n/$gfx" "$gfx"; done
    fi
  done
done
python3 "$HERE/metrics.py" "$OUT" > "$OUT/metrics.txt"
for bench in bench bench2; do
  args=(); for t in "${TREES[@]}"; do n=${t%%=*}; args+=("$n=$OUT/$bench/$n/sm$TIME_SM/bench"); done
  "$HERE/time.sh" "$OUT/timing-$bench" "$ROUNDS" "${args[@]}"
  python3 "$HERE/timing.py" "$OUT/timing-$bench" develop a83bc065f 0dee57950 head > "$OUT/timing-$bench.txt"
done
echo "done: $OUT/metrics.txt $OUT/timing-bench.txt $OUT/timing-bench2.txt"
