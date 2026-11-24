# Register, local memory and time of the alpaka Philox engines

This package measures the device code of alpaka-group/alpaka#2737 against `develop`. It contains the two benchmark sources, the scripts that build and run them, and the outputs of one run of these scripts on a machine with two Tesla T4 GPUs.

## Trees

`run-all.sh` measures four commits. `head` is the head of the PR.

```
develop 61c5a7113a1b4b3ac2648ceaed88874635d05c7d
a83bc065f a83bc065ffbc1e58db87dab7f8e070968a8a85be
0dee57950 0dee57950378dd887a6b02c04942170389ff1868
head 48155b7b27fe2f4252d6248f060e44fd2e67f089
```

## Tools

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Sep_01_08:45:21_PDT_2026
Cuda compilation tools, release 13.4, V13.4.92
Build cuda_13.4.r13.4/compiler.38855100_0
g++ (GCC) 14.3.1 20250602
sh: /cvmfs/cms.cern.ch/el8_amd64_gcc14/external/rocm-llvm/7.14-7f80ee39e9b6f1b378d8f90f45873589/bin/rocm_agent_enumerator: No such file or directory
HIP version: 7.14.60850-0000000
AMD clang version 23.0.0git
Target: x86_64-redhat-linux-gnu
index, name, driver_version
0, Tesla T4, 580.178.04
1, Tesla T4, 580.178.04
```

## Commands

The branch `philox-register-repro` of https://github.com/felicepantaleo/alpaka is the head of the PR plus this directory. It contains the four commits:

```
git clone -b philox-register-repro https://github.com/felicepantaleo/alpaka.git
```

Run everything (builds for sm_75, sm_80 and sm_90, HIP builds for gfx90a and gfx1100 when `HIPCC` is set, timing in 5 interleaved rounds on the GPU of `CUDA_VISIBLE_DEVICES`):

```
export NVCC=/path/to/nvcc CXX=/path/to/g++ HIPCC=/path/to/hipcc CUDA_VISIBLE_DEVICES=0
alpaka/philox-register-repro/run-all.sh alpaka RESULT_DIR 5
```

The timing uses sm_75 (`TIME_SM=75`). On another GPU, set `TIME_SM` and `CUDA_ARCHS`. The run here used `numactl --cpunodebind=1 --membind=1` around `run-all.sh` and GPU 1.

`run-all.sh` calls these scripts. Each one can run alone:

- `build-cuda.sh ALPAKA_SOURCE_DIR BENCH_SOURCE OUT_DIR SM [link]`: the nvcc build, then `cuobjdump -res-usage` and `cuobjdump -sass`.
- `build-hip.sh ALPAKA_SOURCE_DIR BENCH_SOURCE OUT_DIR GFX`: the hipcc build with `-Rpass-analysis=kernel-resource-usage`.
- `metrics.py RESULT_DIR`: one line for each kernel and build, in `metrics-cuda.tsv` and `metrics-hip.tsv`.
- `time.sh OUT_DIR ROUNDS NAME=EXECUTABLE...` and `timing.py OUT_DIR BASE NAME...`: the timing and its summary.

The nvcc command of `bench.cpp` at the head for sm_75, as `build-cuda.sh` ran it:

```
/cvmfs/cms.cern.ch/el8_amd64_gcc14/external/cuda/13.4.2-7839b8d267e0792e99fb983fb2f19ff5/bin/nvcc -ccbin /cvmfs/cms.cern.ch/el8_amd64_gcc14/external/gcc/14.3.1-724da22786638848892aa9ded8fcd995/bin/g++ -std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr --extended-lambda -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEBUG=0 -DALPAKA_HAS_STD_ATOMIC_REF --generate-code=arch=compute_75,code=[sm_75] -IRESULT_DIR/src/head/include -Xptxas -v -x cu -c REPRO_DIR/bench.cpp -o RESULT_DIR/bench/head/sm75/bench.o
```

The hipcc command for gfx90a:

```
/cvmfs/cms.cern.ch/el8_amd64_gcc14/external/rocm-hip/7.14-3ed34bea828920de75b4aee76503d2e2/bin/hipcc -std=c++20 -O3 -DNDEBUG --offload-arch=gfx90a -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEBUG=0 -D__HIP_PLATFORM_AMD__ -IRESULT_DIR/src/head/include -Rpass-analysis=kernel-resource-usage -x hip -c REPRO_DIR/bench.cpp -o RESULT_DIR/bench-hip/head/gfx90a/bench.o
```

## Kernels

Each run launches 2^20 threads. n = 1024 for each thread, and the run-time offset is 6. The comments in the sources describe each kernel.

`bench.cpp` (18 kernels):

- K00: n times the constructor without an offset argument and one draw.
- K01: n times the constructor with a run-time offset and one draw; all threads have the same offset. K02: the same, with a different offset in each thread.
- K03: the constructor with a run-time offset, then n draws of `UniformUint<std::uint32_t>`. K04: n draws of `UniformUint<std::uint64_t>`. K17: the constructor without an offset argument, then n draws of `UniformUint<std::uint32_t>`.
- K05: n times a run-time `skip(k)` of `PhiloxSingle` and one draw (through a derived class, because `skip` is protected), for k = 1, 3 and 7.
- K06, K07, K08: n draws of the `UniformReal<float>`, `UniformReal<double>` and `NormalReal<float>` of the accelerator.
- K09, K10: n draws of `alpaka::rand::UniformReal<float>` and `<double>` in [0, 1) with `Philox4x32x10`. K11: n draws of `alpaka::rand::UniformReal<float>` in a run-time range.
- K12: n times the vector engine constructor with a run-time offset and one vector. K13, K14, K15: n / 4 vectors of `std::uint32_t`, of `UniformReal<float>` and of `UniformReal<double>` in [0, 1). K16: n times a skip of a 64-bit run-time number of vectors and one vector.

`bench2.cpp` (7 kernels, the limits [1.5, 4) and [0, 1) are run-time values):

- R0, R1: single-number engine, `float` and `double`. R2, R3: vector engine, `float` and `double`.
- R4: compile-time [0, 1), vector engine, `double` (the same kernel as K15).
- R5, R6: the distribution is a member of the kernel object, so the compiler does not know the limits [0, 1); single-number engine with `float`, and vector engine with `double`.

## Output: registers (`ptxas -v`), sm_75 / sm_80 / sm_90

From `metrics-cuda.tsv`:

| kernel | develop | a83bc065f | 0dee57950 | head |
|---|---|---|---|---|
| bench K00_EngCtorNoOffset | 20/20/20 | 18/18/18 | 18/18/18 | 18/18/18 |
| bench K01_EngCtor | 20/20/20 | 27/24/23 | 20/20/20 | 20/20/20 |
| bench K02_EngCtorVar | 21/20/20 | 27/25/24 | 20/20/20 | 20/20/20 |
| bench K03_EngUint32 | 21/20/20 | 21/20/26 | 22/20/27 | 22/20/27 |
| bench K04_EngUint64 | 20/22/22 | 22/20/28 | 24/20/28 | 24/20/28 |
| bench K05_EngSkip | 21/21/22 | 24/22/20 | 21/22/28 | 21/22/28 |
| bench K06_TraitRealF | 21/20/20 | 22/22/26 | 22/20/27 | 22/20/27 |
| bench K07_TraitRealD | 22/24/24 | 24/22/30 | 24/24/28 | 24/24/28 |
| bench K08_TraitNormF | 34/32/32 | 34/32/32 | 34/32/32 | 34/32/32 |
| bench K09_PhiloxRealF | 20/20/20 | 22/22/26 | 22/20/27 | 22/20/27 |
| bench K10_PhiloxRealD | 35/32/32 | 22/22/30 | 24/20/28 | 24/20/28 |
| bench K11_PhiloxRealFRange | 22/20/20 | 24/22/28 | 22/22/30 | 22/22/30 |
| bench K12_VecCtor | 19/17/17 | 19/20/20 | 19/20/20 | 19/20/20 |
| bench K13_VecUint32 | 20/18/18 | 19/20/20 | 19/20/20 | 19/20/20 |
| bench K14_VecRealF | 21/19/20 | 19/20/20 | 19/20/18 | 19/20/18 |
| bench K15_VecRealD | 22/24/22 | 23/26/24 | 22/22/22 | 22/22/22 |
| bench K16_VecSkip64 | 24/21/23 | 20/20/22 | 20/20/22 | 20/20/22 |
| bench K17_EngUint32NoOffset | 20/20/20 | 22/20/20 | 22/20/20 | 22/20/20 |
| bench2 R0_RealFRange | 22/20/20 | 24/22/28 | 22/22/30 | 22/22/30 |
| bench2 R1_RealDRange | 39/36/37 | 30/30/28 | 30/30/26 | 30/30/26 |
| bench2 R2_VecRealFRange | 21/21/20 | 21/22/20 | 21/22/20 | 21/22/20 |
| bench2 R3_VecRealDRange | 26/28/26 | 31/32/32 | 31/32/32 | 31/32/32 |
| bench2 R4_VecRealDUnit | 22/24/22 | 23/26/24 | 22/22/22 | 22/22/22 |
| bench2 R5_RealFMember | 21/20/20 | 22/22/26 | 20/19/26 | 20/19/26 |
| bench2 R6_VecRealDMember | 24/24/24 | 25/26/28 | 25/26/28 | 25/26/28 |

## Output: local memory, spills, stack

From `metrics-cuda.tsv` (`LOCAL` from `cuobjdump -res-usage`, spill and stack frame from `ptxas -v`, `LDL` and `STL` counted in `cuobjdump -sass`):

```
develop    sm75: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: [], LDL 0, STL 0
develop    sm80: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
develop    sm90: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
a83bc065f  sm75: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: [], LDL 0, STL 0
a83bc065f  sm80: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
a83bc065f  sm90: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
0dee57950  sm75: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: [], LDL 0, STL 0
0dee57950  sm80: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
0dee57950  sm90: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
head       sm75: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: [], LDL 0, STL 0
head       sm80: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
head       sm90: kernels 25, LOCAL>0: 0, spill bytes>0: 0, stack frame>0: ['bench K08_TraitNormF 32 B'], LDL 6, STL 8
```

## Output: `ptxas -v` lines of K01, K02 and K05, sm_75

The kernel names are cut after the kernel tag; the full lines are in `outputs/bench/<tree>/sm75/ptxas.log`.

develop:

```
ptxas info    : Compiling entry function 'K05_EngSkipNS_... for 'sm_75'
ptxas info    : Function properties for K05_EngSkipNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K02_EngCtorVarNS_... for 'sm_75'
ptxas info    : Function properties for K02_EngCtorVarNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K01_EngCtorNS_... for 'sm_75'
ptxas info    : Function properties for K01_EngCtorNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, used 0 barriers, 392 bytes cmem[0]
```

a83bc065f:

```
ptxas info    : Compiling entry function 'K05_EngSkipNS_... for 'sm_75'
ptxas info    : Function properties for K05_EngSkipNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K02_EngCtorVarNS_... for 'sm_75'
ptxas info    : Function properties for K02_EngCtorVarNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 27 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K01_EngCtorNS_... for 'sm_75'
ptxas info    : Function properties for K01_EngCtorNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 27 registers, used 0 barriers, 392 bytes cmem[0]
```

0dee57950:

```
ptxas info    : Compiling entry function 'K05_EngSkipNS_... for 'sm_75'
ptxas info    : Function properties for K05_EngSkipNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K02_EngCtorVarNS_... for 'sm_75'
ptxas info    : Function properties for K02_EngCtorVarNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K01_EngCtorNS_... for 'sm_75'
ptxas info    : Function properties for K01_EngCtorNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, used 0 barriers, 392 bytes cmem[0]
```

head:

```
ptxas info    : Compiling entry function 'K05_EngSkipNS_... for 'sm_75'
ptxas info    : Function properties for K05_EngSkipNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K02_EngCtorVarNS_... for 'sm_75'
ptxas info    : Function properties for K02_EngCtorVarNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'K01_EngCtorNS_... for 'sm_75'
ptxas info    : Function properties for K01_EngCtorNS_...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, used 0 barriers, 392 bytes cmem[0]
```

## Output: `cuobjdump -res-usage` of K01, K02 and K05, sm_75

develop:

```
 Function K05_EngSkipNS_...
  REG:21 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K02_EngCtorVarNS_...
  REG:21 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K01_EngCtorNS_...
  REG:20 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
```

a83bc065f:

```
 Function K05_EngSkipNS_...
  REG:24 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K02_EngCtorVarNS_...
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K01_EngCtorNS_...
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
```

0dee57950:

```
 Function K05_EngSkipNS_...
  REG:21 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K02_EngCtorVarNS_...
  REG:20 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K01_EngCtorNS_...
  REG:20 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
```

head:

```
 Function K05_EngSkipNS_...
  REG:21 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K02_EngCtorVarNS_...
  REG:20 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function K01_EngCtorNS_...
  REG:20 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
```

## Output: HIP resource usage

Summary over the 18 kernels of `bench.cpp`, from `metrics-hip.tsv`:

```
gfx90a   a83bc065f  VGPRs minus develop: min +0, median +6, max +12 (18 kernels); ScratchSize values ['0']; VGPR spill values ['0']; occupancy values ['8']
gfx90a   0dee57950  VGPRs minus develop: min +0, median +4, max +12 (18 kernels); ScratchSize values ['0']; VGPR spill values ['0']; occupancy values ['8']
gfx90a   head       VGPRs minus develop: min +0, median +4, max +12 (18 kernels); ScratchSize values ['0']; VGPR spill values ['0']; occupancy values ['8']
gfx1100  a83bc065f  VGPRs minus develop: min -1, median +3.5, max +8 (18 kernels); ScratchSize values ['0']; VGPR spill values ['0']; occupancy values ['16']
gfx1100  0dee57950  VGPRs minus develop: min -3, median +3, max +5 (18 kernels); ScratchSize values ['0']; VGPR spill values ['0']; occupancy values ['16']
gfx1100  head       VGPRs minus develop: min -3, median +3, max +5 (18 kernels); ScratchSize values ['0']; VGPR spill values ['0']; occupancy values ['16']
```

The remarks of K01 on gfx90a:

develop:

```
remark: Function Name: K01_EngCtorNS_... [-Rpass-analysis=kernel-resource-usage]
remark:     TotalSGPRs: 19 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs: 14 [-Rpass-analysis=kernel-resource-usage]
remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
```

a83bc065f:

```
remark: Function Name: K01_EngCtorNS_... [-Rpass-analysis=kernel-resource-usage]
remark:     TotalSGPRs: 25 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs: 22 [-Rpass-analysis=kernel-resource-usage]
remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
```

0dee57950:

```
remark: Function Name: K01_EngCtorNS_... [-Rpass-analysis=kernel-resource-usage]
remark:     TotalSGPRs: 22 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs: 18 [-Rpass-analysis=kernel-resource-usage]
remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
```

head:

```
remark: Function Name: K01_EngCtorNS_... [-Rpass-analysis=kernel-resource-usage]
remark:     TotalSGPRs: 22 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs: 18 [-Rpass-analysis=kernel-resource-usage]
remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
```

## Output: time on the T4

`timing.py` output. Each value is the median over 5 rounds of the median of 21 launches, and the ratio to `develop` in the same round (median, and in brackets the lowest and the highest of the 5 rounds). The numbers of `develop` and of the PR are not the same work: `develop` runs 11 rounds, ignores `offset % 4` in the constructor and uses other conversions in `UniformReal`.

`bench.cpp`:

```
kernel                                               develop                     a83bc065f                     0dee57950                          head
K00_EngCtorNoOffset              21.598 ms   +0.0% [ +0.0, +0.0]   18.835 ms  -12.6% [-12.9,-10.6]   18.837 ms  -11.6% [-12.9, -8.8]   18.873 ms  -11.4% [-12.9, -8.5]
K01_EngCtor                      20.704 ms   +0.0% [ +0.0, +0.0]   24.296 ms  +17.6% [+17.0,+20.0]   19.873 ms   -3.9% [ -4.0, -1.8]   19.694 ms   -3.9% [ -5.4, -0.7]
K02_EngCtorVar                   20.587 ms   +0.0% [ +0.0, +0.0]   29.967 ms  +46.4% [+44.6,+48.6]   21.275 ms   +3.0% [ +2.3, +5.3]   20.996 ms   +2.5% [ +2.0, +6.7]
K03_EngUint32                     7.809 ms   +0.0% [ +0.0, +0.0]    7.262 ms   -7.7% [ -9.1, -5.7]    7.235 ms   -7.3% [ -8.8, -4.4]    7.289 ms   -7.3% [ -8.8, -3.7]
K04_EngUint64                    14.655 ms   +0.0% [ +0.0, +0.0]   13.523 ms   -7.7% [ -8.1, -5.3]   13.232 ms   -9.0% [-10.1, -6.3]   13.260 ms   -8.9% [ -9.7, -5.1]
K05_EngSkip(k=1)                 19.311 ms   +0.0% [ +0.0, +0.0]   17.920 ms   -7.2% [ -8.4, -6.4]   15.489 ms  -19.8% [-19.8,-17.8]   15.488 ms  -19.8% [-19.8,-17.1]
K05_EngSkip(k=3)                 31.056 ms   +0.0% [ +0.0, +0.0]   30.771 ms   -0.9% [ -2.2, +0.9]   28.107 ms   -9.5% [-10.7, -6.3]   28.107 ms   -9.5% [ -9.5, -5.5]
K05_EngSkip(k=7)                 31.005 ms   +0.0% [ +0.0, +0.0]   30.771 ms   -0.8% [ -2.4, +0.6]   28.107 ms   -9.3% [-10.8, -6.6]   28.107 ms   -9.3% [ -9.9, -5.8]
K06_TraitRealF                    8.297 ms   +0.0% [ +0.0, +0.0]    7.901 ms   -4.8% [ -6.0, -2.3]    7.857 ms   -5.3% [ -6.6, -1.8]    7.862 ms   -5.2% [ -6.6, -1.8]
K07_TraitRealD                   23.428 ms   +0.0% [ +0.0, +0.0]   22.894 ms   -2.3% [ -3.3, -1.1]   22.640 ms   -2.2% [ -3.5, +0.1]   22.903 ms   -2.2% [ -2.4, +0.1]
K08_TraitNormF                   21.897 ms   +0.0% [ +0.0, +0.0]   21.478 ms   -1.8% [ -4.5, -1.3]   21.257 ms   -2.9% [ -4.8, -0.4]   21.319 ms   -2.6% [ -4.6, -0.3]
K09_PhiloxRealF                   8.147 ms   +0.0% [ +0.0, +0.0]    8.378 ms   +2.9% [ +1.1, +4.8]    7.698 ms   -4.8% [ -6.4, -2.4]    7.752 ms   -4.7% [ -6.5, -2.4]
K10_PhiloxRealD                 106.155 ms   +0.0% [ +0.0, +0.0]   39.854 ms  -62.5% [-62.5,-62.3]   20.185 ms  -81.0% [-81.3,-81.0]   20.184 ms  -81.0% [-81.3,-81.0]
K11_PhiloxRealFRange              8.507 ms   +0.0% [ +0.0, +0.0]    8.741 ms   +2.7% [ +2.1, +5.3]    8.745 ms   +3.5% [ +2.7, +5.6]    8.765 ms   +3.1% [ +2.8, +6.3]
K12_VecCtor                      24.605 ms   +0.0% [ +0.0, +0.0]   21.187 ms  -13.9% [-15.0,-13.0]   21.186 ms  -13.9% [-15.0,-11.5]   21.188 ms  -13.9% [-13.9,-10.7]
K13_VecUint32                     5.677 ms   +0.0% [ +0.0, +0.0]    5.102 ms   -9.1% [-11.7, -7.3]    5.124 ms   -9.7% [-11.4, -8.0]    5.116 ms   -9.9% [-10.6, -6.8]
K14_VecRealF                      6.265 ms   +0.0% [ +0.0, +0.0]    6.378 ms   +2.3% [ +1.4, +5.4]    5.719 ms   -8.8% [ -9.0, -6.6]    5.672 ms   -8.8% [ -9.5, -4.5]
K15_VecRealD                     29.623 ms   +0.0% [ +0.0, +0.0]   39.836 ms  +34.4% [+34.4,+34.9]   22.039 ms  -25.6% [-25.6,-25.3]   22.026 ms  -25.7% [-25.7,-25.4]
K16_VecSkip64(k=5)               25.803 ms   +0.0% [ +0.0, +0.0]   23.359 ms   -9.6% [-10.2, -8.2]   23.399 ms   -9.5% [-11.1, -8.2]   23.319 ms   -9.5% [-10.4, -7.4]
K17_EngUint32NoOffset             7.850 ms   +0.0% [ +0.0, +0.0]    7.330 ms   -6.6% [ -8.6, -6.6]    7.269 ms   -7.4% [ -8.1, -7.2]    7.267 ms   -7.4% [ -8.1, -5.9]
```

`bench2.cpp`:

```
kernel                                               develop                     a83bc065f                     0dee57950                          head
R0_RealFRange(1.5,4)              8.508 ms   +0.0% [ +0.0, +0.0]    8.714 ms   +2.5% [ +2.4, +4.2]    8.739 ms   +2.7% [ +1.0, +2.9]    8.739 ms   +2.7% [ +2.7, +3.1]
R0_RealFRange(0,1)                8.507 ms   +0.0% [ +0.0, +0.0]    8.859 ms   +4.2% [ +2.4, +5.1]    7.719 ms   -9.2% [ -9.4, -7.6]    7.720 ms   -8.6% [ -9.3, -8.4]
R1_RealDRange(1.5,4)            103.932 ms   +0.0% [ +0.0, +0.0]   49.481 ms  -52.4% [-52.4,-52.4]   49.478 ms  -52.4% [-52.4,-52.4]   49.475 ms  -52.4% [-52.4,-52.4]
R1_RealDRange(0,1)              103.911 ms   +0.0% [ +0.0, +0.0]   49.488 ms  -52.4% [-52.4,-52.3]   20.221 ms  -80.5% [-80.5,-80.5]   20.223 ms  -80.5% [-80.5,-80.5]
R2_VecRealFRange(1.5,4)           6.312 ms   +0.0% [ +0.0, +0.0]    6.550 ms   +3.8% [ +3.7, +3.8]    6.551 ms   +3.8% [ +3.7, +3.8]    6.551 ms   +3.8% [ +3.8, +3.8]
R2_VecRealFRange(0,1)             6.265 ms   +0.0% [ +0.0, +0.0]    6.509 ms   +3.9% [ +3.3, +4.8]    5.719 ms   -8.7% [-10.3, -8.4]    5.716 ms   -8.8% [-10.3, -8.5]
R3_VecRealDRange(1.5,4)          28.938 ms   +0.0% [ +0.0, +0.0]   49.512 ms  +71.1% [+70.8,+71.2]   49.495 ms  +71.0% [+70.8,+71.1]   49.488 ms  +71.0% [+70.9,+71.1]
R3_VecRealDRange(0,1)            28.939 ms   +0.0% [ +0.0, +0.0]   49.510 ms  +71.1% [+71.0,+71.2]   22.010 ms  -23.9% [-24.0,-23.9]   22.048 ms  -23.8% [-23.8,-23.7]
R4_VecRealDUnit                  29.635 ms   +0.0% [ +0.0, +0.0]   39.846 ms  +34.5% [+34.4,+34.6]   22.034 ms  -25.6% [-25.7,-25.6]   22.027 ms  -25.7% [-25.7,-25.6]
R5_RealFMember                    8.583 ms   +0.0% [ +0.0, +0.0]    8.772 ms   +1.1% [ -1.0, +2.4]    7.735 ms   -9.9% [ -9.9, -9.6]    7.732 ms   -9.9% [-11.4, -8.4]
R6_VecRealDMember                29.655 ms   +0.0% [ +0.0, +0.0]   49.939 ms  +68.5% [+68.2,+68.5]   22.057 ms  -25.6% [-25.7,-25.6]   22.057 ms  -25.6% [-25.7,-25.6]
```

GPU state before and after the `bench.cpp` timing (`nvidia-smi`):

```
index, name, temperature.gpu, clocks.current.graphics [MHz]
0, Tesla T4, 41, 300 MHz
1, Tesla T4, 33, 300 MHz
index, name, temperature.gpu, clocks.current.graphics [MHz]
0, Tesla T4, 34, 300 MHz
1, Tesla T4, 69, 1575 MHz
```

## Files in `outputs/`


- `trees.txt`, `versions.txt`, `metrics.txt`, `metrics-cuda.tsv`, `metrics-hip.tsv`, `timing-bench.txt`, `timing-bench2.txt`
- `bench/<tree>/sm<NN>/` and `bench2/<tree>/sm<NN>/`: `compile.cmd`, `ptxas.log`, `resusage.txt`, `sass.txt.gz`
- `bench-hip/<tree>/<gfx>/`: `compile.cmd`, `compile.log`
- `timing-bench/` and `timing-bench2/`: the log of each run (`<tree>.r<N>.log`), `preheat.log`, `gpu-before.txt`, `gpu-after.txt`, `summary.tsv`
