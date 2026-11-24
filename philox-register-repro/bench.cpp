// Register, local memory and timing benchmark of the alpaka Philox engines.
// The same source compiles against the develop tree and the branch tree.
// Every kernel has the same argument list. Seeds, offsets and counts are run-time std::uint32_t values.

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <cuda_runtime.h>
#    define GPU_EVENT cudaEvent_t
#    define GPU_EVENT_CREATE cudaEventCreate
#    define GPU_EVENT_RECORD cudaEventRecord
#    define GPU_EVENT_SYNC cudaEventSynchronize
#    define GPU_EVENT_ELAPSED cudaEventElapsedTime
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <hip/hip_runtime.h>
#    define GPU_EVENT hipEvent_t
#    define GPU_EVENT_CREATE hipEventCreate
#    define GPU_EVENT_RECORD hipEventRecord
#    define GPU_EVENT_SYNC hipEventSynchronize
#    define GPU_EVENT_ELAPSED hipEventElapsedTime
#endif

using Dim = alpaka::DimInt<1u>;
using Idx = std::uint32_t;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#else
using Acc = alpaka::AccGpuHipRt<Dim, Idx>;
#endif

using Params = alpaka::rand::engine::PhiloxParams<4, 32, 10>;

// Makes the protected PhiloxSingle::skip callable from a kernel.
struct SkipEngine : alpaka::rand::engine::PhiloxSingle<Params>
{
    using Base = alpaka::rand::engine::PhiloxSingle<Params>;
    ALPAKA_FN_HOST_ACC SkipEngine(std::uint64_t seed, std::uint64_t subsequence, std::uint64_t offset)
        : Base(seed, subsequence, offset)
    {
    }
    using Base::skip;
};

// Makes the protected PhiloxVector::skip callable from a kernel.
struct SkipVectorEngine : alpaka::rand::engine::PhiloxVector<Params>
{
    using Base = alpaka::rand::engine::PhiloxVector<Params>;
    ALPAKA_FN_HOST_ACC SkipVectorEngine(std::uint64_t seed, std::uint64_t subsequence, std::uint64_t offset)
        : Base(seed, subsequence, offset)
    {
    }
    using Base::skip;
};

template<typename TAcc>
ALPAKA_FN_ACC auto threadIndex(TAcc const& acc) -> std::uint32_t
{
    return alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
}

ALPAKA_FN_HOST_ACC inline auto toBits(float const sum) -> std::uint32_t
{
    return static_cast<std::uint32_t>(static_cast<std::int32_t>(sum * 1.0e4f));
}

ALPAKA_FN_HOST_ACC inline auto toBits(double const sum) -> std::uint32_t
{
    return static_cast<std::uint32_t>(static_cast<std::int32_t>(sum * 1.0e4));
}

#define KERNEL_ARGS                                                                                                   \
    TAcc const& acc, std::uint32_t* out, std::uint32_t seed, std::uint32_t offset, std::uint32_t n, std::uint32_t k,  \
        float lo, float hi

// (a) n times: constructor without an offset argument, one draw.
struct K00_EngCtorNoOffset
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto dist = alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            auto engine = alpaka::rand::engine::createDefault(acc, seed + i, tid);
            sum += dist(engine);
        }
        out[tid] = sum;
    }
};

// (a) n times: constructor with a run-time offset, one draw. All threads have the same offset.
struct K01_EngCtor
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto dist = alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset + i);
            sum += dist(engine);
        }
        out[tid] = sum;
    }
};

// (a) n times: constructor with a run-time offset, one draw. The offset is different in each thread.
struct K02_EngCtorVar
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto dist = alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset + i + (tid & 7u));
            sum += dist(engine);
        }
        out[tid] = sum;
    }
};

// (a) Constructor with a run-time offset, n draws of UniformUint<std::uint32_t>.
struct K03_EngUint32
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset);
        auto dist = alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = sum;
    }
};

// n draws of UniformUint<std::uint64_t>.
struct K04_EngUint64
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset);
        auto dist = alpaka::rand::distribution::createUniformUint<std::uint64_t>(acc);
        std::uint64_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = static_cast<std::uint32_t>(sum) ^ static_cast<std::uint32_t>(sum >> 32);
    }
};

// (b) n times: run-time skip(k), then one draw.
struct K05_EngSkip
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        SkipEngine engine(seed, tid, offset);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            engine.skip(k);
            sum += engine();
        }
        out[tid] = sum;
    }
};

// (c) n draws of the UniformReal<float> of the accelerator.
struct K06_TraitRealF
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset);
        auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
        float sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum);
    }
};

// (c) n draws of the UniformReal<double> of the accelerator.
struct K07_TraitRealD
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset);
        auto dist = alpaka::rand::distribution::createUniformReal<double>(acc);
        double sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum);
    }
};

// (c) n draws of the NormalReal<float> of the accelerator.
struct K08_TraitNormF
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto engine = alpaka::rand::engine::createDefault(acc, seed, tid, offset);
        auto dist = alpaka::rand::distribution::createNormalReal<float>(acc);
        float sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum);
    }
};

// (c) n draws of alpaka::rand::UniformReal<float> in [0, 1) with Philox4x32x10.
struct K09_PhiloxRealF
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10 engine(seed, tid, offset);
        alpaka::rand::UniformReal<float> dist;
        float sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum);
    }
};

// (c) n draws of alpaka::rand::UniformReal<double> in [0, 1) with Philox4x32x10.
struct K10_PhiloxRealD
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10 engine(seed, tid, offset);
        alpaka::rand::UniformReal<double> dist;
        double sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum);
    }
};

// (c) n draws of alpaka::rand::UniformReal<float> in a run-time range with Philox4x32x10.
struct K11_PhiloxRealFRange
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10 engine(seed, tid, offset);
        alpaka::rand::UniformReal<float> dist(lo, hi);
        float sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum);
    }
};

// (d) n times: vector engine constructor with a run-time offset, one vector.
struct K12_VecCtor
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset + i);
            auto const v = engine();
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = sum;
    }
};

// (d) Vector engine, n / 4 vectors of std::uint32_t.
struct K13_VecUint32
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n / 4; ++i)
        {
            auto const v = engine();
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = sum;
    }
};

// (d) Vector engine, n / 4 vectors of alpaka::rand::UniformReal<float> in [0, 1).
struct K14_VecRealF
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset);
        alpaka::rand::UniformReal<alpaka::rand::Philox4x32x10Vector::ResultContainer<float>> dist;
        float sum = 0;
        for(std::uint32_t i = 0; i < n / 4; ++i)
        {
            auto const v = dist(engine);
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = toBits(sum);
    }
};

// (d) Vector engine, n / 4 vectors of alpaka::rand::UniformReal<double> in [0, 1).
struct K15_VecRealD
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset);
        alpaka::rand::UniformReal<alpaka::rand::Philox4x32x10Vector::ResultContainer<double>> dist;
        double sum = 0;
        for(std::uint32_t i = 0; i < n / 4; ++i)
        {
            auto const v = dist(engine);
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = toBits(sum);
    }
};

// (d) Vector engine, n times: skip of a 64-bit run-time number of vectors, then one vector.
// The skip is different in each thread, so that the carry into counter word 2 occurs in different iterations.
struct K16_VecSkip64
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        SkipVectorEngine engine(seed, tid, offset);
        std::uint64_t const skip = (static_cast<std::uint64_t>(0x30000000u + (tid << 20)) << 32) | k;
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            engine.skip(skip);
            auto const v = engine();
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = sum;
    }
};

// Constructor without an offset argument, n draws of UniformUint<std::uint32_t>.
struct K17_EngUint32NoOffset
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        auto engine = alpaka::rand::engine::createDefault(acc, seed, tid);
        auto dist = alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc);
        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = sum;
    }
};

struct Settings
{
    std::uint32_t threads = 1u << 20;
    std::uint32_t draws = 256;
    std::uint32_t seed = 12345;
    std::uint32_t offset = 6;
    float warmupMs = 300.f;
    int launches = 21;
};

template<typename TKernel, typename TQueue, typename TDev>
void run(char const* name, TKernel const& kernel, TQueue& queue, TDev const& dev, Settings const& s, std::uint32_t k)
{
    using Vec = alpaka::Vec<Dim, Idx>;
    std::uint32_t const threadsPerBlock = 256;
    alpaka::WorkDivMembers<Dim, Idx> const workDiv{Vec{s.threads / threadsPerBlock}, Vec{threadsPerBlock}, Vec{Idx{1}}};
    auto buffer = alpaka::allocBuf<std::uint32_t, Idx>(dev, Vec{s.threads});
    alpaka::memset(queue, buffer, 0);
    auto* const out = std::data(buffer);
    float const lo = 1.5f;
    float const hi = 4.0f;

    auto const stream = queue.getNativeHandle();
    GPU_EVENT start;
    GPU_EVENT stop;
    GPU_EVENT_CREATE(&start);
    GPU_EVENT_CREATE(&stop);
    std::vector<float> times;
    // Warm-up: launch until the kernel has run for warmupMs, three launches or more.
    float warm = 0;
    for(int i = 0; i < 3 || warm < s.warmupMs; ++i)
    {
        GPU_EVENT_RECORD(start, stream);
        alpaka::exec<Acc>(queue, workDiv, kernel, out, s.seed, s.offset, s.draws, k, lo, hi);
        GPU_EVENT_RECORD(stop, stream);
        GPU_EVENT_SYNC(stop);
        float ms = 0;
        GPU_EVENT_ELAPSED(&ms, start, stop);
        warm += ms;
    }
    for(int i = 0; i < s.launches; ++i)
    {
        GPU_EVENT_RECORD(start, stream);
        alpaka::exec<Acc>(queue, workDiv, kernel, out, s.seed, s.offset, s.draws, k, lo, hi);
        GPU_EVENT_RECORD(stop, stream);
        GPU_EVENT_SYNC(stop);
        float ms = 0;
        GPU_EVENT_ELAPSED(&ms, start, stop);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());

    auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto host = alpaka::allocBuf<std::uint32_t, Idx>(devHost, Vec{s.threads});
    alpaka::memcpy(queue, host, buffer);
    alpaka::wait(queue);
    std::uint64_t checksum = 0;
    auto const* const data = std::data(host);
    for(std::uint32_t i = 0; i < s.threads; ++i)
    {
        checksum += data[i];
    }
    std::printf(
        "%-22s k=%-4u median_ms=%9.4f min_ms=%9.4f max_ms=%9.4f launches=%d checksum=%llu\n",
        name,
        k,
        static_cast<double>(times[times.size() / 2]),
        static_cast<double>(times.front()),
        static_cast<double>(times.back()),
        s.launches,
        static_cast<unsigned long long>(checksum));
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    Settings s;
    if(argc > 1)
        s.threads = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
    if(argc > 2)
        s.draws = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
    if(argc > 3)
        s.launches = std::atoi(argv[3]);
    if(argc > 4)
        s.offset = static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10));

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    alpaka::Queue<Acc, alpaka::NonBlocking> queue{dev};
    std::printf(
        "device=%s threads=%u draws=%u offset=%u warmup_ms=%.0f launches=%d\n",
        alpaka::getName(dev).c_str(),
        s.threads,
        s.draws,
        s.offset,
        static_cast<double>(s.warmupMs),
        s.launches);

    run("K00_EngCtorNoOffset", K00_EngCtorNoOffset{}, queue, dev, s, 0);
    run("K01_EngCtor", K01_EngCtor{}, queue, dev, s, 0);
    run("K02_EngCtorVar", K02_EngCtorVar{}, queue, dev, s, 0);
    run("K03_EngUint32", K03_EngUint32{}, queue, dev, s, 0);
    run("K04_EngUint64", K04_EngUint64{}, queue, dev, s, 0);
    run("K05_EngSkip", K05_EngSkip{}, queue, dev, s, 1);
    run("K05_EngSkip", K05_EngSkip{}, queue, dev, s, 3);
    run("K05_EngSkip", K05_EngSkip{}, queue, dev, s, 7);
    run("K06_TraitRealF", K06_TraitRealF{}, queue, dev, s, 0);
    run("K07_TraitRealD", K07_TraitRealD{}, queue, dev, s, 0);
    run("K08_TraitNormF", K08_TraitNormF{}, queue, dev, s, 0);
    run("K09_PhiloxRealF", K09_PhiloxRealF{}, queue, dev, s, 0);
    run("K10_PhiloxRealD", K10_PhiloxRealD{}, queue, dev, s, 0);
    run("K11_PhiloxRealFRange", K11_PhiloxRealFRange{}, queue, dev, s, 0);
    run("K12_VecCtor", K12_VecCtor{}, queue, dev, s, 0);
    run("K13_VecUint32", K13_VecUint32{}, queue, dev, s, 0);
    run("K14_VecRealF", K14_VecRealF{}, queue, dev, s, 0);
    run("K15_VecRealD", K15_VecRealD{}, queue, dev, s, 0);
    run("K16_VecSkip64", K16_VecSkip64{}, queue, dev, s, 5);
    run("K17_EngUint32NoOffset", K17_EngUint32NoOffset{}, queue, dev, s, 0);
    return 0;
}
