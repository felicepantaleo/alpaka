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


// Run-time limits, single engine, float.
struct R0_RealFRange
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
        out[tid] = toBits(sum) + k;
    }
};

// Run-time limits, single engine, double.
struct R1_RealDRange
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10 engine(seed, tid, offset);
        alpaka::rand::UniformReal<double> dist(static_cast<double>(lo), static_cast<double>(hi));
        double sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += dist(engine);
        }
        out[tid] = toBits(sum) + k;
    }
};

// Run-time limits, vector engine, float.
struct R2_VecRealFRange
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset);
        alpaka::rand::UniformReal<alpaka::rand::Philox4x32x10Vector::ResultContainer<float>> dist(lo, hi);
        float sum = 0;
        for(std::uint32_t i = 0; i < n / 4; ++i)
        {
            auto const v = dist(engine);
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = toBits(sum) + k;
    }
};

// Run-time limits, vector engine, double.
struct R3_VecRealDRange
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset);
        alpaka::rand::UniformReal<alpaka::rand::Philox4x32x10Vector::ResultContainer<double>> dist(
            static_cast<double>(lo),
            static_cast<double>(hi));
        double sum = 0;
        for(std::uint32_t i = 0; i < n / 4; ++i)
        {
            auto const v = dist(engine);
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = toBits(sum) + k;
    }
};

// Compile-time [0, 1), vector engine, double: the control kernel (K15 of the first study).
struct R4_VecRealDUnit
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
        out[tid] = toBits(sum) + k + static_cast<std::uint32_t>(lo + hi) * 0u;
    }
};

// The distribution is a member of the kernel object: the limits are [0, 1) but the compiler does not know them.
struct R5_RealFMember
{
    alpaka::rand::UniformReal<float> dist;

    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10 engine(seed, tid, offset);
        auto local = dist;
        float sum = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            sum += local(engine);
        }
        out[tid] = toBits(sum) + k + static_cast<std::uint32_t>(lo + hi) * 0u;
    }
};

// The same with the vector engine and double.
struct R6_VecRealDMember
{
    alpaka::rand::UniformReal<alpaka::rand::Philox4x32x10Vector::ResultContainer<double>> dist;

    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(KERNEL_ARGS) const
    {
        auto const tid = threadIndex(acc);
        alpaka::rand::Philox4x32x10Vector engine(seed, tid, offset);
        auto local = dist;
        double sum = 0;
        for(std::uint32_t i = 0; i < n / 4; ++i)
        {
            auto const v = local(engine);
            sum += v[0] + v[1] + v[2] + v[3];
        }
        out[tid] = toBits(sum) + k + static_cast<std::uint32_t>(lo + hi) * 0u;
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
void run(
    char const* name,
    TKernel const& kernel,
    TQueue& queue,
    TDev const& dev,
    Settings const& s,
    float const lo,
    float const hi)
{
    using Vec = alpaka::Vec<Dim, Idx>;
    std::uint32_t const threadsPerBlock = 256;
    alpaka::WorkDivMembers<Dim, Idx> const workDiv{Vec{s.threads / threadsPerBlock}, Vec{threadsPerBlock}, Vec{Idx{1}}};
    auto buffer = alpaka::allocBuf<std::uint32_t, Idx>(dev, Vec{s.threads});
    alpaka::memset(queue, buffer, 0);
    auto* const out = std::data(buffer);
    std::uint32_t const k = 0;

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
        "%-26s k=%-4u median_ms=%9.4f min_ms=%9.4f max_ms=%9.4f launches=%d checksum=%llu\n",
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
    std::printf("device=%s threads=%u draws=%u offset=%u\n", alpaka::getName(dev).c_str(), s.threads, s.draws, s.offset);

    // Limits [1.5, 4): the general path. Limits [0, 1) at run time: the shortcut, decided at run time.
    run("R0_RealFRange(1.5,4)", R0_RealFRange{}, queue, dev, s, 1.5f, 4.0f);
    run("R0_RealFRange(0,1)", R0_RealFRange{}, queue, dev, s, 0.0f, 1.0f);
    run("R1_RealDRange(1.5,4)", R1_RealDRange{}, queue, dev, s, 1.5f, 4.0f);
    run("R1_RealDRange(0,1)", R1_RealDRange{}, queue, dev, s, 0.0f, 1.0f);
    run("R2_VecRealFRange(1.5,4)", R2_VecRealFRange{}, queue, dev, s, 1.5f, 4.0f);
    run("R2_VecRealFRange(0,1)", R2_VecRealFRange{}, queue, dev, s, 0.0f, 1.0f);
    run("R3_VecRealDRange(1.5,4)", R3_VecRealDRange{}, queue, dev, s, 1.5f, 4.0f);
    run("R3_VecRealDRange(0,1)", R3_VecRealDRange{}, queue, dev, s, 0.0f, 1.0f);
    run("R4_VecRealDUnit", R4_VecRealDUnit{}, queue, dev, s, 0.0f, 1.0f);
    run("R5_RealFMember", R5_RealFMember{}, queue, dev, s, 0.0f, 1.0f);
    run("R6_VecRealDMember", R6_VecRealDMember{}, queue, dev, s, 0.0f, 1.0f);
    return 0;
}
