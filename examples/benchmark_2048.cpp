#include <dxblas/dxblas.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main()
{
    constexpr std::size_t count = 2048u * 2048u;
    std::vector<float> a_cpu(count), b_cpu(count), cpu_sum(count), cpu_mul(count), gpu_result(count);
    std::mt19937 rng(2048);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (std::size_t i = 0; i < count; ++i) {
        a_cpu[i] = dist(rng);
        b_cpu[i] = dist(rng);
    }

    auto cpu_start = Clock::now();
    for (std::size_t i = 0; i < count; ++i) cpu_sum[i] = a_cpu[i] + b_cpu[i];
    auto cpu_sum_end = Clock::now();
    for (std::size_t i = 0; i < count; ++i) cpu_mul[i] = a_cpu[i] * b_cpu[i];
    auto cpu_mul_end = Clock::now();

    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Context::initialize failed: 0x" << std::hex << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }

    dxblas::gpuMem<float> a(a_cpu), b(b_cpu), out(count);
    auto upload_start = Clock::now();
    hr = context.copy(a, b, out, count);
    auto upload_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy failed: 0x" << std::hex << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 2;
    }

    auto gpu_sum_start = Clock::now();
    hr = context.sum(a, b, out, count);
    auto gpu_sum_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::sum(gpuMem) failed: 0x" << std::hex << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 3;
    }
    hr = context.copy_to_cpu(out, gpu_result.data(), count);
    if (FAILED(hr)) return 4;
    std::size_t sum_mismatches = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (std::fabs(cpu_sum[i] - gpu_result[i]) > 1e-5f) ++sum_mismatches;

    auto gpu_mul_start = Clock::now();
    hr = context.mul(a, b, out, count);
    auto gpu_mul_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::mul(gpuMem) failed: 0x" << std::hex << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 5;
    }
    hr = context.copy_to_cpu(out, gpu_result.data(), count);
    if (FAILED(hr)) return 6;
    std::size_t mul_mismatches = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (std::fabs(cpu_mul[i] - gpu_result[i]) > 1e-5f) ++mul_mismatches;

    std::cout << "dxblas_benchmark_2048: gpuMem-resident FP32 operations\n"
              << "Elements: " << count << ", bytes/input: " << (count * sizeof(float)) << '\n'
              << std::fixed << std::setprecision(3)
              << "CPU sum: " << ms(cpu_sum_end - cpu_start) << " ms\n"
              << "GPU upload A/B: " << ms(upload_end - upload_start) << " ms\n"
              << "GPU sum dispatch + completion: " << ms(gpu_sum_end - gpu_sum_start) << " ms\n"
              << "CPU mul: " << ms(cpu_mul_end - cpu_sum_end) << " ms\n"
              << "GPU mul dispatch + completion: " << ms(gpu_mul_end - gpu_mul_start) << " ms\n"
              << "Sum mismatches: " << sum_mismatches << '\n'
              << "Mul mismatches: " << mul_mismatches << '\n';
    return (sum_mismatches == 0 && mul_mismatches == 0) ? 0 : 7;
}
