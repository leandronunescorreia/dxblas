#include <dxblas/dxblas.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

int main()
{
    constexpr std::size_t count = 1u << 22; // 4,194,304 FP32 elements per input.
    std::vector<float> cpu_a(count), cpu_b(count), cpu_result(count), gpu_result(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < count; ++i) {
        cpu_a[i] = dist(rng);
        cpu_b[i] = dist(rng);
        cpu_result[i] = cpu_a[i] + cpu_b[i];
    }

    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Context::initialize failed: 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }

    dxblas::gpuMem<float> a(cpu_a), b(cpu_b), out(count);
    const auto upload_start = Clock::now();
    hr = context.copy(a, b, out, count);
    const auto upload_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy failed: 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 2;
    }

    const auto sum_start = Clock::now();
    hr = context.sum(a, b, out, count);
    const auto sum_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::sum(gpuMem) failed: 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 3;
    }

    const auto read_start = Clock::now();
    hr = context.copy_to_cpu(out, gpu_result.data(), count);
    const auto read_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "copy_to_cpu failed: 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 4;
    }

    std::size_t mismatches = 0;
    float max_error = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float error = std::fabs(cpu_result[i] - gpu_result[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-5f) ++mismatches;
    }
    const auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    std::cout << "dxblas_big_array: gpuMem FP32 sum\n"
              << "Elements per input: " << count << " (" << std::fixed << std::setprecision(2)
              << (count * sizeof(float) / (1024.0 * 1024.0)) << " MiB each)\n"
              << "CPU -> GPU: " << ms(upload_end - upload_start) << " ms\n"
              << "GPU sum dispatch + completion: " << ms(sum_end - sum_start) << " ms\n"
              << "GPU -> CPU: " << ms(read_end - read_start) << " ms\n"
              << "Mismatches: " << mismatches << ", max abs error: " << max_error << '\n';
    return mismatches == 0 ? 0 : 5;
}
