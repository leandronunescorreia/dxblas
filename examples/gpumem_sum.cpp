#include <dxblas/dxblas.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

static void print_hresult(const char* operation, HRESULT hr) {
    std::cerr << operation << " failed: HRESULT 0x" << std::hex << static_cast<unsigned long>(hr)
              << std::dec << '\n';
}

int main() {
    constexpr std::size_t count = 8192 * 8192;
    constexpr std::uint32_t seed = 0xD1B1A5u;
    std::cout << "DXBLAS gpuMem FP32 sum vs C++ CPU sum\n"
              << "Elements per input: " << count << " ("
              << (count * sizeof(float)) / (1024.0 * 1024.0) << " MiB each)\n";

    std::vector<float> cpu_a(count), cpu_b(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
    for (std::size_t i = 0; i < count; ++i) {
        cpu_a[i] = distribution(rng);
        cpu_b[i] = distribution(rng);
    }

    // CPU baseline: measure only the arithmetic over already-resident arrays.
    std::vector<float> cpu_result(count);
    const auto cpu_start = Clock::now();
    for (std::size_t i = 0; i < count; ++i)
        cpu_result[i] = cpu_a[i] + cpu_b[i];
    const auto cpu_end = Clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("D3D11 initialize", hr);
        return 1;
    }

    // gpuMem owns the host payload until copy() uploads the inputs. The backend
    // stores GPU texture handles in these objects after the upload succeeds.
    dxblas::gpuMem<float> a(cpu_a);
    dxblas::gpuMem<float> b(cpu_b);
    dxblas::gpuMem<float> out(count);

    const auto upload_start = Clock::now();
    hr = context.copy(a, b, out, count);
    const auto upload_end = Clock::now();
    if (FAILED(hr)) {
        print_hresult("CPU -> GPU copy", hr);
        return 2;
    }

    const auto gpu_sum_start = Clock::now();
    hr = context.sum(a, b, out, count);
    const auto gpu_sum_end = Clock::now();
    if (FAILED(hr)) {
        print_hresult("GPU pixel-shader sum", hr);
        return 3;
    }

    std::vector<float> gpu_result(count);
    const auto readback_start = Clock::now();
    hr = context.copy_to_cpu(out, gpu_result.data(), count);
    const auto readback_end = Clock::now();
    if (FAILED(hr)) {
        print_hresult("GPU -> CPU copy", hr);
        return 4;
    }

    std::size_t mismatches = 0;
    float max_abs_error = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float error = std::fabs(cpu_result[i] - gpu_result[i]);
        if (error > max_abs_error)
            max_abs_error = error;
        if (error > 1e-5f) {
            if (mismatches < 5)
                std::cerr << "Mismatch at " << i << ": CPU=" << cpu_result[i]
                          << ", GPU=" << gpu_result[i] << '\n';
            ++mismatches;
        }
    }

    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
    const double sum_ms =
        std::chrono::duration<double, std::milli>(gpu_sum_end - gpu_sum_start).count();
    const double readback_ms =
        std::chrono::duration<double, std::milli>(readback_end - readback_start).count();
    const double gpu_total_ms =
        std::chrono::duration<double, std::milli>(readback_end - upload_start).count();

    std::cout << std::fixed << std::setprecision(3)
              << "\nCPU C++ sum (arithmetic only) : " << cpu_ms << " ms\n"
              << "GPU upload (two inputs)       : " << upload_ms << " ms\n"
              << "GPU sum submission            : " << sum_ms << " ms\n"
              << "GPU readback (result)         : " << readback_ms << " ms\n"
              << "GPU end-to-end total          : " << gpu_total_ms << " ms\n"
              << "CPU arithmetic / GPU end-to-end: "
              << (gpu_total_ms > 0.0 ? cpu_ms / gpu_total_ms : 0.0) << "x\n"
              << "Maximum absolute error        : " << max_abs_error << '\n'
              << "Mismatched elements           : " << mismatches << '\n';

    if (mismatches != 0) {
        std::cerr << "Validation FAILED\n";
        return 5;
    }
    std::cout << "Validation PASSED\n";
    std::cout << "Note: GPU sum submission measures command submission; readback synchronizes "
                 "completion.\n";
    return 0;
}
