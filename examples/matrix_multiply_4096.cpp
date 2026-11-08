#include <dxblas/dxblas.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

static void cpu_matrix_multiply_blocked(const std::vector<float>& a, const std::vector<float>& b,
                                        std::vector<float>& c, std::size_t n) {
    std::fill(c.begin(), c.end(), 0.0f);
    constexpr std::size_t block = 32;
    for (std::size_t ii = 0; ii < n; ii += block)
        for (std::size_t kk = 0; kk < n; kk += block)
            for (std::size_t jj = 0; jj < n; jj += block)
                for (std::size_t i = ii; i < std::min(ii + block, n); ++i)
                    for (std::size_t k = kk; k < std::min(kk + block, n); ++k) {
                        const float av = a[i * n + k];
                        for (std::size_t j = jj; j < std::min(jj + block, n); ++j)
                            c[i * n + j] += av * b[k * n + j];
                    }
}

int main(int argc, char** argv) {
    int total = 2048;
    std::size_t n = total;
    if (argc > 1) {
        try {
            n = static_cast<std::size_t>(std::stoull(argv[1]));
        } catch (...) {
            std::cerr << "Usage: dxblas_matrix_multiply_4096 [matrix-size]\n";
            return 2;
        }
    }
    if (n == 0 || n > total) {
        std::cerr << "Matrix size must be in [1, " << total << "].\n";
        return 2;
    }

    const std::size_t elements = n * n;
    std::vector<float> a(elements), b(elements), cpu_result(elements), gpu_result(elements);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    std::cout << "dxblas_matrix_multiply_4096: gpuMem resident GEMM vs blocked C++\n"
              << "Matrix: " << n << " x " << n
              << ", each input: " << (elements * sizeof(float) / (1024.0 * 1024.0)) << " MiB\n"
              << "GPU path: D3D11 pixel shader + render target; no compute shader\n";

    const auto cpu_start = Clock::now();
    cpu_matrix_multiply_blocked(a, b, cpu_result, n);
    const auto cpu_end = Clock::now();
    std::cout << "CPU blocked GEMM: " << milliseconds(cpu_end - cpu_start) << " ms\n";

    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Context::initialize failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }

    dxblas::gpuMem<float> gpu_a(a), gpu_b(b), gpu_c(elements);
    gpu_a.width = gpu_b.width = gpu_c.width = static_cast<UINT>(n);
    gpu_a.height = gpu_b.height = gpu_c.height = static_cast<UINT>(n);
    const auto upload_start = Clock::now();
    hr = context.copy(gpu_a, gpu_b, gpu_c, elements);
    const auto upload_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const auto gpu_start = Clock::now();
    hr = context.gemm(gpu_a, gpu_b, gpu_c, n, n, n);
    const auto gpu_compute_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::gemm(gpuMem) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    hr = context.copy_to_cpu(gpu_c, gpu_result.data(), elements);
    const auto readback_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy_to_cpu failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }

    std::size_t mismatches = 0;
    float max_error = 0.0f;
    const std::size_t checks = std::min<std::size_t>(elements, 8192);
    for (std::size_t i = 0; i < checks; ++i) {
        const std::size_t index = elements <= checks ? i : i * elements / checks;
        const float error = std::fabs(cpu_result[index] - gpu_result[index]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(gpu_result[index]) ||
            error > 2.0e-3f + 2.0e-4f * std::fabs(cpu_result[index]))
            ++mismatches;
    }
    std::cout << "CPU -> GPU: " << milliseconds(upload_end - upload_start) << " ms\n"
              << "GPU-resident GEMM: " << milliseconds(gpu_compute_end - gpu_start) << " ms\n"
              << "GPU -> CPU: " << milliseconds(readback_end - gpu_compute_end) << " ms\n"
              << "Validation: " << checks << " sampled cells, mismatches=" << mismatches
              << ", max abs error=" << std::scientific << max_error << std::defaultfloat << '\n';
    return mismatches == 0 ? 0 : 3;
}
