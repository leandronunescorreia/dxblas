#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <dxblas/dxblas.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <random>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Timing {
    double milliseconds;
};

Timing elapsed(Clock::time_point start, Clock::time_point end)
{
    return {std::chrono::duration<double, std::milli>(end - start).count()};
}

void cpu_gemm(const std::vector<float>& a,
              const std::vector<float>& b,
              std::vector<float>& c,
              std::size_t m, std::size_t n, std::size_t k)
{
    std::fill(c.begin(), c.end(), 0.0f);
    const unsigned reported = std::thread::hardware_concurrency();
    const unsigned thread_count = std::max(1u, std::min(reported == 0 ? 1u : reported,
        static_cast<unsigned>((m + 15) / 16)));
    constexpr std::size_t rows_per_task = 16;
    std::atomic<std::size_t> next_row{0};
    auto worker = [&]() {
        for (;;) {
            const std::size_t row_start = next_row.fetch_add(rows_per_task);
            if (row_start >= m) break;
            const std::size_t row_end = std::min(m, row_start + rows_per_task);
            for (std::size_t i = row_start; i < row_end; ++i) {
                float* c_row = c.data() + i * n;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const float av = a[i * k + kk];
                    const float* b_row = b.data() + kk * n;
                    for (std::size_t j = 0; j < n; ++j)
                        c_row[j] += av * b_row[j];
                }
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(thread_count > 0 ? thread_count - 1 : 0);
    for (unsigned i = 1; i < thread_count; ++i) workers.emplace_back(worker);
    worker();
    for (auto& thread : workers) thread.join();
}

bool validate(const std::vector<float>& cpu, const std::vector<float>& gpu)
{
    if (cpu.size() != gpu.size()) return false;
    constexpr std::size_t samples = 2048;
    const std::size_t checks = std::min(samples, cpu.size());
    float max_error = 0.0f;
    for (std::size_t i = 0; i < checks; ++i) {
        const std::size_t index = checks == cpu.size() ? i : (i * cpu.size() / checks);
        const float expected = cpu[index];
        const float actual = gpu[index];
        const float error = std::abs(expected - actual);
        max_error = std::max(max_error, error);
        const float tolerance = 2.0e-3f + 2.0e-4f * std::abs(expected);
        if (!std::isfinite(actual) || error > tolerance) {
            std::cerr << "Validation failed at element " << index
                      << ": CPU=" << expected << ", GPU=" << actual
                      << ", abs error=" << error << '\n';
            return false;
        }
    }
    std::cout << "Validation: " << checks << " sampled elements passed; max abs error = "
              << std::scientific << max_error << std::defaultfloat << "\n";
    return true;
}

bool check_hr(HRESULT hr, const char* action)
{
    if (SUCCEEDED(hr)) return true;
    std::cerr << action << " failed (HRESULT 0x" << std::hex
              << static_cast<unsigned long>(hr) << std::dec << ").\n";
    return false;
}
} // namespace

int main(int argc, char** argv)
{
    int total = 2048;
    std::size_t n = total;
    if (argc > 1) {
        try {
            n = static_cast<std::size_t>(std::stoull(argv[1]));
        } catch (...) {
            std::cerr << "Usage: dxblas_gemm [square-matrix-size, default " << total << "]\n";
            return 2;
        }
    }
    if (n == 0 || n > total) {
        std::cerr << "Matrix size must be in [1, " << total << "] for this texture-backed sample.\n";
        return 2;
    }

    const std::size_t elements = n * n;
    std::cout << "dxblas_gemm: row-major FP32 GEMM using D3D11 pixel shaders\n"
              << "Matrix size: " << n << " x " << n << "\n"
              << "Approximate work: " << std::fixed << std::setprecision(3)
              << (2.0 * static_cast<double>(n) * n * n / 1.0e9) << " GFLOP\n"
              << "Per matrix: " << (elements * sizeof(float) / (1024.0 * 1024.0)) << " MiB\n"
              << "GPU path: texture arena + pixel shader + render target (no compute shader)\n\n";

    std::vector<float> a(elements), b(elements), cpu_result(elements), gpu_result(elements);
    std::mt19937 rng(0xD1B1A5u);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for (std::size_t i = 0; i < elements; ++i) {
        a[i] = distribution(rng);
        b[i] = distribution(rng);
    }

    dxblas::Context context;
    if (!check_hr(context.initialize(), "dxblas::Context::initialize")) return 1;

    // Warm up the shader path with gpuMem-owned matrix textures.
    const std::size_t warm_n = std::min<std::size_t>(16, n);
    std::vector<float> warm_a(warm_n * warm_n, 1.0f);
    std::vector<float> warm_b(warm_n * warm_n, 1.0f);
    dxblas::gpuMem<float> warm_gpu_a(warm_a), warm_gpu_b(warm_b), warm_gpu_c(warm_n * warm_n);
    warm_gpu_a.width = warm_gpu_b.width = warm_gpu_c.width = static_cast<UINT>(warm_n);
    warm_gpu_a.height = warm_gpu_b.height = warm_gpu_c.height = static_cast<UINT>(warm_n);
    if (!check_hr(context.copy(warm_gpu_a, warm_gpu_b, warm_gpu_c, warm_n * warm_n), "GEMM warm-up upload")) return 1;
    if (!check_hr(context.gemm(warm_gpu_a, warm_gpu_b, warm_gpu_c, warm_n, warm_n, warm_n), "GEMM warm-up")) return 1;

    std::cout << "Running CPU reference (multi-threaded, row-major)...\n";
    const auto cpu_start = Clock::now();
    cpu_gemm(a, b, cpu_result, n, n, n);
    const auto cpu_end = Clock::now();
    const Timing cpu_time = elapsed(cpu_start, cpu_end);
    const double gflops = 2.0 * static_cast<double>(n) * n * n / (cpu_time.milliseconds * 1.0e6);
    std::cout << "CPU time: " << cpu_time.milliseconds << " ms (" << gflops << " GFLOP/s)\n\n";

    std::cout << "Uploading matrices to gpuMem textures...\n";
    dxblas::gpuMem<float> gpu_a(a), gpu_b(b), gpu_c(elements);
    gpu_a.width = gpu_c.width = static_cast<UINT>(n);
    gpu_a.height = gpu_c.height = static_cast<UINT>(n);
    gpu_b.width = static_cast<UINT>(n);
    gpu_b.height = static_cast<UINT>(n);
    const auto upload_start = Clock::now();
    HRESULT gpu_hr = context.copy(gpu_a, gpu_b, gpu_c, elements);
    const auto upload_end = Clock::now();
    if (!check_hr(gpu_hr, "Context::copy(gpuMem)")) return 1;

    std::cout << "Running GPU-resident GEMM (no input re-upload)...\n";
    const auto gpu_start = Clock::now();
    gpu_hr = context.gemm(gpu_a, gpu_b, gpu_c, n, n, n);
    const auto gpu_compute_end = Clock::now();
    if (!check_hr(gpu_hr, "Context::gemm(gpuMem)")) {
        std::cerr << "Note: the pixel-shader implementation performs a K-length dot product per output pixel. "
                     "Large sizes can be slow or affected by the Windows GPU timeout (TDR).\n";
        return 1;
    }
    gpu_hr = context.copy_to_cpu(gpu_c, gpu_result.data(), elements);
    const auto gpu_end = Clock::now();
    if (!check_hr(gpu_hr, "Context::copy_to_cpu")) return 1;
    const Timing upload_time = elapsed(upload_start, upload_end);
    const Timing compute_time = elapsed(gpu_start, gpu_compute_end);
    std::cout << "CPU -> GPU upload: " << upload_time.milliseconds << " ms\n"
              << "GPU-resident GEMM: " << compute_time.milliseconds << " ms\n"
              << "GPU -> CPU readback: " << elapsed(gpu_compute_end, gpu_end).milliseconds << " ms\n";
    const Timing gpu_time = elapsed(upload_start, gpu_end);
    const double gpu_gflops = 2.0 * static_cast<double>(n) * n * n / (gpu_time.milliseconds * 1.0e6);
    std::cout << "GPU end-to-end time: " << gpu_time.milliseconds << " ms (" << gpu_gflops << " GFLOP/s)\n";
    if (gpu_time.milliseconds > 0.0)
        std::cout << "CPU/GPU speed ratio (greater than 1 means GPU faster): "
                  << (cpu_time.milliseconds / gpu_time.milliseconds) << "x\n";

    if (!validate(cpu_result, gpu_result)) return 1;
    std::cout << "Benchmark complete. Timings depend on GPU, CPU, driver, and power state.\n";
    return 0;
}
