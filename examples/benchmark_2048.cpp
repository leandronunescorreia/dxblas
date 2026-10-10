#include <dxblas/dxblas.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

static void print_hresult(const char* operation, HRESULT hr)
{
    if (FAILED(hr))
        std::cerr << operation << " failed: 0x"
                  << std::hex << static_cast<unsigned long>(hr)
                  << std::dec << '\n';
}

template <typename T>
static bool equal_results(
    const std::vector<T>& actual,
    const std::vector<T>& expected)
{
    if (actual.size() != expected.size())
        return false;

    return std::equal(actual.begin(), actual.end(), expected.begin());
}

static bool equal_float_results(
    const std::vector<float>& actual,
    const std::vector<float>& expected)
{
    if (actual.size() != expected.size())
        return false;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1.0e-5f)
            return false;
    }
    return true;
}

template <typename F>
static double benchmark_ms(std::size_t iterations, F&& operation)
{
    const auto begin = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        operation();
    const auto end = Clock::now();

    return std::chrono::duration<double, std::milli>(end - begin).count()
        / static_cast<double>(iterations);
}

} // namespace

int main()
{
    constexpr std::size_t N = 2048;
    constexpr std::size_t WARMUP = 5;
    constexpr std::size_t ITERATIONS = 250;

    std::cout << "=== dxBLAS: CPU vs DX11 Pixel-Pipeline Benchmark ===\n";
    std::cout << "Elements:   " << N << '\n';
    std::cout << "Iterations: " << ITERATIONS << " (" << WARMUP << " warm-up)\n";
    std::cout << "GPU path:   offscreen Texture2D + pixel shader + staging readback\n\n";

    // Do not enable the D3D11 debug layer in performance measurements.
    dxblas::Context context(
        D3D_DRIVER_TYPE_HARDWARE,
        0);

    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("context.initialize", hr);
        return 1;
    }

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> cpu_sum(N);
    std::vector<float> cpu_mul(N);
    std::vector<float> gpu_sum(N);
    std::vector<float> gpu_mul(N);

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = 0.25f + static_cast<float>(i) * 0.5f;
        b[i] = 1.5f + static_cast<float>(i % 17) * 0.125f;
    }

    // Build CPU reference results once. These are also used to validate the GPU.
    std::transform(a.begin(), a.end(), b.begin(), cpu_sum.begin(),
                   [](float x, float y) { return x + y; });
    std::transform(a.begin(), a.end(), b.begin(), cpu_mul.begin(),
                   [](float x, float y) { return x * y; });

    // Warm up both paths so shader compilation/resource creation does not dominate
    // the measured GPU timings.
    for (std::size_t i = 0; i < WARMUP; ++i) {
        std::copy(a.begin(), a.end(), gpu_sum.begin());
        hr = context.sum(DXGI_FORMAT_R32_FLOAT,
                         a.data(), b.data(), gpu_sum.data(), N);
        if (FAILED(hr)) {
            print_hresult("warm-up context.sum", hr);
            return 2;
        }

        hr = context.mul(DXGI_FORMAT_R32_FLOAT,
                         a.data(), b.data(), gpu_mul.data(), N);
        if (FAILED(hr)) {
            print_hresult("warm-up context.mul", hr);
            return 3;
        }
    }

    // Validate before timing.
    hr = context.sum(DXGI_FORMAT_R32_FLOAT,
                     a.data(), b.data(), gpu_sum.data(), N);
    if (FAILED(hr)) {
        print_hresult("validation context.sum", hr);
        return 4;
    }

    hr = context.mul(DXGI_FORMAT_R32_FLOAT,
                     a.data(), b.data(), gpu_mul.data(), N);
    if (FAILED(hr)) {
        print_hresult("validation context.mul", hr);
        return 5;
    }

    if (!equal_float_results(gpu_sum, cpu_sum) ||
        !equal_float_results(gpu_mul, cpu_mul)) {
        std::cerr << "[FAIL] GPU results do not match CPU results.\n";
        return 6;
    }

    std::cout << "[PASS] GPU results match CPU results for all " << N
              << " elements.\n\n";

    volatile float cpu_sink = 0.0f;
    const double cpu_sum_ms = benchmark_ms(ITERATIONS, [&] {
        std::transform(a.begin(), a.end(), b.begin(), cpu_sum.begin(),
                       [](float x, float y) { return x + y; });
        cpu_sink += cpu_sum[N - 1];
    });

    const double cpu_mul_ms = benchmark_ms(ITERATIONS, [&] {
        std::transform(a.begin(), a.end(), b.begin(), cpu_mul.begin(),
                       [](float x, float y) { return x * y; });
        cpu_sink += cpu_mul[N - 1];
    });

    const double gpu_sum_ms = benchmark_ms(ITERATIONS, [&] {
        const HRESULT result = context.sum(
            DXGI_FORMAT_R32_FLOAT,
            a.data(), b.data(), gpu_sum.data(), N);
        if (FAILED(result))
            std::abort();
        cpu_sink += gpu_sum[N - 1];
    });

    const double gpu_mul_ms = benchmark_ms(ITERATIONS, [&] {
        const HRESULT result = context.mul(
            DXGI_FORMAT_R32_FLOAT,
            a.data(), b.data(), gpu_mul.data(), N);
        if (FAILED(result))
            std::abort();
        cpu_sink += gpu_mul[N - 1];
    });


    // GPU-resident chained path: upload once, keep the intermediate and final
    // arrays as textures, and read back only after the batch has completed.
    dxblas::GpuArray gpu_a, gpu_b, gpu_tmp, gpu_result;
    hr = context.upload_gpu(DXGI_FORMAT_R32_FLOAT, a.data(), N, gpu_a);
    if (FAILED(hr)) { print_hresult("upload_gpu A", hr); return 7; }
    hr = context.upload_gpu(DXGI_FORMAT_R32_FLOAT, b.data(), N, gpu_b);
    if (FAILED(hr)) { print_hresult("upload_gpu B", hr); return 8; }

    for (std::size_t i = 0; i < WARMUP; ++i) {
        hr = context.sum_gpu(gpu_a, gpu_b, gpu_tmp);
        if (FAILED(hr)) { print_hresult("warm-up sum_gpu", hr); return 9; }
        hr = context.mul_gpu(gpu_tmp, gpu_b, gpu_result);
        if (FAILED(hr)) { print_hresult("warm-up mul_gpu", hr); return 10; }
    }

    auto resident_batch = [&]() {
        for (std::size_t i = 0; i < ITERATIONS; ++i) {
            if (FAILED(context.sum_gpu(gpu_a, gpu_b, gpu_tmp))) std::abort();
            if (FAILED(context.mul_gpu(gpu_tmp, gpu_b, gpu_result))) std::abort();
        }
    };

    const auto resident_begin = Clock::now();
    resident_batch();
    std::vector<float> resident_result(N);
    hr = context.readback_gpu(gpu_result, resident_result.data(), N);
    const auto resident_end = Clock::now();
    if (FAILED(hr)) { print_hresult("readback_gpu", hr); return 11; }
    const double resident_chain_ms =
        std::chrono::duration<double, std::milli>(resident_end - resident_begin).count()
        / static_cast<double>(ITERATIONS);

    std::vector<float> expected_chain(N);
    std::transform(cpu_sum.begin(), cpu_sum.end(), b.begin(), expected_chain.begin(),
                   [](float x, float y) { return x * y; });
    if (!equal_float_results(resident_result, expected_chain)) {
        std::cerr << "[FAIL] GPU-resident chained result does not match CPU reference.\\n";
        return 12;
    }

    // Keep the compiler from treating the benchmark loop as dead work.
    std::cout << std::setprecision(4) << std::fixed;
    std::cout << "Operation     CPU ms/op     GPU ms/op     GPU/CPU\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "A + B         " << std::setw(10) << cpu_sum_ms
              << "   " << std::setw(10) << gpu_sum_ms
              << "   " << std::setw(7) << (gpu_sum_ms / cpu_sum_ms) << "x\n";
    std::cout << "A * B         " << std::setw(10) << cpu_mul_ms
              << "   " << std::setw(10) << gpu_mul_ms
              << "   " << std::setw(7) << (gpu_mul_ms / cpu_mul_ms) << "x\n";

    std::cout << "\nGPU-resident chain: sum_gpu(A,B) -> mul_gpu(temp,B)\n";
    std::cout << "  amortized ms/chain (one final readback per batch): " << resident_chain_ms << '\n';
    std::cout << "  [PASS] final chained result validated after one readback.\n";
    std::cout << "\nNote: synchronous timings include per-call staging readback. The resident-chain\n"
              << "      timing measures queued GPU calls; its final readback is outside the timer.\n";
    std::cout << "Benchmark sink: " << cpu_sink << '\n';

    return 0;
}
