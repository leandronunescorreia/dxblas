#include <dxblas/dxblas.hpp>
#include "backend/arena_texture.hpp"

#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kWarmupIterations = 1;
constexpr int kMeasuredIterations = 5;

struct Sample {
    double upload_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double upload_gpu_ms = std::numeric_limits<double>::quiet_NaN();
    double readback_cpu_ms = std::numeric_limits<double>::quiet_NaN();
};

double median(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](double value) { return !std::isfinite(value); }),
                 values.end());
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 != 0)
        return *middle;
    const double upper = *middle;
    const double lower = *std::max_element(values.begin(), middle);
    return (lower + upper) * 0.5;
}

double effective_gb_s(std::size_t bytes, double milliseconds) {
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(bytes) / (milliseconds * 1.0e6);
}

void print_number(double value, int precision = 3) {
    if (std::isfinite(value))
        std::cout << std::setw(12) << std::fixed << std::setprecision(precision) << value;
    else
        std::cout << std::setw(12) << "n/a";
}

bool run_sample(dxblas::Context& context, dxblas::detail::GpuTimer& timer, dxblas::gpuMem<float>& a,
                dxblas::gpuMem<float>& b, dxblas::gpuMem<float>& out,
                const std::vector<float>& data_a, const std::vector<float>& data_b,
                std::vector<float>& readback, Sample& sample) {
    a.assign(data_a.data(), data_a.size());
    b.assign(data_b.data(), data_b.size());

    HRESULT hr = timer.begin();
    if (FAILED(hr)) {
        std::cerr << "GpuTimer::begin failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return false;
    }
    const auto upload_start = Clock::now();
    hr = context.copy(a, b, out, data_a.size());
    const auto upload_end = Clock::now();
    const HRESULT timer_end_hr = timer.end();
    if (FAILED(hr)) {
        std::cerr << "Context::copy failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return false;
    }
    if (FAILED(timer_end_hr)) {
        std::cerr << "GpuTimer::end failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(timer_end_hr) << std::dec << '\n';
        return false;
    }
    sample.upload_cpu_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
    // The query measures GPU timeline time between timestamps around Context::copy.
    // It is intentionally collected outside the CPU wall-time interval above.
    sample.upload_gpu_ms = timer.elapsed_ms();

    const auto readback_start = Clock::now();
    hr = context.copy_to_cpu(a, readback.data(), readback.size());
    const auto readback_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy_to_cpu failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return false;
    }
    sample.readback_cpu_ms =
        std::chrono::duration<double, std::milli>(readback_end - readback_start).count();
    return true;
}

bool run_size(dxblas::Context& context, dxblas::detail::GpuTimer& timer, const char* size_label,
              std::size_t count) {
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> data_a(count, 1.25f);
    std::vector<float> data_b(count, 2.5f);
    std::vector<float> readback(count);
    dxblas::gpuMem<float> a(count), b(count), out(count);

    for (int i = 0; i < kWarmupIterations; ++i) {
        Sample warmup;
        if (!run_sample(context, timer, a, b, out, data_a, data_b, readback, warmup))
            return false;
    }

    std::vector<double> upload_cpu;
    std::vector<double> upload_gpu;
    std::vector<double> readback_cpu;
    upload_cpu.reserve(kMeasuredIterations);
    upload_gpu.reserve(kMeasuredIterations);
    readback_cpu.reserve(kMeasuredIterations);

    for (int i = 0; i < kMeasuredIterations; ++i) {
        Sample sample;
        if (!run_sample(context, timer, a, b, out, data_a, data_b, readback, sample))
            return false;
        upload_cpu.push_back(sample.upload_cpu_ms);
        upload_gpu.push_back(sample.upload_gpu_ms);
        readback_cpu.push_back(sample.readback_cpu_ms);
    }

    const double upload_cpu_ms = median(upload_cpu);
    const double upload_gpu_ms = median(upload_gpu);
    const double readback_cpu_ms = median(readback_cpu);
    const std::size_t upload_bytes = bytes * 2;

    std::cout << std::left << std::setw(10) << size_label << std::right << " | ";
    print_number(upload_cpu_ms);
    std::cout << " | ";
    print_number(effective_gb_s(upload_bytes, upload_cpu_ms));
    std::cout << " | ";
    print_number(upload_gpu_ms);
    std::cout << " | ";
    print_number(effective_gb_s(upload_bytes, upload_gpu_ms));
    std::cout << " | ";
    print_number(readback_cpu_ms);
    std::cout << " | ";
    print_number(effective_gb_s(bytes, readback_cpu_ms));
    std::cout << '\n';
    return true;
}

} // namespace

int main() {
    std::cout << "dxblas_copy_timing: D3D11 upload/readback timing (float32)\n"
              << "CPU upload time measures Context::copy submission; GPU time is the timestamped "
                 "region around that call.\n"
              << "Effective throughput uses decimal GB/s; upload counts both input buffers.\n"
              << "Warm-up: " << kWarmupIterations
              << ", measured iterations: " << kMeasuredIterations
              << ", reported statistic: median\n\n";

    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Context initialization failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }

    dxblas::NativeD3D11 native{};
    hr = dxblas::get_native_d3d11(context, native);
    if (FAILED(hr)) {
        std::cerr << "get_native_d3d11 failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> immediate_context;
    device.Attach(native.device);
    immediate_context.Attach(native.context);

    dxblas::detail::GpuTimer timer;
    hr = timer.initialize(device.Get(), immediate_context.Get());
    if (FAILED(hr)) {
        std::cerr << "GpuTimer initialization failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }

    constexpr std::size_t kib = 1024;
    constexpr std::size_t mib = 1024 * kib;
    const std::size_t sizes[] = {kib / sizeof(float), mib / sizeof(float), 16 * mib / sizeof(float),
                                 256 * mib / sizeof(float)};
    const char* labels[] = {"1 KiB", "1 MiB", "16 MiB", "256 MiB"};

    std::cout << std::left << std::setw(10) << "Size"
              << " | " << std::right << std::setw(13) << "Upload CPU ms"
              << " | " << std::setw(13) << "Upload GB/s"
              << " | " << std::setw(12) << "Upload GPU ms"
              << " | " << std::setw(12) << "GPU GB/s"
              << " | " << std::setw(14) << "Readback CPU ms"
              << " | " << std::setw(12) << "Readback GB/s\n";
    std::cout << std::string(104, '-') << '\n';

    bool passed = true;
    for (std::size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        if (!run_size(context, timer, labels[i], sizes[i])) {
            passed = false;
            break;
        }
    }

    immediate_context.Reset();
    device.Reset();
    return passed ? 0 : 1;
}
