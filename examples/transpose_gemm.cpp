#include <dxblas/dxblas.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double milliseconds(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// CPU reference calculation for A * B^T + c:
// A is (N x N), B is (N x N), B^T is (N x N), c is (N) vector broadcast across rows.
static void cpu_reference(const std::vector<float>& a, const std::vector<float>& b,
                          const std::vector<float>& c, std::vector<float>& out, std::size_t n) {
    // 1. Explicit CPU transpose of B into B^T
    std::vector<float> b_t(n * n);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t col = 0; col < n; ++col) {
            b_t[col * n + r] = b[r * n + col];
        }
    }

    // 2. Initialize out with broadcast vector c
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out[i * n + j] = c[j];
        }
    }

    // 3. Blocked matrix multiplication: out += A * B^T
    constexpr std::size_t block = 32;
    for (std::size_t ii = 0; ii < n; ii += block) {
        for (std::size_t kk = 0; kk < n; kk += block) {
            for (std::size_t jj = 0; jj < n; jj += block) {
                const std::size_t i_max = std::min(ii + block, n);
                const std::size_t k_max = std::min(kk + block, n);
                const std::size_t j_max = std::min(jj + block, n);
                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        const float av = a[i * n + k];
                        for (std::size_t j = jj; j < j_max; ++j) {
                            out[i * n + j] += av * b_t[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    std::size_t n = 1024;
    if (argc > 1) {
        try {
            n = static_cast<std::size_t>(std::stoull(argv[1]));
        } catch (...) {
            std::cerr << "Usage: dxblas_transpose_gemm [matrix-size]\n";
            return 2;
        }
    }

    if (n == 0 || n > 4096) {
        std::cerr << "Matrix size must be in [1, 4096].\n";
        return 2;
    }

    const std::size_t mat_elements = n * n;
    const std::size_t vec_elements = n;

    std::cout << "=================================================================\n";
    std::cout << "dxBLAS Sample: A(" << n << "x" << n << ") * B(" << n << "x" << n << ")^T + c("
              << n << ")\n";
    std::cout
        << "GPU Pipeline: Direct3D 11 classic shading mode (pixel shaders + render targets)\n";
    std::cout << "              No compute shaders used\n";
    std::cout << "=================================================================\n\n";

    // Allocate and initialize inputs
    std::vector<float> a(mat_elements);
    std::vector<float> b(mat_elements);
    std::vector<float> c(vec_elements);
    std::vector<float> cpu_result(mat_elements);
    std::vector<float> gpu_result(mat_elements);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < mat_elements; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    for (std::size_t i = 0; i < vec_elements; ++i) {
        c[i] = dist(rng);
    }

    // Run CPU reference calculation
    std::cout << "Running CPU reference calculation (A * B^T + c)...\n";
    const auto cpu_start = Clock::now();
    cpu_reference(a, b, c, cpu_result, n);
    const auto cpu_end = Clock::now();
    const double cpu_time = milliseconds(cpu_end - cpu_start);
    std::cout << "CPU reference time: " << std::fixed << std::setprecision(3) << cpu_time
              << " ms\n\n";

    // Initialize dxBLAS context
    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Context::initialize failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }

    // Set up GPU-resident memory
    dxblas::gpuMem<float> gpu_a(a);
    gpu_a.width = gpu_a.height = static_cast<UINT>(n);

    dxblas::gpuMem<float> gpu_b(b);
    gpu_b.width = gpu_b.height = static_cast<UINT>(n);

    dxblas::gpuMem<float> gpu_b_t(mat_elements);
    gpu_b_t.width = gpu_b_t.height = static_cast<UINT>(n);

    dxblas::gpuMem<float> gpu_c(c);
    gpu_c.width = static_cast<UINT>(n);
    gpu_c.height = 1;

    dxblas::gpuMem<float> gpu_ab(mat_elements);
    gpu_ab.width = gpu_ab.height = static_cast<UINT>(n);

    dxblas::gpuMem<float> gpu_out(mat_elements);
    gpu_out.width = gpu_out.height = static_cast<UINT>(n);

    // 1. Upload inputs to GPU
    std::cout << "Uploading inputs to GPU...\n";
    const auto upload_start = Clock::now();
    hr = context.copy(gpu_a);
    if (FAILED(hr)) {
        std::cerr << "Context::copy(gpu_a) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    hr = context.copy(gpu_b);
    if (FAILED(hr)) {
        std::cerr << "Context::copy(gpu_b) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    hr = context.copy(gpu_c);
    if (FAILED(hr)) {
        std::cerr << "Context::copy(gpu_c) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const auto upload_end = Clock::now();
    const double upload_time = milliseconds(upload_end - upload_start);

    // 2. GPU Step 1: Matrix Transpose B -> B^T
    std::cout << "Executing GPU Matrix Transpose: B(" << n << "x" << n << ") -> B^T...\n";
    const auto transpose_start = Clock::now();
    hr = context.transpose(gpu_b, gpu_b_t, n, n);
    const auto transpose_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::transpose(gpuMem) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const double transpose_time = milliseconds(transpose_end - transpose_start);

    // 3. GPU Step 2: GEMM A * B^T -> AB
    std::cout << "Executing GPU GEMM: A(" << n << "x" << n << ") * B^T(" << n << "x" << n
              << ")...\n";
    const auto gemm_start = Clock::now();
    hr = context.gemm(gpu_a, gpu_b_t, gpu_ab, n, n, n);
    const auto gemm_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::gemm(gpuMem) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const double gemm_time = milliseconds(gemm_end - gemm_start);

    // 4. GPU Step 3: Vector Addition AB + c -> Out
    std::cout << "Executing GPU Vector Addition: (A * B^T) + c(" << n << ")...\n";
    const auto add_start = Clock::now();
    hr = context.add_vector(gpu_ab, gpu_c, gpu_out, n, n);
    const auto add_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::add_vector(gpuMem) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const double add_time = milliseconds(add_end - add_start);

    // 5. Read back final result from GPU
    std::cout << "Reading back result to host...\n";
    const auto readback_start = Clock::now();
    hr = context.copy_to_cpu(gpu_out, gpu_result.data(), mat_elements);
    const auto readback_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy_to_cpu failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const double readback_time = milliseconds(readback_end - readback_start);

    const double total_gpu_compute = transpose_time + gemm_time + add_time;
    const double total_gpu_e2e = upload_time + total_gpu_compute + readback_time;

    // Validate 3-pass result against CPU
    std::size_t mismatches_3pass = 0;
    float max_error_3pass = 0.0f;
    const std::size_t checks = std::min<std::size_t>(mat_elements, 8192);
    for (std::size_t i = 0; i < checks; ++i) {
        const std::size_t index = mat_elements <= checks ? i : i * mat_elements / checks;
        const float expected = cpu_result[index];
        const float actual = gpu_result[index];
        const float error = std::fabs(expected - actual);
        max_error_3pass = std::max(max_error_3pass, error);
        const float tolerance = 2.0e-3f + 2.0e-4f * std::fabs(expected);
        if (!std::isfinite(actual) || error > tolerance) {
            ++mismatches_3pass;
        }
    }

    // =========================================================================
    // Fused 1-Pass Pipeline: Direct A * B^T + c in a single pixel shader pass!
    // Transpose is computed on-the-fly in the inner loop; bias is added at exit.
    // Zero temporary VRAM allocations (no gpu_b_t or gpu_ab textures).
    // =========================================================================
    std::cout << "\nExecuting 1-Pass Fused GPU Pipeline: A * B^T + c (single draw call, zero temp "
                 "VRAM)...\n";
    dxblas::gpuMem<float> gpu_fused_out(mat_elements);
    gpu_fused_out.width = gpu_fused_out.height = static_cast<UINT>(n);
    std::vector<float> fused_result(mat_elements);

    const auto fused_start = Clock::now();
    hr = context.gemm_transpose_bias(gpu_a, gpu_b, gpu_c, gpu_fused_out, n, n, n);
    const auto fused_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::gemm_transpose_bias failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const double fused_compute_time = milliseconds(fused_end - fused_start);

    const auto fused_readback_start = Clock::now();
    hr = context.copy_to_cpu(gpu_fused_out, fused_result.data(), mat_elements);
    const auto fused_readback_end = Clock::now();
    if (FAILED(hr)) {
        std::cerr << "Context::copy_to_cpu(fused) failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }
    const double fused_readback_time = milliseconds(fused_readback_end - fused_readback_start);
    const double fused_e2e_time = upload_time + fused_compute_time + fused_readback_time;

    // Validate fused result against CPU
    std::size_t mismatches_fused = 0;
    float max_error_fused = 0.0f;
    for (std::size_t i = 0; i < checks; ++i) {
        const std::size_t index = mat_elements <= checks ? i : i * mat_elements / checks;
        const float expected = cpu_result[index];
        const float actual = fused_result[index];
        const float error = std::fabs(expected - actual);
        max_error_fused = std::max(max_error_fused, error);
        const float tolerance = 2.0e-3f + 2.0e-4f * std::fabs(expected);
        if (!std::isfinite(actual) || error > tolerance) {
            ++mismatches_fused;
        }
    }

    std::cout << "\n================ Performance Comparison ================\n";
    std::cout << "CPU Reference Time:                    " << std::setw(10) << cpu_time << " ms\n";
    std::cout << "Host -> GPU Upload:                    " << std::setw(10) << upload_time
              << " ms\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "[3-Pass Pipeline]\n";
    std::cout << "  1. GPU Matrix Transpose (B -> B^T):  " << std::setw(10) << transpose_time
              << " ms\n";
    std::cout << "  2. GPU GEMM (A * B^T):               " << std::setw(10) << gemm_time << " ms\n";
    std::cout << "  3. GPU Vector Add ((A*B^T) + c):     " << std::setw(10) << add_time << " ms\n";
    std::cout << "  Total 3-Pass GPU Kernel Time:        " << std::setw(10) << total_gpu_compute
              << " ms\n";
    std::cout << "  3-Pass Readback Time:                " << std::setw(10) << readback_time
              << " ms\n";
    std::cout << "  3-Pass End-to-End Total:             " << std::setw(10) << total_gpu_e2e
              << " ms\n";
    std::cout << "  Validation: " << checks << " sampled cells, mismatches=" << mismatches_3pass
              << ", max abs error=" << std::scientific << max_error_3pass << std::defaultfloat
              << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "[1-Pass Fused Pipeline: A * B^T + c]\n";
    std::cout << "  Fused GPU Kernel Time:               " << std::setw(10) << fused_compute_time
              << " ms\n";
    std::cout << "  Fused Readback Time:                 " << std::setw(10) << fused_readback_time
              << " ms\n";
    std::cout << "  Fused End-to-End Total:              " << std::setw(10) << fused_e2e_time
              << " ms\n";
    std::cout << "  Validation: " << checks << " sampled cells, mismatches=" << mismatches_fused
              << ", max abs error=" << std::scientific << max_error_fused << std::defaultfloat
              << "\n";
    std::cout << "--------------------------------------------------------\n";
    const double speedup = total_gpu_compute / std::max(fused_compute_time, 0.001);
    std::cout << "Kernel Speedup (3-Pass vs 1-Pass):      " << std::setw(10) << std::fixed
              << std::setprecision(2) << speedup << "x faster\n";
    std::cout << "Temporary Textures Eliminated:                  2 (B^T and AB)\n";

    if (mismatches_3pass == 0 && mismatches_fused == 0) {
        std::cout << "[PASS] Both 3-pass and 1-pass fused pipelines match CPU reference.\n";
        return 0;
    } else {
        std::cout << "[FAIL] Validation failed: 3-pass mismatches=" << mismatches_3pass
                  << ", fused mismatches=" << mismatches_fused << "\n";
        return 3;
    }
}
