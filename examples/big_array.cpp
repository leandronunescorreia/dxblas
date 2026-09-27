#include <dxblas/dxblas.hpp>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

static void print_hresult(const char* operation, HRESULT hr)
{
    if (FAILED(hr))
        std::cerr << operation << " failed: 0x"
                  << std::hex << static_cast<unsigned long>(hr)
                  << std::dec << '\n';
}

int main()
{
    constexpr std::size_t N = 512;
    std::cout << "=== dxBLAS: 512-Element Array Sample ===\n";
    std::cout << "Element count: " << N << " (" << (N / 64) << " thread groups of 64 threads)\n\n";

    // Initialize context
    dxblas::Context context(
        D3D_DRIVER_TYPE_HARDWARE,
        D3D11_CREATE_DEVICE_DEBUG);

    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("context.initialize", hr);
        return 1;
    }

    // 1. Float arrays
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> sum_out(N);
    std::vector<float> mul_out(N);

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = 2.0f;
    }

    // GPU Sum
    hr = context.sum(
        DXGI_FORMAT_R32_FLOAT,
        a.data(),
        b.data(),
        sum_out.data(),
        N);

    if (FAILED(hr)) {
        print_hresult("context.sum (float)", hr);
        return 2;
    }

    // GPU Mul
    hr = context.mul(
        DXGI_FORMAT_R32_FLOAT,
        a.data(),
        b.data(),
        mul_out.data(),
        N);

    if (FAILED(hr)) {
        print_hresult("context.mul (float)", hr);
        return 3;
    }

    // Validate float results against CPU reference
    bool float_valid = true;
    for (std::size_t i = 0; i < N; ++i) {
        float expected_sum = a[i] + b[i];
        float expected_mul = a[i] * b[i];

        if (std::abs(sum_out[i] - expected_sum) > 1e-5f ||
            std::abs(mul_out[i] - expected_mul) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ":\n"
                      << "  sum: got " << sum_out[i] << ", expected " << expected_sum << '\n'
                      << "  mul: got " << mul_out[i] << ", expected " << expected_mul << '\n';
            float_valid = false;
            break;
        }
    }

    if (float_valid) {
        std::cout << "[PASS] Float: all 512 elements matched CPU reference!\n";
    } else {
        std::cerr << "[FAIL] Float validation failed.\n";
        return 4;
    }

    // 2. Display sample results
    std::cout << "\nSample results (first 5 and last 5 elements):\n";
    std::cout << std::setw(6) << "Index"
              << std::setw(12) << "A"
              << std::setw(12) << "B"
              << std::setw(14) << "A + B (sum)"
              << std::setw(14) << "A * B (mul)" << '\n';
    std::cout << std::string(58, '-') << '\n';

    auto print_row = [&](std::size_t i) {
        std::cout << std::setw(6) << i
                  << std::setw(12) << a[i]
                  << std::setw(12) << b[i]
                  << std::setw(14) << sum_out[i]
                  << std::setw(14) << mul_out[i] << '\n';
    };

    for (std::size_t i = 0; i < 5; ++i)
        print_row(i);

    std::cout << std::setw(6) << "..."
              << std::setw(12) << "..."
              << std::setw(12) << "..."
              << std::setw(14) << "..."
              << std::setw(14) << "..." << '\n';

    for (std::size_t i = N - 5; i < N; ++i)
        print_row(i);

    // 3. Uint32 verification
    std::vector<std::uint32_t> ua(N);
    std::vector<std::uint32_t> ub(N);
    std::vector<std::uint32_t> usum(N);
    std::vector<std::uint32_t> umul(N);

    for (std::size_t i = 0; i < N; ++i) {
        ua[i] = static_cast<std::uint32_t>(i + 1);
        ub[i] = 10;
    }

    hr = context.sum(DXGI_FORMAT_R32_UINT, ua.data(), ub.data(), usum.data(), N);
    if (FAILED(hr)) {
        print_hresult("context.sum (uint32)", hr);
        return 5;
    }

    hr = context.mul(DXGI_FORMAT_R32_UINT, ua.data(), ub.data(), umul.data(), N);
    if (FAILED(hr)) {
        print_hresult("context.mul (uint32)", hr);
        return 6;
    }

    bool uint_valid = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (usum[i] != (ua[i] + ub[i]) || umul[i] != (ua[i] * ub[i])) {
            std::cerr << "Uint32 mismatch at index " << i << '\n';
            uint_valid = false;
            break;
        }
    }

    if (uint_valid) {
        std::cout << "\n[PASS] Uint32: all 512 elements matched CPU reference!\n";
    } else {
        std::cerr << "[FAIL] Uint32 validation failed.\n";
        return 7;
    }

    return 0;
}
