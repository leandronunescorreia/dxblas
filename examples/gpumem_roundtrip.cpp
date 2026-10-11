#include <dxblas/dxblas.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using Clock = std::chrono::steady_clock;

static void print_hresult(const char* operation, HRESULT hr)
{
    std::cerr << operation << " failed: HRESULT 0x" << std::hex
              << static_cast<unsigned long>(hr) << std::dec << '\n';
}

template<class T>
static bool values_match(const T& a, const T& b)
{
    if constexpr (std::is_floating_point_v<T>)
        return std::fabs(a - b) <= 1e-6f;
    else
        return a == b;
}

template<class T>
static int run_scenario(dxblas::Context& context,
                        const std::string& label,
                        const std::vector<T>& cpu_a,
                        const std::vector<T>& cpu_b)
{
    if (cpu_a.size() != cpu_b.size()) {
        std::cerr << label << ": input sizes differ\n";
        return 1;
    }

    // Constructing gpuMem owns CPU staging data; this is intentionally outside
    // the transfer timing so the measurements focus on the backend operations.
    dxblas::gpuMem<T> a(cpu_a);
    dxblas::gpuMem<T> b(cpu_b);
    dxblas::gpuMem<T> out(cpu_a.size());

    const auto upload_start = Clock::now();
    HRESULT hr = context.copy(a, b, out, cpu_a.size());
    const auto upload_end = Clock::now();
    if (FAILED(hr)) {
        print_hresult((label + " CPU -> GPU upload").c_str(), hr);
        return 2;
    }

    std::vector<T> roundtrip_a(cpu_a.size());
    std::vector<T> roundtrip_b(cpu_b.size());

    const auto readback_start = Clock::now();
    hr = context.copy_to_cpu_batch({
        {&a, roundtrip_a.data(), roundtrip_a.size()},
        {&b, roundtrip_b.data(), roundtrip_b.size()},
    });
    const auto readback_end = Clock::now();
    if (FAILED(hr)) {
        print_hresult((label + " batched GPU -> CPU readback").c_str(), hr);
        return 3;
    }

    for (std::size_t i = 0; i < cpu_a.size(); ++i) {
        if (!values_match(roundtrip_a[i], cpu_a[i]) ||
            !values_match(roundtrip_b[i], cpu_b[i])) {
            std::cerr << label << ": roundtrip mismatch at element " << i << '\n';
            return 5;
        }
    }

    const double upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
    const double readback_ms = std::chrono::duration<double, std::milli>(readback_end - readback_start).count();
    const double total_ms = std::chrono::duration<double, std::milli>(readback_end - upload_start).count();
    const std::size_t bytes = cpu_a.size() * sizeof(T);

    std::cout << "\nScenario: " << label << '\n'
              << "  Elements per input : " << cpu_a.size() << '\n'
              << "  Bytes per input    : " << bytes << '\n'
              << std::fixed << std::setprecision(3)
              << "  CPU -> GPU (both inputs + output allocation): " << upload_ms << " ms\n"
              << "  GPU -> CPU (both input readbacks)           : " << readback_ms << " ms\n"
              << "  Total measured roundtrip                    : " << total_ms << " ms\n"
              << "  Validation                                  : PASSED\n";
    return 0;
}

int main()
{
    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("initialize", hr);
        return 1;
    }

    // Scenario 1: small, deterministic values for an easy-to-read smoke test.
    const std::vector<float> small_a{1.25f, 2.5f, 3.75f, 5.0f, 6.25f};
    const std::vector<float> small_b{10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    int result = run_scenario(context, "small deterministic float", small_a, small_b);
    if (result != 0) return result;

    // Scenarios 2 and 3: two independently generated arrays of 4096 values.
    constexpr std::size_t count = 4096;
    std::mt19937 rng(0xD1B1A5u); // fixed seed makes failures reproducible
    std::uniform_real_distribution<float> float_distribution(-1000.0f, 1000.0f);
    std::uniform_int_distribution<std::int32_t> int_distribution(-100000, 100000);

    std::vector<float> random_float_a(count), random_float_b(count);
    for (std::size_t i = 0; i < count; ++i) {
        random_float_a[i] = float_distribution(rng);
        random_float_b[i] = float_distribution(rng);
    }
    result = run_scenario(context, "4096 random float32 values", random_float_a, random_float_b);
    if (result != 0) return result;

    std::vector<std::int32_t> random_int_a(count), random_int_b(count);
    for (std::size_t i = 0; i < count; ++i) {
        random_int_a[i] = int_distribution(rng);
        random_int_b[i] = int_distribution(rng);
    }
    result = run_scenario(context, "4096 random int32 values", random_int_a, random_int_b);
    if (result != 0) return result;

    std::cout << "\nAll GPU memory roundtrip scenarios passed.\n";
    return 0;
}
