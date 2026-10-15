#include <dxblas/dxblas.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(!std::is_copy_constructible_v<dxblas::gpuMem<float>>,
              "gpuMem must not be copy-constructible");
static_assert(!std::is_copy_assignable_v<dxblas::gpuMem<float>>,
              "gpuMem must not be copy-assignable");
static_assert(std::is_move_constructible_v<dxblas::gpuMem<float>>,
              "gpuMem must remain move-constructible");
static_assert(std::is_move_assignable_v<dxblas::gpuMem<float>>,
              "gpuMem must remain move-assignable");

namespace {

constexpr std::size_t kSizes[] = {
    1, 2, 5, 255, 4095, 4096, 4097, 16383, 16384, 16385,
    32768, 32769, 1'000'003
};

template<class T>
void fill_pattern(std::vector<T>& values, std::uint32_t salt = 0)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            values[i] = static_cast<T>(i + salt) * static_cast<T>(0.25);
        } else {
            using U = std::make_unsigned_t<T>;
            const U bits = static_cast<U>(
                static_cast<std::uint32_t>(i + salt) * 2654435761u);
            std::memcpy(&values[i], &bits, sizeof(T));
        }
    }
}

template<class T>
bool bit_equal(const std::vector<T>& actual, const std::vector<T>& expected,
               std::size_t count)
{
    return actual.size() >= count && expected.size() >= count &&
        (count == 0 || std::memcmp(actual.data(), expected.data(), count * sizeof(T)) == 0);
}

void report(const std::string& label, bool passed, int& failures)
{
    std::cout << (passed ? "[PASS] " : "[FAIL] ") << label << '\n';
    if (!passed) ++failures;
}

template<class T>
bool upload_pair(dxblas::Context& context,
                 dxblas::gpuMem<T>& a,
                 dxblas::gpuMem<T>& b,
                 dxblas::gpuMem<T>& out,
                 const std::vector<T>& values_a,
                 const std::vector<T>& values_b)
{
    if (values_a.size() != values_b.size()) return false;
    a.assign(values_a.data(), values_a.size());
    b.assign(values_b.data(), values_b.size());
    if (out.size() != values_a.size()) out = dxblas::gpuMem<T>(values_a.size());

    const HRESULT hr = context.copy(a, b, out, values_a.size());
    if (FAILED(hr)) {
        std::cerr << "  Context::copy failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return false;
    }
    return true;
}

template<class T>
void run_size_case(dxblas::Context& context, std::size_t count,
                   const char* type_name, int& failures)
{
    std::vector<T> expected_a(count), expected_b(count), actual_a(count), actual_b(count);
    fill_pattern(expected_a, 0);
    fill_pattern(expected_b, 37);

    dxblas::gpuMem<T> a(count), b(count), out(count);
    bool passed = upload_pair(context, a, b, out, expected_a, expected_b);
    if (passed) {
        const HRESULT hr = context.copy_to_cpu_batch({
            {&a, actual_a.data(), count},
            {&b, actual_b.data(), count},
        });
        if (FAILED(hr)) {
            std::cerr << "  copy_to_cpu_batch failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
            passed = false;
        } else {
            passed = bit_equal(actual_a, expected_a, count) &&
                     bit_equal(actual_b, expected_b, count);
        }
    }
    report(std::string(type_name) + " count=" + std::to_string(count), passed, failures);
}

template<class T>
void run_type_sizes(dxblas::Context& context, const char* type_name, int& failures)
{
    for (const auto count : kSizes)
        run_size_case<T>(context, count, type_name, failures);
}

template<class T>
void run_reupload_case(dxblas::Context& context, const char* type_name, int& failures)
{
    constexpr std::size_t count = 4097;
    std::vector<T> first_a(count), first_b(count), second_a(count), second_b(count);
    std::vector<T> actual_a(count), actual_b(count);
    fill_pattern(first_a, 1);
    fill_pattern(first_b, 2);
    fill_pattern(second_a, 101);
    fill_pattern(second_b, 202);

    dxblas::gpuMem<T> a(count), b(count), out(count);
    bool passed = upload_pair(context, a, b, out, first_a, first_b);
    ID3D11Texture2D* const first_texture = passed ? a.texture.Get() : nullptr;
    bool texture_reused = false;
    if (passed) {
        passed = upload_pair(context, a, b, out, second_a, second_b);
        texture_reused = passed && a.texture.Get() == first_texture;
    }
    if (passed) {
        const HRESULT hr = context.copy_to_cpu_batch({
            {&a, actual_a.data(), count},
            {&b, actual_b.data(), count},
        });
        passed = SUCCEEDED(hr) && bit_equal(actual_a, second_a, count) &&
                 bit_equal(actual_b, second_b, count);
        if (FAILED(hr))
            std::cerr << "  second-upload readback failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
    }
    report(std::string(type_name) + " reupload uses second data set", passed, failures);
    report(std::string(type_name) + " same-size reupload reuses texture", texture_reused, failures);
}

void run_move_semantics_case(dxblas::Context& context, int& failures)
{
    constexpr std::size_t count = 4097;
    std::vector<float> values_a(count), values_b(count);
    fill_pattern(values_a, 17);
    fill_pattern(values_b, 29);

    dxblas::gpuMem<float> a(count), b(count), out(count);
    bool passed = upload_pair(context, a, b, out, values_a, values_b);
    if (passed) {
        ID3D11Texture2D* const a_texture = a.texture.Get();
        ID3D11ShaderResourceView* const a_srv = a.srv.Get();
        ID3D11RenderTargetView* const a_rtv = a.rtv.Get();
        dxblas::gpuMem<float> moved_a(std::move(a));
        passed = moved_a.texture.Get() == a_texture && moved_a.srv.Get() == a_srv &&
                 moved_a.rtv.Get() == a_rtv && !a.texture && !a.srv && !a.rtv &&
                 a.width == 0 && a.height == 0 && a.count == 0 &&
                 a.storage_format == DXGI_FORMAT_UNKNOWN &&
                 !a.shape_is_user_defined && a.data() == nullptr;

        ID3D11Texture2D* const out_texture = out.texture.Get();
        ID3D11ShaderResourceView* const out_srv = out.srv.Get();
        ID3D11RenderTargetView* const out_rtv = out.rtv.Get();
        dxblas::gpuMem<float> moved_out(1);
        moved_out = std::move(out);
        passed = passed && moved_out.texture.Get() == out_texture &&
                 moved_out.srv.Get() == out_srv && moved_out.rtv.Get() == out_rtv &&
                 !out.texture && !out.srv && !out.rtv && out.width == 0 &&
                 out.height == 0 && out.count == 0 &&
                 out.storage_format == DXGI_FORMAT_UNKNOWN &&
                 !out.shape_is_user_defined && out.data() == nullptr;
    }
    report("gpuMem move transfers resources and empties source", passed, failures);
}

void run_explicit_shape_case(dxblas::Context& context, int& failures)
{
    constexpr std::size_t count = 100;
    std::vector<float> values_a(count), values_b(count), expected(count), actual(count);
    fill_pattern(values_a, 7);
    fill_pattern(values_b, 19);

    dxblas::gpuMem<float> a(count), b(count), out(count);
    // Explicit dimensions are used by the resident GEMM path. The inputs have
    // the same row width but different allocated heights; the requested prefix
    // still covers ten rows in each texture.
    a.width = 10; a.height = 10;
    b.width = 10; b.height = 12;
    out.width = 10; out.height = 10;

    bool passed = upload_pair(context, a, b, out, values_a, values_b);
    passed = passed && a.width == 10 && a.height == 10 &&
             b.width == 10 && b.height == 12 &&
             a.shape_is_user_defined && b.shape_is_user_defined;
    if (passed) {
        // Reupload a shorter prefix; explicitly shaped matrix resources should
        // retain their row widths/heights instead of reverting to a 1D layout.
        a.assign(values_a.data(), 50);
        b.assign(values_b.data(), 50);
        const HRESULT upload_hr = context.copy(a, b, out, 50);
        passed = SUCCEEDED(upload_hr) &&
                 a.width == 10 && a.height == 10 &&
                 b.width == 10 && b.height == 12 &&
                 a.shape_is_user_defined && b.shape_is_user_defined;
    }
    if (passed) {
        for (std::size_t i = 0; i < 50; ++i)
            expected[i] = values_a[i] + values_b[i];
        const HRESULT hr = context.sum(a, b, out, 50);
        passed = SUCCEEDED(hr) &&
                 SUCCEEDED(context.copy_to_cpu(out, actual.data(), 50)) &&
                 bit_equal(actual, expected, 50) &&
                 out.width == 10 && out.height == 5;
    }
    report("explicit matrix shape retained; arithmetic accepts differing heights",
           passed, failures);
}

void run_reuse_arithmetic_case(dxblas::Context& context,
                               std::size_t large_count,
                               std::size_t small_count,
                               const char* label,
                               int& failures)
{
    std::vector<float> initial_a(large_count), initial_b(large_count);
    std::vector<float> small_a(small_count), small_b(small_count);
    std::vector<float> expected(small_count), actual(small_count);
    fill_pattern(initial_a, 1);
    fill_pattern(initial_b, 2);
    fill_pattern(small_a, 11);
    fill_pattern(small_b, 29);

    dxblas::gpuMem<float> a(large_count), initial_peer(large_count), initial_out(large_count);
    bool passed = upload_pair(context, a, initial_peer, initial_out, initial_a, initial_b);
    Microsoft::WRL::ComPtr<ID3D11Texture2D> old_texture;
    if (passed) old_texture = a.texture;

    // Keep a's gpuMem object (and its prior allocation) but shrink its pending
    // CPU payload. b and out are fresh objects with the smaller shape.
    dxblas::gpuMem<float> b(small_count), out(small_count);
    if (passed) {
        a.assign(small_a.data(), small_count);
        b.assign(small_b.data(), small_count);
        const HRESULT hr = context.copy(a, b, out, small_count);
        passed = SUCCEEDED(hr);
        if (FAILED(hr))
            std::cerr << "  shrink upload failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
    }

    if (passed) {
        passed = a.width == b.width &&
                 a.width == small_count &&
                 a.height >= 1 && b.height >= 1 &&
                 a.texture.Get() != nullptr && a.texture.Get() != old_texture.Get();
    }

    bool sum_ok = false;
    bool mul_ok = false;
    if (passed) {
        for (std::size_t i = 0; i < small_count; ++i)
            expected[i] = small_a[i] + small_b[i];
        HRESULT hr = context.sum(a, b, out, small_count);
        if (SUCCEEDED(hr)) hr = context.copy_to_cpu(out, actual.data(), small_count);
        sum_ok = SUCCEEDED(hr) && bit_equal(actual, expected, small_count);

        for (std::size_t i = 0; i < small_count; ++i)
            expected[i] = small_a[i] * small_b[i];
        hr = context.mul(a, b, out, small_count);
        if (SUCCEEDED(hr)) hr = context.copy_to_cpu(out, actual.data(), small_count);
        mul_ok = SUCCEEDED(hr) && bit_equal(actual, expected, small_count);
    }

    report(std::string(label) + " shrink/reuse sum", sum_ok, failures);
    report(std::string(label) + " shrink/reuse mul", mul_ok, failures);

    bool alias_ok = false;
    if (passed) {
        const HRESULT sum_alias_a = context.sum(a, b, a, small_count);
        const HRESULT sum_alias_b = context.sum(a, b, b, small_count);
        const HRESULT mul_alias_a = context.mul(a, b, a, small_count);
        const HRESULT mul_alias_b = context.mul(a, b, b, small_count);
        alias_ok = sum_alias_a == E_INVALIDARG && sum_alias_b == E_INVALIDARG &&
                   mul_alias_a == E_INVALIDARG && mul_alias_b == E_INVALIDARG;
    }
    report(std::string(label) + " rejects output/input texture aliasing", alias_ok, failures);
}

template<class T>
void run_shrink_case(dxblas::Context& context, const char* type_name, int& failures)
{
    constexpr std::size_t large_count = 32769;
    constexpr std::size_t small_count = 255;
    std::vector<T> large_a(large_count), large_b(large_count);
    std::vector<T> small_a(small_count), small_b(small_count);
    std::vector<T> actual_a(small_count), actual_b(small_count);
    fill_pattern(large_a, 3);
    fill_pattern(large_b, 4);
    fill_pattern(small_a, 303);
    fill_pattern(small_b, 404);

    dxblas::gpuMem<T> a(large_count), b(large_count), out(large_count);
    bool passed = upload_pair(context, a, b, out, large_a, large_b);
    if (passed) passed = upload_pair(context, a, b, out, small_a, small_b);
    if (passed) {
        const HRESULT hr = context.copy_to_cpu_batch({
            {&a, actual_a.data(), small_count},
            {&b, actual_b.data(), small_count},
        });
        passed = SUCCEEDED(hr) && bit_equal(actual_a, small_a, small_count) &&
                 bit_equal(actual_b, small_b, small_count);
        if (FAILED(hr))
            std::cerr << "  smaller-upload readback failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
    }
    report(std::string(type_name) + " smaller overwrite " +
           std::to_string(large_count) + " -> " + std::to_string(small_count),
           passed, failures);
}

template<class T>
void run_prefix_case(dxblas::Context& context, const char* type_name, int& failures)
{
    constexpr std::size_t count = 4097;
    constexpr std::size_t prefix = 255;
    std::vector<T> expected_a(count), expected_b(count), actual(prefix);
    fill_pattern(expected_a, 5);
    fill_pattern(expected_b, 6);
    dxblas::gpuMem<T> a(count), b(count), out(count);
    bool passed = upload_pair(context, a, b, out, expected_a, expected_b);
    if (passed) {
        const HRESULT hr = context.copy_to_cpu(a, actual.data(), prefix);
        passed = SUCCEEDED(hr) && bit_equal(actual, expected_a, prefix);
        if (FAILED(hr))
            std::cerr << "  prefix readback failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
    }
    report(std::string(type_name) + " prefix readback " + std::to_string(prefix) +
           "/" + std::to_string(count), passed, failures);
}

void run_batch_size_case(dxblas::Context& context, int source_count, int& failures)
{
    constexpr std::size_t counts[] = {255, 4097, 32769};
    std::vector<float> expected[3];
    std::vector<float> peer[3];
    std::vector<float> actual[3];
    dxblas::gpuMem<float> a[3];
    dxblas::gpuMem<float> b[3];
    dxblas::gpuMem<float> out[3];

    bool passed = true;
    for (int i = 0; i < source_count; ++i) {
        expected[i].resize(counts[i]);
        peer[i].resize(counts[i]);
        actual[i].resize(counts[i]);
        fill_pattern(expected[i], static_cast<std::uint32_t>(10 + i));
        fill_pattern(peer[i], static_cast<std::uint32_t>(20 + i));
        a[i] = dxblas::gpuMem<float>(counts[i]);
        b[i] = dxblas::gpuMem<float>(counts[i]);
        out[i] = dxblas::gpuMem<float>(counts[i]);
        if (!upload_pair(context, a[i], b[i], out[i], expected[i], peer[i])) {
            passed = false;
            break;
        }
    }

    if (passed && source_count == 2) {
        const HRESULT hr = context.copy_to_cpu_batch({
            {&a[0], actual[0].data(), counts[0]},
            {&a[1], actual[1].data(), counts[1]},
        });
        passed = SUCCEEDED(hr) &&
            bit_equal(actual[0], expected[0], counts[0]) &&
            bit_equal(actual[1], expected[1], counts[1]);
        if (FAILED(hr))
            std::cerr << "  2-source batch readback failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
    } else if (passed) {
        const HRESULT hr = context.copy_to_cpu_batch({
            {&a[0], actual[0].data(), counts[0]},
            {&a[1], actual[1].data(), counts[1]},
            {&a[2], actual[2].data(), counts[2]},
        });
        passed = SUCCEEDED(hr);
        for (int i = 0; passed && i < source_count; ++i)
            passed = bit_equal(actual[i], expected[i], counts[i]);
        if (FAILED(hr))
            std::cerr << "  3-source batch readback failed: HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
    }

    report(std::string("copy_to_cpu_batch with ") + std::to_string(source_count) +
           " differently sized sources", passed, failures);
}

template<class T>
void run_mixed_format_case(dxblas::Context& context, const char* type_name, int& failures)
{
    // This helper is intentionally called in float, int32, uint8, float order.
    run_size_case<T>(context, 4097, type_name, failures);
}

} // namespace

int main()
{
    // Request the D3D11 debug layer for this test run. If the SDK debug runtime
    // is unavailable, device creation may retry without it in the backend.
    dxblas::Context context(D3D_DRIVER_TYPE_HARDWARE, D3D11_CREATE_DEVICE_DEBUG);
    const HRESULT init_hr = context.initialize();
    if (FAILED(init_hr)) {
        std::cerr << "Context initialization failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(init_hr) << std::dec << '\n';
        return 2;
    }

    int failures = 0;
    run_type_sizes<float>(context, "float", failures);
    run_type_sizes<std::int32_t>(context, "int32", failures);
    run_type_sizes<std::uint8_t>(context, "uint8", failures);
    run_type_sizes<std::int16_t>(context, "int16", failures);

    run_reupload_case<float>(context, "float", failures);
    run_move_semantics_case(context, failures);
    run_shrink_case<float>(context, "float", failures);
    run_explicit_shape_case(context, failures);
    run_reuse_arithmetic_case(context, 100, 50, "100 -> 50", failures);
    run_reuse_arithmetic_case(context, 32769, 255, "32769 -> 255", failures);
    run_prefix_case<float>(context, "float", failures);
    run_batch_size_case(context, 2, failures);
    run_batch_size_case(context, 3, failures);

    run_mixed_format_case<float>(context, "mixed float #1", failures);
    run_mixed_format_case<std::int32_t>(context, "mixed int32", failures);
    run_mixed_format_case<std::uint8_t>(context, "mixed uint8", failures);
    run_mixed_format_case<float>(context, "mixed float #2", failures);

    std::cout << "\nCopy test summary: " << (failures == 0 ? "PASS" : "FAIL")
              << " (" << failures << " failed case(s))\n";
    return failures == 0 ? 0 : 1;
}
