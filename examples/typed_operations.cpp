#include <dxblas/dxblas.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
template <class T>
bool roundtrip(dxblas::Context& context, const std::vector<T>& input, const char* label) {
    dxblas::gpuMem<T> a(input), b(input), out(input.size());
    HRESULT hr = context.copy(a, b, out, input.size());
    if (FAILED(hr)) {
        std::cerr << label << " upload failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return false;
    }
    std::vector<T> result(input.size());
    hr = context.copy_to_cpu(a, result.data(), result.size());
    if (FAILED(hr)) {
        std::cerr << label << " readback failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return false;
    }
    if (std::memcmp(result.data(), input.data(), input.size() * sizeof(T)) != 0) {
        std::cerr << label << " roundtrip mismatch\n";
        return false;
    }
    std::cout << label << ": " << result.size() << " elements uploaded and read back\n";
    return true;
}
} // namespace

int main() {
    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Context::initialize failed (HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << ")\n";
        return 1;
    }

    // gpuMem currently performs arithmetic for FP32; all five storage types can
    // use the same texture upload/readback path.
    if (!roundtrip(context, std::vector<std::uint8_t>{2, 3, 4, 5}, "uint8"))
        return 2;
    if (!roundtrip(context, std::vector<std::int16_t>{-2, 3, -4, 5}, "int16"))
        return 3;
    if (!roundtrip(context, std::vector<std::int32_t>{-20, 30, -40, 50}, "int32"))
        return 4;
    if (!roundtrip(context,
                   std::vector<dxblas::float16>{dxblas::float16{0x3c00}, dxblas::float16{0xc000}},
                   "float16"))
        return 5;

    const std::vector<float> a_cpu{1.0f, 2.0f, 3.0f};
    const std::vector<float> b_cpu{10.0f, 20.0f, 30.0f};
    dxblas::gpuMem<float> a(a_cpu), b(b_cpu), out(a_cpu.size());
    hr = context.copy(a, b, out, a_cpu.size());
    if (FAILED(hr))
        return 6;
    hr = context.sum(a, b, out, a_cpu.size());
    if (FAILED(hr))
        return 7;
    std::vector<float> result(a_cpu.size());
    hr = context.copy_to_cpu(out, result.data(), result.size());
    if (FAILED(hr))
        return 8;
    std::cout << "float32 gpuMem sum: ";
    for (float value : result)
        std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
