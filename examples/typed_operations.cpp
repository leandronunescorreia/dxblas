#include <dxblas/dxblas.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize dxblas: 0x" << std::hex << hr << '\n';
        return 1;
    }

    // The same Context can execute different typed operations. Each call maps
    // its C++ element type to a DXGI format and selects a cached shader variant.
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{10.0f, 20.0f, 30.0f};
    std::vector<float> floats;
    hr = context.sum(a, b, floats);
    if (FAILED(hr)) return 2;

    const std::vector<std::uint32_t> x{1u, 2u, 3u};
    const std::vector<std::uint32_t> y{100u, 200u, 300u};
    std::vector<std::uint32_t> integers;
    hr = context.sum(x, y, integers);
    if (FAILED(hr)) return 3;

    const std::vector<std::uint8_t> p{2u, 3u, 4u};
    const std::vector<std::uint8_t> q{5u, 6u, 7u};
    std::vector<std::uint8_t> bytes;
    hr = context.mul(p, q, bytes);
    if (FAILED(hr)) return 4;

    std::cout << "float: " << floats[0] << ", " << floats[1] << ", " << floats[2] << '\n';
    std::cout << "uint32: " << integers[0] << ", " << integers[1] << ", " << integers[2] << '\n';
    std::cout << "uint8: " << static_cast<unsigned>(bytes[0]) << ", "
              << static_cast<unsigned>(bytes[1]) << ", "
              << static_cast<unsigned>(bytes[2]) << '\n';
    return 0;
}
