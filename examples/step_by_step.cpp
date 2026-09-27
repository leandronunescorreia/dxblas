#include <dxblas/dxblas.hpp>

#include <cstdint>
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
    // Native D3D11 flags: debug layer requested, automatically falls back if not installed.
    dxblas::Context context(
        D3D_DRIVER_TYPE_HARDWARE,
        D3D11_CREATE_DEVICE_DEBUG);

    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("initialize", hr);
        return 1;
    }

    std::vector<std::uint32_t> a{1, 2, 3, 4, 5};
    std::vector<std::uint32_t> b{10, 20, 30, 40, 50};
    std::vector<std::uint32_t> out(a.size());

    // Native DirectX format.
    hr = context.sum(
        DXGI_FORMAT_R32_UINT,
        a.data(),
        b.data(),
        out.data(),
        out.size());

    if (FAILED(hr)) {
        print_hresult("sum", hr);
        return 2;
    }

    for (std::size_t i = 0; i < out.size(); ++i)
        std::cout << a[i] << " + " << b[i]
                  << " = " << out[i] << '\n';

    // Element-wise multiplication
    std::vector<std::uint32_t> mul_out(a.size());
    hr = context.mul(
        DXGI_FORMAT_R32_UINT,
        a.data(),
        b.data(),
        mul_out.data(),
        mul_out.size());

    if (FAILED(hr)) {
        print_hresult("mul (uint32)", hr);
        return 3;
    }

    std::cout << "Uint32 mul:\n";
    for (std::size_t i = 0; i < mul_out.size(); ++i)
        std::cout << a[i] << " * " << b[i]
                  << " = " << mul_out[i] << '\n';

    // Verify float computation and shader caching
    std::vector<float> fa{1.5f, 2.5f, 3.5f};
    std::vector<float> fb{10.0f, 20.0f, 30.0f};
    std::vector<float> fout(fa.size());

    hr = context.sum(
        DXGI_FORMAT_R32_FLOAT,
        fa.data(),
        fb.data(),
        fout.data(),
        fout.size());

    if (FAILED(hr)) {
        print_hresult("sum (float)", hr);
        return 4;
    }

    std::cout << "Float sum:\n";
    for (std::size_t i = 0; i < fout.size(); ++i)
        std::cout << fa[i] << " + " << fb[i]
                  << " = " << fout[i] << '\n';

    std::vector<float> fmul_out(fa.size());
    hr = context.mul(
        DXGI_FORMAT_R32_FLOAT,
        fa.data(),
        fb.data(),
        fmul_out.data(),
        fmul_out.size());

    if (FAILED(hr)) {
        print_hresult("mul (float)", hr);
        return 5;
    }

    std::cout << "Float mul:\n";
    for (std::size_t i = 0; i < fmul_out.size(); ++i)
        std::cout << fa[i] << " * " << fb[i]
                  << " = " << fmul_out[i] << '\n';

    dxblas::NativeD3D11 native{};
    hr = dxblas::get_native_d3d11(context, native);
    if (SUCCEEDED(hr)) {
        // These are the actual native D3D11 objects.
        std::cout << "ID3D11Device: " << native.device << '\n';
        std::cout << "ID3D11DeviceContext: " << native.context << '\n';

        native.context->Release();
        native.device->Release();
    }

    return 0;
}
