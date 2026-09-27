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
    // Native D3D11 flags: no dxBLAS state enum is needed.
    dxblas::Context context(
        D3D_DRIVER_TYPE_HARDWARE,
        D3D11_CREATE_DEVICE_DEBUG);

    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("step_one_creating_the_device", hr);

        // The debug layer may not be installed. Retry without inventing a
        // library-specific flag/state.
        dxblas::Context fallback(
            D3D_DRIVER_TYPE_HARDWARE,
            0);

        hr = fallback.initialize();
        if (FAILED(hr)) {
            print_hresult("D3D11CreateDevice", hr);
            return 1;
        }

        context = std::move(fallback);
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
