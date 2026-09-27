#pragma once

#include <cstddef>
#include <memory>

#include <d3d11.h>
#include <dxgiformat.h>
#include <windows.h>

#include <dxblas/backend.hpp>

namespace dxblas {

class Context {
public:
    Context(
        D3D_DRIVER_TYPE driver_type = D3D_DRIVER_TYPE_HARDWARE,
        UINT device_flags = 0);

    explicit Context(std::unique_ptr<IBackend> backend);

    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    HRESULT initialize();

    HRESULT sum(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count);

    HRESULT mul(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count);

    IBackend* backend() noexcept;
    const IBackend* backend() const noexcept;

private:
    std::unique_ptr<IBackend> backend_;
};

struct NativeD3D11 {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
};

HRESULT get_native_d3d11(
    Context& context,
    NativeD3D11& native);

} // namespace dxblas
