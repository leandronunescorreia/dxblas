#include <dxblas/dxblas.hpp>

#include "backend/d3d11_backend.hpp"

#include <utility>

namespace dxblas {

Context::Context(D3D_DRIVER_TYPE driver_type, UINT device_flags)
    : backend_(std::make_unique<detail::D3D11Backend>(driver_type, device_flags))
{
}

Context::Context(std::unique_ptr<IBackend> backend)
    : backend_(std::move(backend))
{
}

Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

HRESULT Context::initialize()
{
    return backend_ ? backend_->initialize() : E_POINTER;
}

HRESULT Context::sum(
    DXGI_FORMAT format,
    const void* a,
    const void* b,
    void* out,
    std::size_t count)
{
    return backend_ ? backend_->sum(format, a, b, out, count) : E_POINTER;
}

IBackend* Context::backend() noexcept
{
    return backend_.get();
}

const IBackend* Context::backend() const noexcept
{
    return backend_.get();
}

HRESULT get_native_d3d11(Context& context, NativeD3D11& native)
{
    auto* backend =
        dynamic_cast<detail::D3D11Backend*>(context.backend());

    if (!backend)
        return E_NOINTERFACE;

    return backend->get_native(
        &native.device,
        &native.context);
}

} // namespace dxblas
