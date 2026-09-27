#include <dxblas/dxblas.hpp>

#include "backend/d3d11_backend.hpp"

#include <utility>

namespace dxblas {

class Context::Impl {
public:
    Impl(D3D_DRIVER_TYPE driver_type, UINT device_flags)
        : backend(driver_type, device_flags)
    {
    }

    detail::D3D11Backend backend;
};

Context::Context(D3D_DRIVER_TYPE driver_type, UINT device_flags)
    : impl_(new Impl(driver_type, device_flags))
{
}

Context::~Context()
{
    delete impl_;
}

Context::Context(Context&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr))
{
}

Context& Context::operator=(Context&& other) noexcept
{
    if (this != &other) {
        delete impl_;
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}

HRESULT Context::initialize()
{
    return impl_->backend.initialize();
}

HRESULT Context::sum(
    DXGI_FORMAT format,
    const void* a,
    const void* b,
    void* out,
    std::size_t count)
{
    return impl_->backend.sum(format, a, b, out, count);
}

IBackend* Context::backend() noexcept
{
    return &impl_->backend;
}

const IBackend* Context::backend() const noexcept
{
    return &impl_->backend;
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
