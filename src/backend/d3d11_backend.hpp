#pragma once

#include <cstddef>
#include <memory>

#include <d3d11.h>
#include <dxgiformat.h>
#include <wrl/client.h>

#include <dxblas/backend.hpp>

namespace dxblas::detail {

class D3D11Backend final : public IBackend {
public:
    D3D11Backend(
        D3D_DRIVER_TYPE driver_type,
        UINT device_flags);

    ~D3D11Backend() override;

    HRESULT initialize();

    HRESULT sum(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count) override;

    HRESULT get_native(
        ID3D11Device** device,
        ID3D11DeviceContext** context) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dxblas::detail
