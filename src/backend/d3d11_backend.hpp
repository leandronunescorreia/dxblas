#pragma once

#include <cstddef>
#include <memory>

#include <d3d11.h>
#include <dxgiformat.h>

#include <dxblas/backend.hpp>

namespace dxblas::detail {

class D3D11Backend final : public IBackend {
public:
    D3D11Backend(
        D3D_DRIVER_TYPE driver_type,
        UINT device_flags);

    ~D3D11Backend() override;

    D3D11Backend(const D3D11Backend&) = delete;
    D3D11Backend& operator=(const D3D11Backend&) = delete;
    D3D11Backend(D3D11Backend&&) noexcept;
    D3D11Backend& operator=(D3D11Backend&&) noexcept;

    HRESULT initialize() override;

    HRESULT sum(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count) override;

    HRESULT mul(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count) override;

    HRESULT gemm(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t n, std::size_t k) override;

    HRESULT copy_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t count) override;
    HRESULT readback_gpu_mem(gpuMemBase& source, void* out, std::size_t count) override;
    HRESULT sum_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t count) override;

    HRESULT upload_gpu(DXGI_FORMAT format, const void* data, std::size_t count, std::shared_ptr<void>& out) override;
    HRESULT copy_gpu(DXGI_FORMAT format, const void* a, const void* b, std::size_t count, std::shared_ptr<void>& gpu_a, std::shared_ptr<void>& gpu_b, std::shared_ptr<void>& gpu_out) override;
    HRESULT readback_copy_gpu(const std::shared_ptr<void>& array, void* out, std::size_t count) override;
    HRESULT sum_texture_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b, std::size_t count, std::shared_ptr<void>& out) override;
    HRESULT sum_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b, std::size_t count, std::shared_ptr<void>& out) override;
    HRESULT mul_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b, std::size_t count, std::shared_ptr<void>& out) override;
    HRESULT readback_gpu(const std::shared_ptr<void>& array, void* out, std::size_t count) override;

    HRESULT get_native(
        ID3D11Device** device,
        ID3D11DeviceContext** context) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dxblas::detail
