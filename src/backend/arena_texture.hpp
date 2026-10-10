#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <d3d11.h>
#include <dxgiformat.h>
#include <wrl/client.h>

namespace dxblas::detail {

struct GpuTexture final {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv;
    UINT width = 0;
    UINT height = 0;
    std::size_t count = 0;
};

class ArenaTexture final {
public:
    ArenaTexture() = default;
    ~ArenaTexture() = default;

    ArenaTexture(const ArenaTexture&) = delete;
    ArenaTexture& operator=(const ArenaTexture&) = delete;

    HRESULT initialize(
        ID3D11Device* device,
        UINT initial_width = 2048,
        UINT initial_height = 1);

    HRESULT ensure(
        DXGI_FORMAT input_format,
        DXGI_FORMAT output_format,
        UINT width,
        UINT height);

    void upload(
        ID3D11DeviceContext* context,
        const void* a,
        const void* b,
        UINT row_pitch,
        UINT width,
        UINT height);

    HRESULT readback(
        ID3D11DeviceContext* context,
        void* destination,
        UINT width,
        UINT height,
        UINT elements,
        UINT bytes_per_element,
        bool packed_integer);

    ID3D11ShaderResourceView* input_a_srv() const noexcept { return input_a_srv_.Get(); }
    ID3D11ShaderResourceView* input_b_srv() const noexcept { return input_b_srv_.Get(); }
    ID3D11RenderTargetView* output_rtv() const noexcept { return output_rtv_.Get(); }

    HRESULT upload_float(ID3D11DeviceContext* context, const void* data, std::size_t count, std::shared_ptr<GpuTexture>& out);
    HRESULT acquire_float(std::size_t count, std::shared_ptr<GpuTexture>& out);
    HRESULT acquire_matrix_float(UINT width, UINT height, std::shared_ptr<GpuTexture>& out);
    HRESULT upload_matrix_float(ID3D11DeviceContext* context, const float* data, UINT width, UINT height, std::shared_ptr<GpuTexture>& out);
    HRESULT readback_float(ID3D11DeviceContext* context, const std::shared_ptr<GpuTexture>& source, void* destination, std::size_t count);

    UINT capacity_width() const noexcept { return capacity_width_; }
    UINT capacity_height() const noexcept { return capacity_height_; }

private:
    HRESULT allocate(
        DXGI_FORMAT input_format,
        DXGI_FORMAT output_format,
        UINT width,
        UINT height);

    Microsoft::WRL::ComPtr<ID3D11Device> device_;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> input_a_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> input_b_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> output_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> readback_;

    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_a_srv_;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_b_srv_;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> output_rtv_;

    DXGI_FORMAT input_format_ = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT output_format_ = DXGI_FORMAT_UNKNOWN;
    UINT capacity_width_ = 0;
    UINT capacity_height_ = 0;
    std::vector<std::shared_ptr<GpuTexture>> gpu_slots_;
};

} // namespace dxblas::detail
