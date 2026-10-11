#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <d3d11.h>
#include <dxgiformat.h>
#include <wrl/client.h>
#include <dxblas/gpu_mem.hpp>

namespace dxblas::detail {

UINT bytes_per_element(DXGI_FORMAT format);
DXGI_FORMAT staging_format_for(DXGI_FORMAT format);

struct Layout1D {
    UINT width;
    UINT height;
};

HRESULT layout_1d(std::size_t count, Layout1D& out);

class ArenaTexture final {
public:
    ArenaTexture() = default;
    ~ArenaTexture() = default;

    ArenaTexture(const ArenaTexture&) = delete;
    ArenaTexture& operator=(const ArenaTexture&) = delete;

    HRESULT initialize(ID3D11Device* device);
    HRESULT initialize_adaptive(ID3D11Device* device);

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

    HRESULT upload_float(ID3D11DeviceContext* context, const void* data, std::size_t count, std::shared_ptr<gpuMemBase>& out);
    HRESULT acquire_float(std::size_t count, std::shared_ptr<gpuMemBase>& out);
    HRESULT acquire_matrix_float(UINT width, UINT height, std::shared_ptr<gpuMemBase>& out);
    HRESULT upload_matrix_float(ID3D11DeviceContext* context, const float* data, UINT width, UINT height, std::shared_ptr<gpuMemBase>& out);
    HRESULT readback_float(ID3D11DeviceContext* context, const std::shared_ptr<gpuMemBase>& source, void* destination, std::size_t count);

    HRESULT copy_gpu_mem_buffers(ID3D11DeviceContext* context, gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t count);
    HRESULT upload_gpu_mem(ID3D11DeviceContext* context, gpuMemBase& memory, const void* data, std::size_t count);
    HRESULT prepare_gpu_mem(gpuMemBase& memory, DXGI_FORMAT format, std::size_t count);
    HRESULT prepare_gpu_mem_matrix(gpuMemBase& memory, DXGI_FORMAT format, UINT width, UINT height);
    HRESULT readback_gpu_mem(ID3D11DeviceContext* context, gpuMemBase& memory, void* out, std::size_t count);
    HRESULT readback_gpu_mem_batch(ID3D11DeviceContext* context, const std::vector<GpuMemReadbackRequest>& requests);

    HRESULT create_gpu_buffer(
        ID3D11DeviceContext* context, DXGI_FORMAT format, const void* data,
        std::size_t count, std::shared_ptr<gpuMemBase>& out);
    HRESULT copy_gpu_buffers(
        ID3D11DeviceContext* context, DXGI_FORMAT format, const void* a,
        const void* b, std::size_t count, std::shared_ptr<void>& gpu_a,
        std::shared_ptr<void>& gpu_b, std::shared_ptr<void>& gpu_out);
    HRESULT readback_gpu_buffer(
        ID3D11DeviceContext* context, const std::shared_ptr<void>& array,
        void* out, std::size_t count);
    // Returns the texture resource associated with a gpuMem handle. The arena owns
    // allocation, views, upload and readback; the backend owns all draw/render state.
    std::shared_ptr<gpuMemBase> texture_from_handle(const std::shared_ptr<void>& handle) const noexcept;

    UINT capacity_width() const noexcept { return capacity_width_; }
    UINT capacity_height() const noexcept { return capacity_height_; }

private:
    HRESULT allocate(
        DXGI_FORMAT input_format,
        DXGI_FORMAT output_format,
        UINT width,
        UINT height);
    HRESULT ensure_readback_staging(DXGI_FORMAT format, UINT width, UINT height);

    Microsoft::WRL::ComPtr<ID3D11Device> device_;

    // Shared staging resource for sequential and batched readbacks. Batch requests
    // are packed into separate vertical regions and mapped once per format/pass.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> shared_staging_;
    DXGI_FORMAT shared_staging_format_ = DXGI_FORMAT_UNKNOWN;
    UINT shared_staging_width_ = 0;
    UINT shared_staging_height_ = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> input_a_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> input_b_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> output_;

    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_a_srv_;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_b_srv_;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> output_rtv_;

    DXGI_FORMAT input_format_ = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT output_format_ = DXGI_FORMAT_UNKNOWN;
    UINT capacity_width_ = 0;
    UINT capacity_height_ = 0;
    std::vector<std::shared_ptr<gpuMemBase>> gpu_slots_;
};

} // namespace dxblas::detail
