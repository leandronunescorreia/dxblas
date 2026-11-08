#pragma once

#include <cstddef>
#include <d3d11.h>
#include <dxgiformat.h>
#include <wrl/client.h>
#include <vector>

namespace dxblas {

// Non-templated GPU resource payload shared by gpuMem<T>, the backend and
// ArenaTexture. The texture is owned by the memory object; ArenaTexture only
// creates/uploads/reads it and does not maintain a separate texture-resource wrapper.
class gpuMemBase {
  public:
    virtual ~gpuMemBase() = default;
    virtual const void* cpu_data() const noexcept {
        return nullptr;
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv;
    UINT width = 0;
    UINT height = 0;
    std::size_t count = 0;
    DXGI_FORMAT storage_format = DXGI_FORMAT_UNKNOWN;
    // True only for shapes explicitly requested by matrix/GEMM operations.
    // Ordinary gpuMem buffers use canonical 1D layouts when their count changes.
    bool shape_is_user_defined = false;
};

// One destination in a batched GPU-to-CPU transfer. The caller owns both the
// gpuMem object and destination memory for the duration of the call.
struct GpuMemReadbackRequest {
    gpuMemBase* source = nullptr;
    void* destination = nullptr;
    std::size_t count = 0;
};

} // namespace dxblas
