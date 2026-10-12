#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <dxgiformat.h>
#include <windows.h>

#include <dxblas/gpu_mem.hpp>

namespace dxblas {

class GpuArray;

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual HRESULT initialize() = 0;
    virtual HRESULT sum(DXGI_FORMAT format, const void* a, const void* b, void* out, std::size_t count) = 0;
    virtual HRESULT mul(DXGI_FORMAT format, const void* a, const void* b, void* out, std::size_t count) = 0;
    // Row-major FP32 GEMM: C[M,N] = A[M,K] * B[K,N].
    virtual HRESULT gemm(const float*, const float*, float*, std::size_t, std::size_t, std::size_t) { return E_NOTIMPL; }

    // Optional GPU-resident operations. Backends that do not implement them return E_NOTIMPL.
    virtual HRESULT copy_gpu_mem(gpuMemBase&, gpuMemBase&, gpuMemBase&, std::size_t) { return E_NOTIMPL; }
    virtual HRESULT readback_gpu_mem(gpuMemBase&, void*, std::size_t) { return E_NOTIMPL; }
    virtual HRESULT readback_gpu_mem_batch(const std::vector<GpuMemReadbackRequest>&) { return E_NOTIMPL; }
    virtual HRESULT sum_gpu_mem(gpuMemBase&, gpuMemBase&, gpuMemBase&, std::size_t) { return E_NOTIMPL; }
    virtual HRESULT mul_gpu_mem(gpuMemBase&, gpuMemBase&, gpuMemBase&, std::size_t) { return E_NOTIMPL; }
    virtual HRESULT gemm_gpu_mem(gpuMemBase&, gpuMemBase&, gpuMemBase&, std::size_t, std::size_t, std::size_t) { return E_NOTIMPL; }
    virtual HRESULT upload_gpu(DXGI_FORMAT, const void*, std::size_t, std::shared_ptr<void>&) { return E_NOTIMPL; }
    virtual HRESULT sum_gpu(const std::shared_ptr<void>&, const std::shared_ptr<void>&, std::size_t, std::shared_ptr<void>&) { return E_NOTIMPL; }
    virtual HRESULT mul_gpu(const std::shared_ptr<void>&, const std::shared_ptr<void>&, std::size_t, std::shared_ptr<void>&) { return E_NOTIMPL; }
    virtual HRESULT readback_gpu(const std::shared_ptr<void>&, void*, std::size_t) { return E_NOTIMPL; }
};

} // namespace dxblas
