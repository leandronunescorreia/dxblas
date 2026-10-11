#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include <d3d11.h>
#include <dxgiformat.h>
#include <windows.h>

#include <dxblas/backend.hpp>

namespace dxblas {

// Maps supported C++ element types to the corresponding single-channel DXGI
// storage format. Unsupported types fail at compile time when used by a typed
// Context operation.
template<class T> struct dxgi_format;
template<> struct dxgi_format<float> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R32_FLOAT; };
template<> struct dxgi_format<std::uint8_t> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R8_UINT; };
template<> struct dxgi_format<std::uint16_t> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R16_UINT; };
template<> struct dxgi_format<std::uint32_t> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R32_UINT; };
template<> struct dxgi_format<std::int8_t> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R8_SINT; };
template<> struct dxgi_format<std::int16_t> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R16_SINT; };
template<> struct dxgi_format<std::int32_t> { static constexpr DXGI_FORMAT value = DXGI_FORMAT_R32_SINT; };
template<class T> inline constexpr DXGI_FORMAT dxgi_format_v = dxgi_format<T>::value;


// IEEE-754 binary16 storage type. Arithmetic/conversion helpers can be added
// separately; this type deliberately preserves the 16-bit payload verbatim.
struct float16 {
    std::uint16_t bits = 0;
    constexpr float16() noexcept = default;
    explicit constexpr float16(std::uint16_t raw_bits) noexcept : bits(raw_bits) {}
};

// Typed host staging object for data that will be mapped into the GPU texture
// arena by subsequent operations. This first API step owns the CPU payload and
// records its intended DXGI element format; it does not allocate a separate
// texture per object.
template<class T> struct gpu_mem_traits;
template<> struct gpu_mem_traits<std::uint8_t> { static constexpr DXGI_FORMAT format = DXGI_FORMAT_R8_UINT; };
template<> struct gpu_mem_traits<std::int16_t> { static constexpr DXGI_FORMAT format = DXGI_FORMAT_R16_SINT; };
template<> struct gpu_mem_traits<std::int32_t> { static constexpr DXGI_FORMAT format = DXGI_FORMAT_R32_SINT; };
template<> struct gpu_mem_traits<float16> { static constexpr DXGI_FORMAT format = DXGI_FORMAT_R16_FLOAT; };
template<> struct gpu_mem_traits<float> { static constexpr DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT; };

template<class T>
class gpuMem {
    static_assert(
        std::is_same_v<T, std::uint8_t> ||
        std::is_same_v<T, std::int16_t> ||
        std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, float16> ||
        std::is_same_v<T, float>,
        "gpuMem<T> supports uint8_t, int16_t, int32_t, dxblas::float16 and float");
public:
    using value_type = T;
    gpuMem() = default;
    explicit gpuMem(std::size_t count) : count_(count), host_data_(count) {}
    gpuMem(const T* data, std::size_t count) : count_(count) { assign(data, count); }
    explicit gpuMem(const std::vector<T>& data) : count_(data.size()), host_data_(data) {}
    gpuMem(std::initializer_list<T> data) : count_(data.size()), host_data_(data) {}

    void assign(const T* data, std::size_t count) {
        if (count != 0 && data == nullptr) throw std::invalid_argument("gpuMem data must not be null");
        texture_address_.reset();
        count_ = count;
        if (count == 0) { host_data_.clear(); return; }
        host_data_.assign(data, data + count);
    }
    void assign(const std::vector<T>& data) { assign(data.data(), data.size()); }
    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    DXGI_FORMAT format() const noexcept { return gpu_mem_traits<T>::format; }
    bool resident() const noexcept { return static_cast<bool>(texture_address_); }

    // Before Context::copy this is the CPU upload payload. After a successful
    // copy, the payload is discarded and the object retains only its GPU handle.
    const T* data() const noexcept { return host_data_.empty() ? nullptr : host_data_.data(); }
    std::vector<T>& host_data() noexcept { return host_data_; }
    const std::vector<T>& host_data() const noexcept { return host_data_; }

private:
    friend class Context;
    std::size_t count_ = 0;
    std::vector<T> host_data_;
    std::shared_ptr<void> texture_address_;
};

// A reference-counted GPU-resident float array. Its texture storage is returned
// to the context-owned texture arena when the last handle is released.
class GpuArray {
public:
    GpuArray() = default;
    std::size_t size() const noexcept { return count_; }
    DXGI_FORMAT format() const noexcept { return format_; }
    explicit operator bool() const noexcept { return static_cast<bool>(resource_); }

private:
    friend class Context;
    GpuArray(std::shared_ptr<void> resource, std::size_t count, DXGI_FORMAT format)
        : resource_(std::move(resource)), count_(count), format_(format) {}
    std::shared_ptr<void> resource_;
    std::size_t count_ = 0;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
};

class Context {
public:
    Context(D3D_DRIVER_TYPE driver_type = D3D_DRIVER_TYPE_HARDWARE, UINT device_flags = 0);
    explicit Context(std::unique_ptr<IBackend> backend);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    HRESULT initialize();
    // Runtime-format API for dynamically typed buffers.
    HRESULT sum(DXGI_FORMAT format, const void* a, const void* b, void* out, std::size_t count);
    HRESULT mul(DXGI_FORMAT format, const void* a, const void* b, void* out, std::size_t count);

    // Row-major single-precision matrix multiplication: C[M,N] = A[M,K] * B[K,N].
    // This uses the D3D11 pixel-shader/render-to-texture path, not compute shaders.
    HRESULT gemm(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t n, std::size_t k);

    // Typed convenience API. Type selection is per call, not per Context.
    template<class T>
    HRESULT sum(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out)
    {
        if (a.size() != b.size()) return E_INVALIDARG;
        out.resize(a.size());
        if (a.empty()) return S_OK;
        return sum(dxgi_format_v<T>, a.data(), b.data(), out.data(), a.size());
    }

    template<class T>
    HRESULT mul(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out)
    {
        if (a.size() != b.size()) return E_INVALIDARG;
        out.resize(a.size());
        if (a.empty()) return S_OK;
        return mul(dxgi_format_v<T>, a.data(), b.data(), out.data(), a.size());
    }

    // GPU-resident path currently supports DXGI_FORMAT_R32_FLOAT.
    HRESULT upload_gpu(DXGI_FORMAT format, const void* data, std::size_t count, GpuArray& out);
    HRESULT sum_gpu(const GpuArray& a, const GpuArray& b, GpuArray& out);
    HRESULT mul_gpu(const GpuArray& a, const GpuArray& b, GpuArray& out);
    HRESULT readback_gpu(const GpuArray& array, void* out, std::size_t count);

    // Upload both CPU inputs into backend-owned D3D11 textures and allocate
    // an output texture. gpuMem then retains the texture address/handle only.
    template<class T>
    HRESULT copy(gpuMem<T>& a, gpuMem<T>& b, gpuMem<T>& out, std::size_t count)
    {
        if (!backend_) return E_POINTER;
        if (count > a.size() || count > b.size() || count > out.size()) return E_INVALIDARG;
        if (count == 0) return S_OK;
        if (a.data() == nullptr || b.data() == nullptr) return E_INVALIDARG;
        std::shared_ptr<void> gpu_a, gpu_b, gpu_out;
        HRESULT hr = backend_->copy_gpu(a.format(), a.data(), b.data(), count, gpu_a, gpu_b, gpu_out);
        if (FAILED(hr)) return hr;
        a.texture_address_ = std::move(gpu_a);
        b.texture_address_ = std::move(gpu_b);
        out.texture_address_ = std::move(gpu_out);
        a.host_data_.clear(); a.host_data_.shrink_to_fit();
        b.host_data_.clear(); b.host_data_.shrink_to_fit();
        out.host_data_.clear(); out.host_data_.shrink_to_fit();
        a.count_ = b.count_ = out.count_ = count;
        return S_OK;
    }

    // Execute an elementwise GPU sum using gpuMem texture handles. The first
    // implementation intentionally supports FP32, using a D3D11 pixel shader.
    template<class T>
    HRESULT sum(gpuMem<T>& a, gpuMem<T>& b, gpuMem<T>& out, std::size_t count)
    {
        if (!backend_) return E_POINTER;
        if (count > a.size() || count > b.size() || count > out.size()) return E_INVALIDARG;
        if (count == 0) return S_OK;
        if constexpr (!std::is_same_v<T, float>) {
            return E_NOTIMPL;
        } else {
            if (!a.texture_address_ || !b.texture_address_) return E_INVALIDARG;
            std::shared_ptr<void> gpu_out;
            HRESULT hr = backend_->sum_texture_gpu(a.texture_address_, b.texture_address_, count, gpu_out);
            if (FAILED(hr)) return hr;
            out.texture_address_ = std::move(gpu_out);
            out.host_data_.clear(); out.host_data_.shrink_to_fit();
            out.count_ = count;
            return S_OK;
        }
    }

    // Read a typed gpuMem allocation back into caller-owned CPU memory. The
    // backend performs the D3D11 staging copy and row-pitch-aware transfer.
    template<class T>
    HRESULT copy_to_cpu(gpuMem<T>& source, T* destination, std::size_t count)
    {
        if (!backend_) return E_POINTER;
        if (count > source.size() || (count != 0 && destination == nullptr)) return E_INVALIDARG;
        if (count == 0) return S_OK;
        if (!source.texture_address_) return E_INVALIDARG;
        return backend_->readback_copy_gpu(source.texture_address_, destination, count);
    }

    IBackend* backend() noexcept;
    const IBackend* backend() const noexcept;

private:
    std::unique_ptr<IBackend> backend_;
};

struct NativeD3D11 {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
};

HRESULT get_native_d3d11(Context& context, NativeD3D11& native);

} // namespace dxblas
