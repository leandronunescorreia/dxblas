#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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
