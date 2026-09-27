#include "backend/d3d11_backend.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <d3dcompiler.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace dxblas::detail {

namespace {

struct FormatInfo {
    UINT bytes_per_element;
    const char* hlsl_type;
};

HRESULT get_format_info(DXGI_FORMAT format, FormatInfo& info)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
        info = {1, "uint"};
        return S_OK;
    case DXGI_FORMAT_R16_UINT:
        info = {2, "uint"};
        return S_OK;
    case DXGI_FORMAT_R32_UINT:
        info = {4, "uint"};
        return S_OK;
    case DXGI_FORMAT_R32_FLOAT:
        info = {4, "float"};
        return S_OK;
    default:
        return E_INVALIDARG;
    }
}

const char* shader_source_for(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R32_UINT:
        return R"(
RWTexture2D<uint> output_texture : register(u0);
Texture2D<uint> input_a : register(t0);
Texture2D<uint> input_b : register(t1);

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 p = uint2(dispatch_thread_id.x, 0);
    output_texture[p] = input_a[p] + input_b[p];
}
)";
    case DXGI_FORMAT_R32_FLOAT:
        return R"(
RWTexture2D<float> output_texture : register(u0);
Texture2D<float> input_a : register(t0);
Texture2D<float> input_b : register(t1);

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 p = uint2(dispatch_thread_id.x, 0);
    output_texture[p] = input_a[p] + input_b[p];
}
)";
    default:
        return nullptr;
    }
}

HRESULT create_texture(
    ID3D11Device* device,
    DXGI_FORMAT format,
    UINT width,
    UINT height,
    UINT bind_flags,
    UINT usage,
    UINT cpu_access_flags,
    ComPtr<ID3D11Texture2D>& texture)
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = static_cast<D3D11_USAGE>(usage);
    desc.BindFlags = bind_flags;
    desc.CPUAccessFlags = cpu_access_flags;

    return device->CreateTexture2D(
        &desc,
        nullptr,
        &texture);
}

HRESULT create_srv(
    ID3D11Device* device,
    ID3D11Texture2D* texture,
    DXGI_FORMAT format,
    ComPtr<ID3D11ShaderResourceView>& srv)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MostDetailedMip = 0;
    desc.Texture2D.MipLevels = 1;

    return device->CreateShaderResourceView(
        texture, &desc, &srv);
}

HRESULT create_uav(
    ID3D11Device* device,
    ID3D11Texture2D* texture,
    DXGI_FORMAT format,
    ComPtr<ID3D11UnorderedAccessView>& uav)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipSlice = 0;

    return device->CreateUnorderedAccessView(
        texture, &desc, &uav);
}

HRESULT create_readback_texture(
    ID3D11Device* device,
    DXGI_FORMAT format,
    UINT width,
    UINT height,
    ComPtr<ID3D11Texture2D>& texture)
{
    return create_texture(
        device,
        format,
        width,
        height,
        0,
        D3D11_USAGE_STAGING,
        D3D11_CPU_ACCESS_READ,
        texture);
}

} // namespace

class D3D11Backend::Impl {
public:
    Impl(D3D_DRIVER_TYPE driver_type, UINT device_flags)
        : driver_type(driver_type),
          device_flags(device_flags)
    {
    }

    HRESULT step_one_creating_the_device()
    {
        D3D_FEATURE_LEVEL requested[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };

        D3D_FEATURE_LEVEL created_level{};

        HRESULT hr = D3D11CreateDevice(
            nullptr,
            driver_type,
            nullptr,
            device_flags,
            requested,
            ARRAYSIZE(requested),
            D3D11_SDK_VERSION,
            &device,
            &created_level,
            &context);

        if (FAILED(hr)) {
            return hr;
        }

        feature_level = created_level;
        return S_OK;
    }

    HRESULT step_two_creating_the_texture(
        DXGI_FORMAT format,
        UINT width,
        UINT height,
        UINT bind_flags,
        ComPtr<ID3D11Texture2D>& texture)
    {
        FormatInfo info{};
        HRESULT hr = get_format_info(format, info);
        if (FAILED(hr))
            return hr;

        return create_texture(
            device.Get(),
            format,
            width,
            height,
            bind_flags,
            D3D11_USAGE_DEFAULT,
            0,
            texture);
    }

    HRESULT step_three_uploading_the_texture(
        DXGI_FORMAT format,
        const void* data,
        UINT width,
        UINT height,
        ID3D11Texture2D* texture)
    {
        FormatInfo info{};
        HRESULT hr = get_format_info(format, info);
        if (FAILED(hr))
            return hr;

        const UINT row_pitch = width * info.bytes_per_element;

        // One-row texture for the first baby step. Keeping the row pitch
        // explicit makes it easy to inspect in the graphics debugger.
        context->UpdateSubresource(
            texture,
            0,
            nullptr,
            data,
            row_pitch,
            row_pitch * height);

        return S_OK;
    }

    HRESULT step_four_compiling_the_shader(
        DXGI_FORMAT format,
        ComPtr<ID3D11ComputeShader>& shader)
    {
        const char* source = shader_source_for(format);
        if (!source)
            return E_INVALIDARG;

        ComPtr<ID3DBlob> bytecode;
        ComPtr<ID3DBlob> errors;

        HRESULT hr = D3DCompile(
            source,
            std::strlen(source),
            "dxblas_sum.hlsl",
            nullptr,
            nullptr,
            "main",
            "cs_5_0",
            D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
            0,
            &bytecode,
            &errors);

        if (FAILED(hr)) {
            if (errors) {
                shader_error.assign(
                    static_cast<const char*>(errors->GetBufferPointer()),
                    errors->GetBufferSize());
            }
            return hr;
        }

        return device->CreateComputeShader(
            bytecode->GetBufferPointer(),
            bytecode->GetBufferSize(),
            nullptr,
            &shader);
    }

    HRESULT step_five_dispatching_the_shader(
        ID3D11ComputeShader* shader,
        ID3D11ShaderResourceView* a,
        ID3D11ShaderResourceView* b,
        ID3D11UnorderedAccessView* output,
        UINT element_count)
    {
        context->CSSetShader(shader, nullptr, 0);

        ID3D11ShaderResourceView* srvs[] = {a, b};
        context->CSSetShaderResources(0, 2, srvs);

        ID3D11UnorderedAccessView* uavs[] = {output};
        UINT initial_counts[] = {0};
        context->CSSetUnorderedAccessViews(
            0, 1, uavs, initial_counts);

        // One-dimensional sample: 64 threads per group.
        const UINT groups = (element_count + 63u) / 64u;
        context->Dispatch(groups, 1, 1);

        // Explicitly clear bindings. This is useful when stepping through
        // the API with the DirectX debugger.
        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->CSSetShaderResources(0, 2, null_srvs);

        ID3D11UnorderedAccessView* null_uavs[] = {nullptr};
        context->CSSetUnorderedAccessViews(
            0, 1, null_uavs, nullptr);

        context->CSSetShader(nullptr, nullptr, 0);
        return S_OK;
    }

    HRESULT step_six_reading_back_the_texture(
        ID3D11Texture2D* output,
        ID3D11Texture2D* readback,
        void* destination,
        UINT width,
        UINT height,
        UINT bytes_per_element)
    {
        context->CopyResource(readback, output);

        D3D11_MAPPED_SUBRESOURCE mapped{};
        HRESULT hr = context->Map(
            readback,
            0,
            D3D11_MAP_READ,
            0,
            &mapped);

        if (FAILED(hr))
            return hr;

        const UINT row_bytes = width * bytes_per_element;

        for (UINT y = 0; y < height; ++y) {
            std::memcpy(
                static_cast<std::uint8_t*>(destination) +
                    static_cast<std::size_t>(y) * row_bytes,
                static_cast<const std::uint8_t*>(mapped.pData) +
                    static_cast<std::size_t>(y) * mapped.RowPitch,
                row_bytes);
        }

        context->Unmap(readback, 0);
        return S_OK;
    }

    HRESULT sum(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count)
    {
        if (!a || !b || !out || count == 0)
            return E_INVALIDARG;

        if (count > static_cast<std::size_t>(
                        std::numeric_limits<UINT>::max()))
            return E_INVALIDARG;

        FormatInfo info{};
        HRESULT hr = get_format_info(format, info);
        if (FAILED(hr))
            return hr;

        const UINT width = static_cast<UINT>(count);
        const UINT height = 1;

        ComPtr<ID3D11Texture2D> texture_a;
        ComPtr<ID3D11Texture2D> texture_b;
        ComPtr<ID3D11Texture2D> texture_out;
        ComPtr<ID3D11Texture2D> readback;
        ComPtr<ID3D11ShaderResourceView> srv_a;
        ComPtr<ID3D11ShaderResourceView> srv_b;
        ComPtr<ID3D11UnorderedAccessView> uav_out;
        ComPtr<ID3D11ComputeShader> shader;

        hr = step_two_creating_the_texture(
            format, width, height,
            D3D11_BIND_SHADER_RESOURCE,
            texture_a);
        if (FAILED(hr)) return hr;

        hr = step_two_creating_the_texture(
            format, width, height,
            D3D11_BIND_SHADER_RESOURCE,
            texture_b);
        if (FAILED(hr)) return hr;

        hr = step_two_creating_the_texture(
            format, width, height,
            D3D11_BIND_UNORDERED_ACCESS,
            texture_out);
        if (FAILED(hr)) return hr;

        hr = step_three_uploading_the_texture(
            format, a, width, height, texture_a.Get());
        if (FAILED(hr)) return hr;

        hr = step_three_uploading_the_texture(
            format, b, width, height, texture_b.Get());
        if (FAILED(hr)) return hr;

        hr = create_srv(
            device.Get(), texture_a.Get(), format, srv_a);
        if (FAILED(hr)) return hr;

        hr = create_srv(
            device.Get(), texture_b.Get(), format, srv_b);
        if (FAILED(hr)) return hr;

        hr = create_uav(
            device.Get(), texture_out.Get(), format, uav_out);
        if (FAILED(hr)) return hr;

        hr = create_readback_texture(
            device.Get(), format, width, height, readback);
        if (FAILED(hr)) return hr;

        hr = step_four_compiling_the_shader(format, shader);
        if (FAILED(hr)) return hr;

        hr = step_five_dispatching_the_shader(
            shader.Get(),
            srv_a.Get(),
            srv_b.Get(),
            uav_out.Get(),
            width);
        if (FAILED(hr)) return hr;

        return step_six_reading_back_the_texture(
            texture_out.Get(),
            readback.Get(),
            out,
            width,
            height,
            info.bytes_per_element);
    }

    D3D_DRIVER_TYPE driver_type;
    UINT device_flags;
    D3D_FEATURE_LEVEL feature_level{};

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::string shader_error;
};

D3D11Backend::D3D11Backend(
    D3D_DRIVER_TYPE driver_type,
    UINT device_flags)
    : impl_(std::make_unique<Impl>(driver_type, device_flags))
{
}

D3D11Backend::~D3D11Backend() = default;

HRESULT D3D11Backend::initialize()
{
    return impl_->step_one_creating_the_device();
}

HRESULT D3D11Backend::sum(
    DXGI_FORMAT format,
    const void* a,
    const void* b,
    void* out,
    std::size_t count)
{
    return impl_->sum(format, a, b, out, count);
}

HRESULT D3D11Backend::get_native(
    ID3D11Device** device,
    ID3D11DeviceContext** context) const
{
    if (!device || !context)
        return E_POINTER;

    *device = impl_->device.Get();
    *context = impl_->context.Get();

    if (*device)
        (*device)->AddRef();
    if (*context)
        (*context)->AddRef();

    return (*device && *context) ? S_OK : E_FAIL;
}

} // namespace dxblas::detail
