#include "backend/d3d11_backend.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <d3dcompiler.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace dxblas::detail {

namespace {

enum class BinaryOp {
    Add,
    Mul,
};

struct ShaderKey {
    BinaryOp op;
    DXGI_FORMAT format;

    bool operator==(const ShaderKey& other) const noexcept
    {
        return op == other.op && format == other.format;
    }
};

struct ShaderKeyHash {
    std::size_t operator()(const ShaderKey& key) const noexcept
    {
        return (static_cast<std::size_t>(key.op) << 16) ^ static_cast<std::size_t>(key.format);
    }
};

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

const char* shader_source_for(BinaryOp op, DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R32_UINT:
        if (op == BinaryOp::Add) {
            return R"(
Buffer<uint> input_a : register(t0);
Buffer<uint> input_b : register(t1);
RWBuffer<uint> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] + input_b[id.x];
}
)";
        } else {
            return R"(
Buffer<uint> input_a : register(t0);
Buffer<uint> input_b : register(t1);
RWBuffer<uint> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] * input_b[id.x];
}
)";
        }
    case DXGI_FORMAT_R32_FLOAT:
        if (op == BinaryOp::Add) {
            return R"(
Buffer<float> input_a : register(t0);
Buffer<float> input_b : register(t1);
RWBuffer<float> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] + input_b[id.x];
}
)";
        } else {
            return R"(
Buffer<float> input_a : register(t0);
Buffer<float> input_b : register(t1);
RWBuffer<float> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] * input_b[id.x];
}
)";
        }
    default:
        return nullptr;
    }
}

HRESULT create_buffer(
    ID3D11Device* device,
    UINT byte_width,
    UINT bind_flags,
    D3D11_USAGE usage,
    UINT cpu_access_flags,
    ComPtr<ID3D11Buffer>& buffer)
{
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = byte_width;
    desc.Usage = usage;
    desc.BindFlags = bind_flags;
    desc.CPUAccessFlags = cpu_access_flags;

    return device->CreateBuffer(&desc, nullptr, &buffer);
}

HRESULT create_buffer_srv(
    ID3D11Device* device,
    ID3D11Buffer* buffer,
    DXGI_FORMAT format,
    UINT num_elements,
    ComPtr<ID3D11ShaderResourceView>& srv)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = num_elements;

    return device->CreateShaderResourceView(buffer, &desc, &srv);
}

HRESULT create_buffer_uav(
    ID3D11Device* device,
    ID3D11Buffer* buffer,
    DXGI_FORMAT format,
    UINT num_elements,
    ComPtr<ID3D11UnorderedAccessView>& uav)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = num_elements;
    desc.Buffer.Flags = 0;

    return device->CreateUnorderedAccessView(buffer, &desc, &uav);
}

HRESULT create_readback_buffer(
    ID3D11Device* device,
    UINT byte_width,
    ComPtr<ID3D11Buffer>& buffer)
{
    return create_buffer(
        device,
        byte_width,
        0,
        D3D11_USAGE_STAGING,
        D3D11_CPU_ACCESS_READ,
        buffer);
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

        if (FAILED(hr) && (device_flags & D3D11_CREATE_DEVICE_DEBUG)) {
            // The D3D11 debug layer (Graphics Tools) may not be installed.
            // Automatically fallback to device creation without the debug flag.
            device_flags &= ~D3D11_CREATE_DEVICE_DEBUG;

            hr = D3D11CreateDevice(
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
        }

        if (FAILED(hr)) {
            return hr;
        }

        feature_level = created_level;
        return S_OK;
    }

    HRESULT step_two_creating_the_buffer(
        UINT byte_width,
        UINT bind_flags,
        ComPtr<ID3D11Buffer>& buffer)
    {
        return create_buffer(
            device.Get(),
            byte_width,
            bind_flags,
            D3D11_USAGE_DEFAULT,
            0,
            buffer);
    }

    HRESULT step_three_uploading_the_buffer(
        ID3D11Buffer* buffer,
        const void* data,
        UINT byte_width)
    {
        context->UpdateSubresource(buffer, 0, nullptr, data, byte_width, 0);
        return S_OK;
    }

    HRESULT step_four_compiling_the_shader(
        BinaryOp op,
        DXGI_FORMAT format,
        ComPtr<ID3D11ComputeShader>& shader)
    {
        const ShaderKey key{op, format};
        auto it = shader_cache.find(key);
        if (it != shader_cache.end()) {
            shader = it->second;
            return S_OK;
        }

        const char* source = shader_source_for(op, format);
        if (!source)
            return E_INVALIDARG;

        ComPtr<ID3DBlob> bytecode;
        ComPtr<ID3DBlob> errors;

        const char* shader_name = (op == BinaryOp::Add) ? "dxblas_sum.hlsl" : "dxblas_mul.hlsl";

        HRESULT hr = D3DCompile(
            source,
            std::strlen(source),
            shader_name,
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

        hr = device->CreateComputeShader(
            bytecode->GetBufferPointer(),
            bytecode->GetBufferSize(),
            nullptr,
            &shader);

        if (SUCCEEDED(hr)) {
            shader_cache[key] = shader;
        }

        return hr;
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
        context->CSSetUnorderedAccessViews(0, 1, uavs, initial_counts);

        // One-dimensional dispatch: 64 threads per thread-group
        const UINT groups = (element_count + 63u) / 64u;
        context->Dispatch(groups, 1, 1);

        // Clean up bindings
        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->CSSetShaderResources(0, 2, null_srvs);

        ID3D11UnorderedAccessView* null_uavs[] = {nullptr};
        context->CSSetUnorderedAccessViews(0, 1, null_uavs, nullptr);

        context->CSSetShader(nullptr, nullptr, 0);
        return S_OK;
    }

    HRESULT step_six_reading_back_the_buffer(
        ID3D11Buffer* output,
        ID3D11Buffer* readback,
        void* destination,
        UINT byte_width)
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

        std::memcpy(destination, mapped.pData, byte_width);
        context->Unmap(readback, 0);
        return S_OK;
    }

    HRESULT binary_op(
        BinaryOp op,
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count)
    {
        if (!a || !b || !out || count == 0)
            return E_INVALIDARG;

        if (count > static_cast<std::size_t>(std::numeric_limits<UINT>::max()))
            return E_INVALIDARG;

        FormatInfo info{};
        HRESULT hr = get_format_info(format, info);
        if (FAILED(hr))
            return hr;

        const UINT element_count = static_cast<UINT>(count);
        const UINT byte_width = element_count * info.bytes_per_element;

        ComPtr<ID3D11Buffer> buffer_a;
        ComPtr<ID3D11Buffer> buffer_b;
        ComPtr<ID3D11Buffer> buffer_out;
        ComPtr<ID3D11Buffer> readback;
        ComPtr<ID3D11ShaderResourceView> srv_a;
        ComPtr<ID3D11ShaderResourceView> srv_b;
        ComPtr<ID3D11UnorderedAccessView> uav_out;
        ComPtr<ID3D11ComputeShader> shader;

        // Step 2: Create buffers
        hr = step_two_creating_the_buffer(byte_width, D3D11_BIND_SHADER_RESOURCE, buffer_a);
        if (FAILED(hr)) return hr;

        hr = step_two_creating_the_buffer(byte_width, D3D11_BIND_SHADER_RESOURCE, buffer_b);
        if (FAILED(hr)) return hr;

        hr = step_two_creating_the_buffer(byte_width, D3D11_BIND_UNORDERED_ACCESS, buffer_out);
        if (FAILED(hr)) return hr;

        // Step 3: Upload input data
        hr = step_three_uploading_the_buffer(buffer_a.Get(), a, byte_width);
        if (FAILED(hr)) return hr;

        hr = step_three_uploading_the_buffer(buffer_b.Get(), b, byte_width);
        if (FAILED(hr)) return hr;

        // Create buffer views
        hr = create_buffer_srv(device.Get(), buffer_a.Get(), format, element_count, srv_a);
        if (FAILED(hr)) return hr;

        hr = create_buffer_srv(device.Get(), buffer_b.Get(), format, element_count, srv_b);
        if (FAILED(hr)) return hr;

        hr = create_buffer_uav(device.Get(), buffer_out.Get(), format, element_count, uav_out);
        if (FAILED(hr)) return hr;

        // Staging buffer for readback
        hr = create_readback_buffer(device.Get(), byte_width, readback);
        if (FAILED(hr)) return hr;

        // Step 4: Compile shader (cached)
        hr = step_four_compiling_the_shader(op, format, shader);
        if (FAILED(hr)) return hr;

        // Step 5: Dispatch shader
        hr = step_five_dispatching_the_shader(
            shader.Get(), srv_a.Get(), srv_b.Get(), uav_out.Get(), element_count);
        if (FAILED(hr)) return hr;

        // Step 6: Readback result
        return step_six_reading_back_the_buffer(
            buffer_out.Get(), readback.Get(), out, byte_width);
    }

    HRESULT sum(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count)
    {
        return binary_op(BinaryOp::Add, format, a, b, out, count);
    }

    HRESULT mul(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count)
    {
        return binary_op(BinaryOp::Mul, format, a, b, out, count);
    }

    D3D_DRIVER_TYPE driver_type;
    UINT device_flags;
    D3D_FEATURE_LEVEL feature_level{};

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::unordered_map<ShaderKey, ComPtr<ID3D11ComputeShader>, ShaderKeyHash> shader_cache;
    std::string shader_error;
};

D3D11Backend::D3D11Backend(
    D3D_DRIVER_TYPE driver_type,
    UINT device_flags)
    : impl_(std::make_unique<Impl>(driver_type, device_flags))
{
}

D3D11Backend::~D3D11Backend() = default;
D3D11Backend::D3D11Backend(D3D11Backend&&) noexcept = default;
D3D11Backend& D3D11Backend::operator=(D3D11Backend&&) noexcept = default;

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

HRESULT D3D11Backend::mul(
    DXGI_FORMAT format,
    const void* a,
    const void* b,
    void* out,
    std::size_t count)
{
    return impl_->mul(format, a, b, out, count);
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
