#include "backend/d3d11_backend.hpp"
#include "backend/d3d11_shader_manager.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace dxblas::detail {

namespace {

struct FormatInfo {
    UINT bytes_per_element;
    bool integer;
};

HRESULT get_format_info(DXGI_FORMAT format, FormatInfo& info)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
        info = {1, true};
        return S_OK;
    case DXGI_FORMAT_R16_UINT:
        info = {2, true};
        return S_OK;
    case DXGI_FORMAT_R32_UINT:
        info = {4, true};
        return S_OK;
    case DXGI_FORMAT_R32_FLOAT:
        info = {4, false};
        return S_OK;
    default:
        return E_INVALIDARG;
    }
}

HRESULT create_texture(
    ID3D11Device* device,
    UINT width,
    UINT height,
    DXGI_FORMAT format,
    UINT bind_flags,
    D3D11_USAGE usage,
    UINT cpu_access_flags,
    const void* initial_data,
    UINT row_pitch,
    ComPtr<ID3D11Texture2D>& texture)
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = usage;
    desc.BindFlags = bind_flags;
    desc.CPUAccessFlags = cpu_access_flags;

    D3D11_SUBRESOURCE_DATA data{};
    if (initial_data) {
        data.pSysMem = initial_data;
        data.SysMemPitch = row_pitch;
    }

    return device->CreateTexture2D(
        &desc,
        initial_data ? &data : nullptr,
        &texture);
}

HRESULT create_texture_srv(
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
    return device->CreateShaderResourceView(texture, &desc, &srv);
}

HRESULT create_texture_rtv(
    ID3D11Device* device,
    ID3D11Texture2D* texture,
    DXGI_FORMAT format,
    ComPtr<ID3D11RenderTargetView>& rtv)
{
    D3D11_RENDER_TARGET_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipSlice = 0;
    return device->CreateRenderTargetView(texture, &desc, &rtv);
}

HRESULT create_staging_texture(
    ID3D11Device* device,
    UINT width,
    UINT height,
    DXGI_FORMAT format,
    ComPtr<ID3D11Texture2D>& texture)
{
    return create_texture(
        device,
        width,
        height,
        format,
        0,
        D3D11_USAGE_STAGING,
        D3D11_CPU_ACCESS_READ,
        nullptr,
        0,
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

        if (FAILED(hr) && (device_flags & D3D11_CREATE_DEVICE_DEBUG)) {
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

        if (FAILED(hr))
            return hr;

        feature_level = created_level;
        return S_OK;
    }

    HRESULT step_two_creating_input_texture(
        DXGI_FORMAT format,
        UINT width,
        UINT height,
        const void* data,
        UINT row_pitch,
        ComPtr<ID3D11Texture2D>& texture,
        ComPtr<ID3D11ShaderResourceView>& srv)
    {
        HRESULT hr = create_texture(
            device.Get(),
            width,
            height,
            format,
            D3D11_BIND_SHADER_RESOURCE,
            D3D11_USAGE_DEFAULT,
            0,
            data,
            row_pitch,
            texture);
        if (FAILED(hr))
            return hr;

        return create_texture_srv(device.Get(), texture.Get(), format, srv);
    }

    HRESULT step_three_creating_output_texture(
        DXGI_FORMAT format,
        UINT width,
        UINT height,
        ComPtr<ID3D11Texture2D>& texture,
        ComPtr<ID3D11RenderTargetView>& rtv)
    {
        HRESULT hr = create_texture(
            device.Get(),
            width,
            height,
            format,
            D3D11_BIND_RENDER_TARGET,
            D3D11_USAGE_DEFAULT,
            0,
            nullptr,
            0,
            texture);
        if (FAILED(hr))
            return hr;

        return create_texture_rtv(device.Get(), texture.Get(), format, rtv);
    }

    HRESULT step_four_compiling_shaders(
        BinaryOp op,
        DXGI_FORMAT format,
        ComPtr<ID3D11VertexShader>& vertex_shader,
        ComPtr<ID3D11PixelShader>& pixel_shader)
    {
        ComPtr<ID3DBlob> vertex_bytecode;
        HRESULT hr = shader_manager.get_or_compile_fullscreen_vertex_shader(
            device.Get(), vertex_shader, vertex_bytecode);
        if (FAILED(hr))
            return hr;

        return shader_manager.get_or_compile(
            device.Get(), op, format, pixel_shader);
    }

    HRESULT step_five_rendering_the_shader(
        ID3D11VertexShader* vertex_shader,
        ID3D11PixelShader* pixel_shader,
        ID3D11ShaderResourceView* a,
        ID3D11ShaderResourceView* b,
        ID3D11RenderTargetView* output,
        UINT width,
        UINT height)
    {
        D3D11_VIEWPORT viewport{};
        viewport.TopLeftX = 0.0f;
        viewport.TopLeftY = 0.0f;
        viewport.Width = static_cast<float>(width);
        viewport.Height = static_cast<float>(height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;

        context->OMSetRenderTargets(1, &output, nullptr);
        context->RSSetViewports(1, &viewport);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex_shader, nullptr, 0);
        context->PSSetShader(pixel_shader, nullptr, 0);

        ID3D11ShaderResourceView* srvs[] = {a, b};
        context->PSSetShaderResources(0, 2, srvs);

        // Three vertices are generated by SV_VertexID. No vertex/index buffer is used.
        context->Draw(3, 0);

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);

        return S_OK;
    }

    HRESULT step_six_reading_back_texture(
        ID3D11Texture2D* output,
        ID3D11Texture2D* readback,
        void* destination,
        UINT width,
        UINT height,
        UINT elements,
        UINT bytes_per_element,
        bool packed_integer)
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

        const UINT last_row_elements = elements - (height - 1u) * width;
        auto* dst = static_cast<std::uint8_t*>(destination);
        const auto* src = static_cast<const std::uint8_t*>(mapped.pData);

        for (UINT y = 0; y < height; ++y) {
            const UINT row_elements = (y + 1u == height) ? last_row_elements : width;
            auto* row_dst = dst + static_cast<std::size_t>(y) * width * bytes_per_element;
            const auto* row_src = src + static_cast<std::size_t>(y) * mapped.RowPitch;

            if (!packed_integer) {
                std::memcpy(row_dst, row_src, row_elements * bytes_per_element);
                continue;
            }

            // Integer results are rendered into universally common R8G8B8A8_UNORM
            // and packed little-endian by the pixel shader. Reconstruct the
            // original scalar width before returning to the C++ caller.
            for (UINT x = 0; x < row_elements; ++x) {
                const auto* pixel = row_src + static_cast<std::size_t>(x) * 4u;
                const std::uint32_t value =
                    static_cast<std::uint32_t>(pixel[0]) |
                    (static_cast<std::uint32_t>(pixel[1]) << 8) |
                    (static_cast<std::uint32_t>(pixel[2]) << 16) |
                    (static_cast<std::uint32_t>(pixel[3]) << 24);
                std::memcpy(
                    row_dst + static_cast<std::size_t>(x) * bytes_per_element,
                    &value,
                    bytes_per_element);
            }
        }

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

        // Texture2D dimensions are bounded by the device. Keeping a moderate
        // tile size also avoids allocating enormous temporary textures.
        constexpr UINT kTileWidth = 4096;
        constexpr UINT kTileHeight = 4096;
        constexpr std::size_t kTileElements =
            static_cast<std::size_t>(kTileWidth) * kTileHeight;

        ComPtr<ID3D11VertexShader> vertex_shader;
        ComPtr<ID3D11PixelShader> pixel_shader;
        hr = step_four_compiling_shaders(op, format, vertex_shader, pixel_shader);
        if (FAILED(hr))
            return hr;

        const auto* src_a = static_cast<const std::uint8_t*>(a);
        const auto* src_b = static_cast<const std::uint8_t*>(b);
        auto* dst = static_cast<std::uint8_t*>(out);

        std::vector<std::uint8_t> padded_a;
        std::vector<std::uint8_t> padded_b;

        std::size_t offset = 0;
        while (offset < count) {
            const std::size_t remaining = count - offset;
            const std::size_t tile_elements = std::min(remaining, kTileElements);
            const UINT width = static_cast<UINT>(std::min<std::size_t>(kTileWidth, tile_elements));
            const UINT height = static_cast<UINT>((tile_elements + width - 1u) / width);
            const std::size_t texture_elements = static_cast<std::size_t>(width) * height;
            const UINT row_pitch = width * info.bytes_per_element;
            const DXGI_FORMAT output_format =
                info.integer ? DXGI_FORMAT_R8G8B8A8_UNORM : format;
            const UINT tile_bytes = static_cast<UINT>(texture_elements * info.bytes_per_element);

            const void* upload_a = src_a + offset * info.bytes_per_element;
            const void* upload_b = src_b + offset * info.bytes_per_element;

            // The last texture row may contain padding pixels. D3D11 uploads
            // the entire texture, so pad those pixels with zeroes. The output
            // copy below only copies the real element count.
            if (texture_elements != tile_elements) {
                padded_a.assign(tile_bytes, 0);
                padded_b.assign(tile_bytes, 0);
                const UINT real_bytes = static_cast<UINT>(tile_elements * info.bytes_per_element);
                std::memcpy(padded_a.data(), upload_a, real_bytes);
                std::memcpy(padded_b.data(), upload_b, real_bytes);
                upload_a = padded_a.data();
                upload_b = padded_b.data();
            }

            ComPtr<ID3D11Texture2D> input_a;
            ComPtr<ID3D11Texture2D> input_b;
            ComPtr<ID3D11ShaderResourceView> srv_a;
            ComPtr<ID3D11ShaderResourceView> srv_b;
            ComPtr<ID3D11Texture2D> output_texture;
            ComPtr<ID3D11RenderTargetView> output_rtv;
            ComPtr<ID3D11Texture2D> readback;

            hr = step_two_creating_input_texture(
                format, width, height, upload_a, row_pitch, input_a, srv_a);
            if (FAILED(hr)) return hr;

            hr = step_two_creating_input_texture(
                format, width, height, upload_b, row_pitch, input_b, srv_b);
            if (FAILED(hr)) return hr;

            hr = step_three_creating_output_texture(
                output_format, width, height, output_texture, output_rtv);
            if (FAILED(hr)) return hr;

            hr = create_staging_texture(
                device.Get(), width, height, output_format, readback);
            if (FAILED(hr)) return hr;

            hr = step_five_rendering_the_shader(
                vertex_shader.Get(),
                pixel_shader.Get(),
                srv_a.Get(),
                srv_b.Get(),
                output_rtv.Get(),
                width,
                height);
            if (FAILED(hr)) return hr;

            hr = step_six_reading_back_texture(
                output_texture.Get(),
                readback.Get(),
                dst + offset * info.bytes_per_element,
                width,
                height,
                static_cast<UINT>(tile_elements),
                info.bytes_per_element,
                info.integer);
            if (FAILED(hr)) return hr;

            offset += tile_elements;
        }

        return S_OK;
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
    D3D11ShaderManager shader_manager;
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
