#include "backend/d3d11_backend.hpp"
#include "backend/arena_texture.hpp"
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
};

HRESULT get_format_info(DXGI_FORMAT format, FormatInfo& info)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT:
        info = {1};
        return S_OK;
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
        info = {2};
        return S_OK;
    case DXGI_FORMAT_R16_FLOAT:
        info = {2};
        return S_OK;
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
        info = {4};
        return S_OK;
    case DXGI_FORMAT_R32_FLOAT:
        info = {4};
        return S_OK;
    default:
        return E_INVALIDARG;
    }
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

        // The arena is context-owned and lives for the lifetime of the backend.
        // Its first allocation is deliberately modest; ensure() grows it only
        // when an operation actually needs more texture space.
        return arena_texture.initialize(device.Get(), 2048, 1);
    }

    HRESULT compile_shaders(
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

    HRESULT render(
        ID3D11VertexShader* vertex_shader,
        ID3D11PixelShader* pixel_shader,
        UINT width,
        UINT height)
    {
        D3D11_VIEWPORT viewport{};
        viewport.Width = static_cast<float>(width);
        viewport.Height = static_cast<float>(height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;

        ID3D11RenderTargetView* output = arena_texture.output_rtv();

        context->OMSetRenderTargets(1, &output, nullptr);
        context->RSSetViewports(1, &viewport);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex_shader, nullptr, 0);
        context->PSSetShader(pixel_shader, nullptr, 0);

        ID3D11ShaderResourceView* srvs[] = {
            arena_texture.input_a_srv(),
            arena_texture.input_b_srv()
        };
        context->PSSetShaderResources(0, 2, srvs);

        context->Draw(3, 0);

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);

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

        constexpr UINT kTileWidth = 4096;
        constexpr UINT kTileHeight = 4096;
        constexpr std::size_t kTileElements =
            static_cast<std::size_t>(kTileWidth) * kTileHeight;

        ComPtr<ID3D11VertexShader> vertex_shader;
        ComPtr<ID3D11PixelShader> pixel_shader;
        hr = compile_shaders(op, format, vertex_shader, pixel_shader);
        if (FAILED(hr))
            return hr;

        const auto* src_a = static_cast<const std::uint8_t*>(a);
        const auto* src_b = static_cast<const std::uint8_t*>(b);
        auto* dst = static_cast<std::uint8_t*>(out);

        std::size_t offset = 0;

        while (offset < count) {
            const std::size_t remaining = count - offset;
            const std::size_t tile_elements =
                std::min(remaining, kTileElements);

            const UINT width = static_cast<UINT>(
                std::min<std::size_t>(kTileWidth, tile_elements));
            const UINT height = static_cast<UINT>(
                (tile_elements + width - 1u) / width);

            const UINT row_pitch = width * info.bytes_per_element;
            // Integer render targets use native integer formats and integer
            // pixel-shader signatures; no UNORM byte-packing is required.
            const DXGI_FORMAT output_format = format;

            hr = arena_texture.ensure(
                format, output_format, width, height);
            if (FAILED(hr))
                return hr;

            const void* upload_a =
                src_a + offset * info.bytes_per_element;
            const void* upload_b =
                src_b + offset * info.bytes_per_element;

            // One CPU->GPU transfer for each input. The arena textures are
            // reused between operations instead of being created every call.
            arena_texture.upload(
                context.Get(),
                upload_a,
                upload_b,
                row_pitch,
                width,
                height);

            hr = render(
                vertex_shader.Get(),
                pixel_shader.Get(),
                width,
                height);
            if (FAILED(hr))
                return hr;

            // One GPU->CPU transfer for the result. Mapping the persistent
            // staging texture also synchronizes the GPU work before returning.
            hr = arena_texture.readback(
                context.Get(),
                dst + offset * info.bytes_per_element,
                width,
                height,
                static_cast<UINT>(tile_elements),
                info.bytes_per_element,
                false);
            if (FAILED(hr))
                return hr;

            offset += tile_elements;
        }

        return S_OK;
    }


    HRESULT render_gpu(
        BinaryOp op,
        const std::shared_ptr<GpuTexture>& a,
        const std::shared_ptr<GpuTexture>& b,
        const std::shared_ptr<GpuTexture>& output)
    {
        if (!a || !b || !output || a->count != b->count || output->count != a->count)
            return E_INVALIDARG;
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(op, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr)) return hr;

        D3D11_VIEWPORT viewport{};
        viewport.Width = static_cast<float>(output->width);
        viewport.Height = static_cast<float>(output->height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        ID3D11RenderTargetView* rtv = output->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->RSSetViewports(1, &viewport);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = { a->srv.Get(), b->srv.Get() };
        context->PSSetShaderResources(0, 2, srvs);
        context->Draw(3, 0);
        ID3D11ShaderResourceView* null_srvs[] = { nullptr, nullptr };
        context->PSSetShaderResources(0, 2, null_srvs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        return S_OK;
    }

    HRESULT upload_gpu(DXGI_FORMAT format, const void* data, std::size_t count, std::shared_ptr<void>& out)
    {
        if (format != DXGI_FORMAT_R32_FLOAT || !data || count == 0) return E_INVALIDARG;
        std::shared_ptr<GpuTexture> texture;
        HRESULT hr = arena_texture.upload_float(context.Get(), data, count, texture);
        if (FAILED(hr)) return hr;
        out = std::move(texture);
        return S_OK;
    }

    HRESULT binary_gpu(
        BinaryOp op,
        const std::shared_ptr<void>& a_opaque,
        const std::shared_ptr<void>& b_opaque,
        std::size_t count,
        std::shared_ptr<void>& out)
    {
        if (!a_opaque || !b_opaque || count == 0) return E_INVALIDARG;
        auto a = std::static_pointer_cast<GpuTexture>(a_opaque);
        auto b = std::static_pointer_cast<GpuTexture>(b_opaque);
        if (a->count != count || b->count != count) return E_INVALIDARG;
        std::shared_ptr<GpuTexture> output;
        HRESULT hr = arena_texture.acquire_float(count, output);
        if (FAILED(hr)) return hr;
        hr = render_gpu(op, a, b, output);
        if (FAILED(hr)) return hr;
        out = std::move(output);
        return S_OK;
    }

    HRESULT readback_gpu(const std::shared_ptr<void>& array, void* out, std::size_t count)
    {
        if (!array || !out) return E_INVALIDARG;
        auto texture = std::static_pointer_cast<GpuTexture>(array);
        if (count != texture->count) return E_INVALIDARG;
        return arena_texture.readback_float(context.Get(), texture, out, count);
    }

    HRESULT gemm(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t n, std::size_t k)
    {
        if (!a || !b || !c || m == 0 || n == 0 || k == 0)
            return E_INVALIDARG;
        constexpr std::size_t max_dimension = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        if (m > max_dimension || n > max_dimension || k > max_dimension)
            return E_INVALIDARG;
        if (m > std::numeric_limits<std::size_t>::max() / k ||
            k > std::numeric_limits<std::size_t>::max() / n ||
            m > std::numeric_limits<std::size_t>::max() / n)
            return E_INVALIDARG;

        const UINT rows = static_cast<UINT>(m);
        const UINT columns = static_cast<UINT>(n);
        const UINT inner = static_cast<UINT>(k);

        std::shared_ptr<GpuTexture> matrix_a;
        std::shared_ptr<GpuTexture> matrix_b;
        std::shared_ptr<GpuTexture> matrix_c;
        HRESULT hr = arena_texture.upload_matrix_float(context.Get(), a, inner, rows, matrix_a);
        if (FAILED(hr)) return hr;
        hr = arena_texture.upload_matrix_float(context.Get(), b, columns, inner, matrix_b);
        if (FAILED(hr)) return hr;
        hr = arena_texture.acquire_matrix_float(columns, rows, matrix_c);
        if (FAILED(hr)) return hr;

        ComPtr<ID3D11VertexShader> vertex_shader;
        ComPtr<ID3D11PixelShader> pixel_shader;
        hr = compile_shaders(BinaryOp::Gemm, DXGI_FORMAT_R32_FLOAT,
                             vertex_shader, pixel_shader);
        if (FAILED(hr)) return hr;

        struct GemmConstants { UINT m, n, k, padding; } constants{rows, columns, inner, 0};
        D3D11_BUFFER_DESC cb_desc{};
        cb_desc.ByteWidth = sizeof(GemmConstants);
        cb_desc.Usage = D3D11_USAGE_DEFAULT;
        cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        ComPtr<ID3D11Buffer> constant_buffer;
        hr = device->CreateBuffer(&cb_desc, nullptr, &constant_buffer);
        if (FAILED(hr)) return hr;
        context->UpdateSubresource(constant_buffer.Get(), 0, nullptr, &constants, 0, 0);

        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex_shader.Get(), nullptr, 0);
        context->PSSetShader(pixel_shader.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {matrix_a->srv.Get(), matrix_b->srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        ID3D11Buffer* cbs[] = {constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);
        ID3D11RenderTargetView* rtv = matrix_c->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);

        // Render a limited number of rows per draw to reduce the chance that a
        // very long pixel-shader invocation triggers the Windows GPU watchdog.
        // This remains a pixel-shader path: no compute shader or UAV is used.
        constexpr UINT rows_per_draw = 8;
        for (UINT row_start = 0; row_start < rows; row_start += rows_per_draw) {
            const UINT batch_rows = std::min(rows_per_draw, rows - row_start);
            D3D11_VIEWPORT viewport{};
            viewport.TopLeftX = 0.0f;
            viewport.TopLeftY = static_cast<float>(row_start);
            viewport.Width = static_cast<float>(columns);
            viewport.Height = static_cast<float>(batch_rows);
            viewport.MinDepth = 0.0f;
            viewport.MaxDepth = 1.0f;
            context->RSSetViewports(1, &viewport);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);

        return arena_texture.readback_float(context.Get(), matrix_c, c, m * n);
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
    ArenaTexture arena_texture;
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

HRESULT D3D11Backend::gemm(const float* a, const float* b, float* c,
                           std::size_t m, std::size_t n, std::size_t k)
{
    return impl_->gemm(a, b, c, m, n, k);
}


HRESULT D3D11Backend::upload_gpu(DXGI_FORMAT format, const void* data, std::size_t count, std::shared_ptr<void>& out)
{
    return impl_->upload_gpu(format, data, count, out);
}
HRESULT D3D11Backend::sum_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b, std::size_t count, std::shared_ptr<void>& out)
{
    return impl_->binary_gpu(BinaryOp::Add, a, b, count, out);
}
HRESULT D3D11Backend::mul_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b, std::size_t count, std::shared_ptr<void>& out)
{
    return impl_->binary_gpu(BinaryOp::Mul, a, b, count, out);
}
HRESULT D3D11Backend::readback_gpu(const std::shared_ptr<void>& array, void* out, std::size_t count)
{
    return impl_->readback_gpu(array, out, count);
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
