#include "backend/d3d11_backend.hpp"
#include "backend/arena_texture.hpp"
#include "backend/d3d11_shader_manager.hpp"
#include <dxblas/device_selection.hpp>

#include <memory>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <wrl/client.h>
#include <dxgi1_4.h>

using Microsoft::WRL::ComPtr;

namespace dxblas::detail {

class D3D11Backend::Impl {
  public:
    struct GemmConstants {
        UINT m, n, k, a_width, b_width, out_width, padding0, padding1;
    };
    struct TransposeConstants {
        UINT rows, cols, in_width, out_width;
    };
    struct AddVectorConstants {
        UINT rows, cols, mat_width, vec_width;
    };
    struct FusedGemmConstants {
        UINT m, n, k, a_width, b_width, c_width, out_width, padding;
    };

    Impl(D3D_DRIVER_TYPE driver_type, UINT device_flags)
        : driver_type(driver_type), device_flags(device_flags) {
    }

    HRESULT step_one_creating_the_device() {
        D3D_FEATURE_LEVEL created_level{};
        HRESULT hr = E_FAIL;

        if (driver_type == D3D_DRIVER_TYPE_HARDWARE) {
            // Select the hardware adapter with the largest dedicated VRAM.
            // The helper retries feature level 11_0 and retries without the
            // debug flag when necessary. It deliberately has no WARP fallback.
            hr = create_best_d3d11_device(device_flags, device, created_level, context);
        } else {
            // Preserve explicitly requested WARP/reference/software driver modes.
            const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
            const D3D_FEATURE_LEVEL fallback[] = {D3D_FEATURE_LEVEL_11_0};

            auto create_requested_device = [&](UINT flags) {
                device.Reset();
                context.Reset();
                created_level = {};
                hr = D3D11CreateDevice(nullptr, driver_type, nullptr, flags, requested,
                                       ARRAYSIZE(requested), D3D11_SDK_VERSION, &device,
                                       &created_level, &context);

                if (hr == E_INVALIDARG) {
                    device.Reset();
                    context.Reset();
                    created_level = {};
                    hr = D3D11CreateDevice(nullptr, driver_type, nullptr, flags, fallback,
                                           ARRAYSIZE(fallback), D3D11_SDK_VERSION, &device,
                                           &created_level, &context);
                }
            };

            create_requested_device(device_flags);
            if (FAILED(hr) && (device_flags & D3D11_CREATE_DEVICE_DEBUG)) {
                device_flags &= ~D3D11_CREATE_DEVICE_DEBUG;
                create_requested_device(device_flags);
            }
        }

        if (FAILED(hr))
            return hr;

        feature_level = created_level;

        return arena_texture.initialize_adaptive(device.Get());
    }

    HRESULT compile_shaders(BinaryOp op, DXGI_FORMAT format,
                            ComPtr<ID3D11VertexShader>& vertex_shader,
                            ComPtr<ID3D11PixelShader>& pixel_shader) {
        ComPtr<ID3DBlob> vertex_bytecode;
        HRESULT hr = shader_manager.get_or_compile_fullscreen_vertex_shader(
            device.Get(), vertex_shader, vertex_bytecode);
        if (FAILED(hr))
            return hr;

        return shader_manager.get_or_compile(device.Get(), op, format, pixel_shader);
    }

    HRESULT binary_op(BinaryOp op, DXGI_FORMAT format, const void* a, const void* b, void* out,
                      std::size_t count) {
        if (!a || !b || !out || count == 0)
            return E_INVALIDARG;
        const UINT bytes = bytes_per_element(format);
        if (bytes == 0)
            return E_INVALIDARG;
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(op, format, vs, ps);
        if (FAILED(hr))
            return hr;

        // Rendering is deliberately orchestrated here. ArenaTexture only creates,
        // uploads, and reads back resources; it never binds a render target or draws.
        constexpr UINT tile_width = 4096, tile_height = 4096;
        constexpr std::size_t tile_elements = static_cast<std::size_t>(tile_width) * tile_height;
        const auto* src_a = static_cast<const std::uint8_t*>(a);
        const auto* src_b = static_cast<const std::uint8_t*>(b);
        auto* dst = static_cast<std::uint8_t*>(out);
        std::size_t offset = 0;
        while (offset < count) {
            const std::size_t tile_count = std::min(count - offset, tile_elements);
            const UINT width = static_cast<UINT>(std::min<std::size_t>(tile_width, tile_count));
            const UINT height = static_cast<UINT>((tile_count + width - 1u) / width);
            hr = arena_texture.ensure(format, format, width, height);
            if (FAILED(hr))
                return hr;
            arena_texture.upload(context.Get(), src_a + offset * bytes, src_b + offset * bytes,
                                 width * bytes, width, height);

            D3D11_VIEWPORT viewport{};
            viewport.Width = static_cast<float>(width);
            viewport.Height = static_cast<float>(height);
            viewport.MinDepth = 0.0f;
            viewport.MaxDepth = 1.0f;
            ID3D11RenderTargetView* rtv = arena_texture.output_rtv();
            context->OMSetRenderTargets(1, &rtv, nullptr);
            context->RSSetViewports(1, &viewport);
            context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(vs.Get(), nullptr, 0);
            context->PSSetShader(ps.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srvs[] = {arena_texture.input_a_srv(),
                                                arena_texture.input_b_srv()};
            context->PSSetShaderResources(0, 2, srvs);
            context->Draw(3, 0);
            ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
            context->PSSetShaderResources(0, 2, null_srvs);
            context->PSSetShader(nullptr, nullptr, 0);
            context->VSSetShader(nullptr, nullptr, 0);
            context->OMSetRenderTargets(0, nullptr, nullptr);
            hr = arena_texture.readback(context.Get(), dst + offset * bytes, width, height,
                                        static_cast<UINT>(tile_count), bytes);
            if (FAILED(hr))
                return hr;
            offset += tile_count;
        }
        return S_OK;
    }

    HRESULT copy_gpu_mem(gpuMemBase& a, std::size_t count) {
        return arena_texture.copy_gpu_mem_buffer(context.Get(), a, count);
    }

    HRESULT copy_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t count) {
        return arena_texture.copy_gpu_mem_buffers(context.Get(), a, b, out, count);
    }

    HRESULT readback_gpu_mem(gpuMemBase& source, void* out, std::size_t count) {
        return arena_texture.readback_gpu_mem(context.Get(), source, out, count);
    }

    HRESULT readback_gpu_mem_batch(const std::vector<GpuMemReadbackRequest>& requests) {
        return arena_texture.readback_gpu_mem_batch(context.Get(), requests);
    }

    HRESULT binary_gpu_mem(BinaryOp op, gpuMemBase& a, gpuMemBase& b, gpuMemBase& out,
                           std::size_t count) {
        // Both inputs must use the same pixel row width. Their allocated heights
        // may differ as long as the requested prefix is present in each texture.
        if (a.storage_format != DXGI_FORMAT_R32_FLOAT ||
            b.storage_format != DXGI_FORMAT_R32_FLOAT ||
            out.storage_format != DXGI_FORMAT_R32_FLOAT || !a.texture.Get() || !a.srv.Get() ||
            !b.texture.Get() || !b.srv.Get() || count == 0 || count > a.count || count > b.count ||
            a.width == 0 || a.height == 0 || b.height == 0)
            return E_INVALIDARG;
        if (a.width != b.width)
            return E_INVALIDARG;
        if (out.texture.Get() == a.texture.Get() || out.texture.Get() == b.texture.Get())
            return E_INVALIDARG;

        const std::size_t rows_size = count / a.width + (count % a.width != 0 ? 1u : 0u);
        if (rows_size == 0 || rows_size > a.height || rows_size > b.height ||
            rows_size > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;
        const UINT rows = static_cast<UINT>(rows_size);

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(op, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        // The output must use the same pixel row width as the inputs. Treat this
        // as an operation-specific shape rather than retaining a prior GEMM shape.
        hr = arena_texture.prepare_gpu_mem_matrix(out, DXGI_FORMAT_R32_FLOAT, a.width, rows);
        if (FAILED(hr))
            return hr;
        out.shape_is_user_defined = false;
        if (!out.texture.Get() || !out.rtv.Get() || out.width != a.width || out.height < rows)
            return E_INVALIDARG;

        D3D11_VIEWPORT vp{};
        vp.Width = static_cast<float>(a.width);
        vp.Height = static_cast<float>(rows);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        ID3D11RenderTargetView* rtv = out.rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->RSSetViewports(1, &vp);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {a.srv.Get(), b.srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        context->Draw(3, 0);
        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        out.count = count;
        return S_OK;
    }

    HRESULT sum_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t count) {
        return binary_gpu_mem(BinaryOp::Add, a, b, out, count);
    }

    HRESULT mul_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t count) {
        return binary_gpu_mem(BinaryOp::Mul, a, b, out, count);
    }

    HRESULT update_gemm_constant_buffer(const GemmConstants& constants) {
        if (!gemm_constant_buffer) {
            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = sizeof(GemmConstants);
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            HRESULT hr = device->CreateBuffer(&bd, nullptr, &gemm_constant_buffer);
            if (FAILED(hr))
                return hr;
        }

        // Reuse the same DEFAULT-usage constant buffer for both GEMM paths.
        context->UpdateSubresource(gemm_constant_buffer.Get(), 0, nullptr, &constants, 0, 0);
        return S_OK;
    }

    HRESULT gemm_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t m,
                         std::size_t n, std::size_t k) {
        if (!m || !n || !k || a.storage_format != DXGI_FORMAT_R32_FLOAT ||
            b.storage_format != DXGI_FORMAT_R32_FLOAT ||
            out.storage_format != DXGI_FORMAT_R32_FLOAT || !a.texture.Get() || !a.srv.Get() ||
            !b.texture.Get() || !b.srv.Get())
            return E_INVALIDARG;
        if (m > static_cast<std::size_t>(-1) / k || k > static_cast<std::size_t>(-1) / n ||
            m > static_cast<std::size_t>(-1) / n || a.count < m * k || b.count < k * n)
            return E_INVALIDARG;
        if (m > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || n > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            k > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;

        HRESULT hr = arena_texture.prepare_gpu_mem_matrix(
            out, DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(n), static_cast<UINT>(m));
        if (FAILED(hr))
            return hr;
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        hr = compile_shaders(BinaryOp::Gemm, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const GemmConstants constants{static_cast<UINT>(m),
                                      static_cast<UINT>(n),
                                      static_cast<UINT>(k),
                                      a.width,
                                      b.width,
                                      out.width,
                                      0,
                                      0};
        hr = update_gemm_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        D3D11_VIEWPORT vp{};
        vp.Width = static_cast<float>(out.width);
        vp.Height = 1.0f;
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        ID3D11RenderTargetView* rtv = out.rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {a.srv.Get(), b.srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        ID3D11Buffer* cbs[] = {gemm_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);
        // Bound each draw to a small number of output rows to avoid long-running
        // pixel-shader workloads triggering the Windows GPU timeout detector.
        constexpr UINT rows_per_draw = 8;
        for (UINT start_row = 0; start_row < out.height; start_row += rows_per_draw) {
            D3D11_VIEWPORT tile_vp{};
            tile_vp.TopLeftY = static_cast<float>(start_row);
            tile_vp.Width = static_cast<float>(out.width);
            tile_vp.Height = static_cast<float>(std::min(rows_per_draw, out.height - start_row));
            tile_vp.MinDepth = 0.0f;
            tile_vp.MaxDepth = 1.0f;
            context->RSSetViewports(1, &tile_vp);
            context->Draw(3, 0);
        }
        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        out.count = m * n;
        return S_OK;
    }

    HRESULT update_transpose_constant_buffer(const TransposeConstants& constants) {
        if (!transpose_constant_buffer) {
            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = sizeof(TransposeConstants);
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            HRESULT hr = device->CreateBuffer(&bd, nullptr, &transpose_constant_buffer);
            if (FAILED(hr))
                return hr;
        }
        context->UpdateSubresource(transpose_constant_buffer.Get(), 0, nullptr, &constants, 0, 0);
        return S_OK;
    }

    HRESULT transpose_gpu_mem(gpuMemBase& a, gpuMemBase& out, std::size_t rows, std::size_t cols) {
        if (!rows || !cols || a.storage_format != DXGI_FORMAT_R32_FLOAT ||
            out.storage_format != DXGI_FORMAT_R32_FLOAT || !a.texture.Get() || !a.srv.Get())
            return E_INVALIDARG;
        if (rows > static_cast<std::size_t>(-1) / cols || a.count < rows * cols)
            return E_INVALIDARG;
        if (rows > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            cols > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;
        if (out.texture.Get() && out.texture.Get() == a.texture.Get())
            return E_INVALIDARG;

        // Output matrix has cols rows and rows cols: width = rows, height = cols.
        HRESULT hr = arena_texture.prepare_gpu_mem_matrix(
            out, DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(rows), static_cast<UINT>(cols));
        if (FAILED(hr))
            return hr;

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        hr = compile_shaders(BinaryOp::Transpose, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const TransposeConstants constants{static_cast<UINT>(rows), static_cast<UINT>(cols),
                                           a.width, out.width};
        hr = update_transpose_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        ID3D11RenderTargetView* rtv = out.rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {a.srv.Get()};
        context->PSSetShaderResources(0, 1, srvs);
        ID3D11Buffer* cbs[] = {transpose_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);

        constexpr UINT rows_per_draw = 64;
        for (UINT start_row = 0; start_row < out.height; start_row += rows_per_draw) {
            D3D11_VIEWPORT tile_vp{};
            tile_vp.TopLeftY = static_cast<float>(start_row);
            tile_vp.Width = static_cast<float>(out.width);
            tile_vp.Height = static_cast<float>(std::min(rows_per_draw, out.height - start_row));
            tile_vp.MinDepth = 0.0f;
            tile_vp.MaxDepth = 1.0f;
            context->RSSetViewports(1, &tile_vp);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr};
        context->PSSetShaderResources(0, 1, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        out.count = rows * cols;
        return S_OK;
    }

    HRESULT update_add_vector_constant_buffer(const AddVectorConstants& constants) {
        if (!add_vector_constant_buffer) {
            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = sizeof(AddVectorConstants);
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            HRESULT hr = device->CreateBuffer(&bd, nullptr, &add_vector_constant_buffer);
            if (FAILED(hr))
                return hr;
        }
        context->UpdateSubresource(add_vector_constant_buffer.Get(), 0, nullptr, &constants, 0, 0);
        return S_OK;
    }

    HRESULT add_vector_gpu_mem(gpuMemBase& matrix, gpuMemBase& vector, gpuMemBase& out,
                               std::size_t rows, std::size_t cols) {
        if (!rows || !cols || matrix.storage_format != DXGI_FORMAT_R32_FLOAT ||
            vector.storage_format != DXGI_FORMAT_R32_FLOAT ||
            out.storage_format != DXGI_FORMAT_R32_FLOAT || !matrix.texture.Get() ||
            !matrix.srv.Get() || !vector.texture.Get() || !vector.srv.Get())
            return E_INVALIDARG;
        if (rows > static_cast<std::size_t>(-1) / cols || matrix.count < rows * cols ||
            vector.count < cols)
            return E_INVALIDARG;
        if (rows > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            cols > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;
        if (out.texture.Get() && (out.texture.Get() == matrix.texture.Get() ||
                                  out.texture.Get() == vector.texture.Get()))
            return E_INVALIDARG;

        HRESULT hr = arena_texture.prepare_gpu_mem_matrix(
            out, DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(cols), static_cast<UINT>(rows));
        if (FAILED(hr))
            return hr;

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        hr = compile_shaders(BinaryOp::AddVector, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const AddVectorConstants constants{static_cast<UINT>(rows), static_cast<UINT>(cols),
                                           matrix.width, vector.width};
        hr = update_add_vector_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        ID3D11RenderTargetView* rtv = out.rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {matrix.srv.Get(), vector.srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        ID3D11Buffer* cbs[] = {add_vector_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);

        constexpr UINT rows_per_draw = 64;
        for (UINT start_row = 0; start_row < out.height; start_row += rows_per_draw) {
            D3D11_VIEWPORT tile_vp{};
            tile_vp.TopLeftY = static_cast<float>(start_row);
            tile_vp.Width = static_cast<float>(out.width);
            tile_vp.Height = static_cast<float>(std::min(rows_per_draw, out.height - start_row));
            tile_vp.MinDepth = 0.0f;
            tile_vp.MaxDepth = 1.0f;
            context->RSSetViewports(1, &tile_vp);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        out.count = rows * cols;
        return S_OK;
    }

    HRESULT update_fused_gemm_constant_buffer(const FusedGemmConstants& constants) {
        if (!fused_gemm_constant_buffer) {
            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = sizeof(FusedGemmConstants);
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            HRESULT hr = device->CreateBuffer(&bd, nullptr, &fused_gemm_constant_buffer);
            if (FAILED(hr))
                return hr;
        }
        context->UpdateSubresource(fused_gemm_constant_buffer.Get(), 0, nullptr, &constants, 0, 0);
        return S_OK;
    }

    HRESULT gemm_transpose_bias_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& c,
                                        gpuMemBase& out, std::size_t m, std::size_t n,
                                        std::size_t k) {
        if (!m || !n || !k || a.storage_format != DXGI_FORMAT_R32_FLOAT ||
            b.storage_format != DXGI_FORMAT_R32_FLOAT ||
            c.storage_format != DXGI_FORMAT_R32_FLOAT ||
            out.storage_format != DXGI_FORMAT_R32_FLOAT || !a.texture.Get() || !a.srv.Get() ||
            !b.texture.Get() || !b.srv.Get() || !c.texture.Get() || !c.srv.Get())
            return E_INVALIDARG;
        if (m > static_cast<std::size_t>(-1) / k || n > static_cast<std::size_t>(-1) / k ||
            m > static_cast<std::size_t>(-1) / n || a.count < m * k || b.count < n * k ||
            c.count < n)
            return E_INVALIDARG;
        if (m > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || n > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            k > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;
        if (out.texture.Get() &&
            (out.texture.Get() == a.texture.Get() || out.texture.Get() == b.texture.Get() ||
             out.texture.Get() == c.texture.Get()))
            return E_INVALIDARG;

        HRESULT hr = arena_texture.prepare_gpu_mem_matrix(
            out, DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(n), static_cast<UINT>(m));
        if (FAILED(hr))
            return hr;

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        hr = compile_shaders(BinaryOp::GemmTransposeBias, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const FusedGemmConstants constants{static_cast<UINT>(m),
                                           static_cast<UINT>(n),
                                           static_cast<UINT>(k),
                                           a.width,
                                           b.width,
                                           c.width,
                                           out.width,
                                           0};
        hr = update_fused_gemm_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        D3D11_VIEWPORT vp{};
        vp.Width = static_cast<float>(out.width);
        vp.Height = 1.0f;
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        ID3D11RenderTargetView* rtv = out.rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {a.srv.Get(), b.srv.Get(), c.srv.Get()};
        context->PSSetShaderResources(0, 3, srvs);
        ID3D11Buffer* cbs[] = {fused_gemm_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);

        constexpr UINT rows_per_draw = 8;
        for (UINT start_row = 0; start_row < out.height; start_row += rows_per_draw) {
            D3D11_VIEWPORT tile_vp{};
            tile_vp.TopLeftY = static_cast<float>(start_row);
            tile_vp.Width = static_cast<float>(out.width);
            tile_vp.Height = static_cast<float>(std::min(rows_per_draw, out.height - start_row));
            tile_vp.MinDepth = 0.0f;
            tile_vp.MaxDepth = 1.0f;
            context->RSSetViewports(1, &tile_vp);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr, nullptr};
        context->PSSetShaderResources(0, 3, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        out.count = m * n;
        return S_OK;
    }

    HRESULT render_binary_gpu(ID3D11VertexShader* vs, ID3D11PixelShader* ps,
                              const std::shared_ptr<void>& a_handle,
                              const std::shared_ptr<void>& b_handle, std::size_t count,
                              std::shared_ptr<void>& out) {
        if (!vs || !ps || !a_handle || !b_handle || count == 0)
            return E_INVALIDARG;
        auto a = arena_texture.texture_from_handle(a_handle);
        auto b = arena_texture.texture_from_handle(b_handle);
        if (!a || !b || a->count != count || b->count != count || a->width != b->width ||
            a->height != b->height)
            return E_INVALIDARG;
        std::shared_ptr<gpuMemBase> output;
        HRESULT hr = arena_texture.acquire_matrix_float(a->width, a->height, output);
        if (FAILED(hr))
            return hr;
        output->count = count;
        D3D11_VIEWPORT vp{};
        vp.Width = static_cast<float>(output->width);
        vp.Height = static_cast<float>(output->height);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        ID3D11RenderTargetView* rtv = output->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        context->RSSetViewports(1, &vp);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs, nullptr, 0);
        context->PSSetShader(ps, nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {a->srv.Get(), b->srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        context->Draw(3, 0);
        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        out = std::move(output);
        return S_OK;
    }

    HRESULT upload_gpu(DXGI_FORMAT format, const void* data, std::size_t count,
                       std::shared_ptr<void>& out) {
        std::shared_ptr<gpuMemBase> texture;
        HRESULT hr = arena_texture.create_gpu_buffer(context.Get(), format, data, count, texture);
        if (SUCCEEDED(hr))
            out = std::move(texture);
        return hr;
    }

    HRESULT binary_gpu(BinaryOp op, const std::shared_ptr<void>& a, const std::shared_ptr<void>& b,
                       std::size_t count, std::shared_ptr<void>& out) {
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(op, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;
        return render_binary_gpu(vs.Get(), ps.Get(), a, b, count, out);
    }

    HRESULT readback_gpu(const std::shared_ptr<void>& array, void* out, std::size_t count) {
        return arena_texture.readback_gpu_buffer(context.Get(), array, out, count);
    }

    HRESULT gemm(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                 std::size_t k) {
        if (!a || !b || !c || !m || !n || !k)
            return E_INVALIDARG;
        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(BinaryOp::Gemm, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;
        if (m > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || n > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            k > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;
        const UINT rows = static_cast<UINT>(m), cols = static_cast<UINT>(n),
                   inner = static_cast<UINT>(k);
        std::shared_ptr<gpuMemBase> ma, mb, mc;
        hr = arena_texture.upload_matrix_float(context.Get(), a, inner, rows, ma);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.upload_matrix_float(context.Get(), b, cols, inner, mb);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.acquire_matrix_float(cols, rows, mc);
        if (FAILED(hr))
            return hr;
        const GemmConstants constants{rows, cols, inner, inner, cols, cols, 0, 0};
        hr = update_gemm_constant_buffer(constants);
        if (FAILED(hr))
            return hr;
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {ma->srv.Get(), mb->srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        ID3D11Buffer* cbs[] = {gemm_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);
        ID3D11RenderTargetView* rtv = mc->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);
        constexpr UINT rows_per_draw = 8;
        for (UINT start = 0; start < rows; start += rows_per_draw) {
            const UINT batch = std::min(rows_per_draw, rows - start);
            D3D11_VIEWPORT vp{};
            vp.TopLeftY = static_cast<float>(start);
            vp.Width = static_cast<float>(cols);
            vp.Height = static_cast<float>(batch);
            vp.MinDepth = 0;
            vp.MaxDepth = 1;
            context->RSSetViewports(1, &vp);
            context->Draw(3, 0);
        }
        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        return arena_texture.readback_float(context.Get(), mc, c, m * n);
    }

    HRESULT sum(DXGI_FORMAT format, const void* a, const void* b, void* out, std::size_t count) {
        return binary_op(BinaryOp::Add, format, a, b, out, count);
    }

    HRESULT mul(DXGI_FORMAT format, const void* a, const void* b, void* out, std::size_t count) {
        return binary_op(BinaryOp::Mul, format, a, b, out, count);
    }

    HRESULT transpose(const float* a, float* out, std::size_t rows, std::size_t cols) {
        if (!a || !out || !rows || !cols)
            return E_INVALIDARG;
        if (rows > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            cols > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            rows > static_cast<std::size_t>(-1) / cols)
            return E_INVALIDARG;

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(BinaryOp::Transpose, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const UINT r = static_cast<UINT>(rows), c = static_cast<UINT>(cols);
        std::shared_ptr<gpuMemBase> ma, mout;
        hr = arena_texture.upload_matrix_float(context.Get(), a, c, r, ma);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.acquire_matrix_float(r, c, mout);
        if (FAILED(hr))
            return hr;

        const TransposeConstants constants{r, c, c, r};
        hr = update_transpose_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {ma->srv.Get()};
        context->PSSetShaderResources(0, 1, srvs);
        ID3D11Buffer* cbs[] = {transpose_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);
        ID3D11RenderTargetView* rtv = mout->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);

        constexpr UINT rows_per_draw = 64;
        for (UINT start = 0; start < c; start += rows_per_draw) {
            const UINT batch = std::min(rows_per_draw, c - start);
            D3D11_VIEWPORT vp{};
            vp.TopLeftY = static_cast<float>(start);
            vp.Width = static_cast<float>(r);
            vp.Height = static_cast<float>(batch);
            vp.MinDepth = 0;
            vp.MaxDepth = 1;
            context->RSSetViewports(1, &vp);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr};
        context->PSSetShaderResources(0, 1, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        return arena_texture.readback_float(context.Get(), mout, out, rows * cols);
    }

    HRESULT add_vector(const float* matrix, const float* vector, float* out, std::size_t rows,
                       std::size_t cols) {
        if (!matrix || !vector || !out || !rows || !cols)
            return E_INVALIDARG;
        if (rows > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            cols > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            rows > static_cast<std::size_t>(-1) / cols)
            return E_INVALIDARG;

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(BinaryOp::AddVector, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const UINT r = static_cast<UINT>(rows), c = static_cast<UINT>(cols);
        std::shared_ptr<gpuMemBase> mm, mv, mout;
        hr = arena_texture.upload_matrix_float(context.Get(), matrix, c, r, mm);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.upload_matrix_float(context.Get(), vector, c, 1, mv);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.acquire_matrix_float(c, r, mout);
        if (FAILED(hr))
            return hr;

        const AddVectorConstants constants{r, c, c, c};
        hr = update_add_vector_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {mm->srv.Get(), mv->srv.Get()};
        context->PSSetShaderResources(0, 2, srvs);
        ID3D11Buffer* cbs[] = {add_vector_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);
        ID3D11RenderTargetView* rtv = mout->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);

        constexpr UINT rows_per_draw = 64;
        for (UINT start = 0; start < r; start += rows_per_draw) {
            const UINT batch = std::min(rows_per_draw, r - start);
            D3D11_VIEWPORT vp{};
            vp.TopLeftY = static_cast<float>(start);
            vp.Width = static_cast<float>(c);
            vp.Height = static_cast<float>(batch);
            vp.MinDepth = 0;
            vp.MaxDepth = 1;
            context->RSSetViewports(1, &vp);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr};
        context->PSSetShaderResources(0, 2, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        return arena_texture.readback_float(context.Get(), mout, out, rows * cols);
    }

    HRESULT gemm_transpose_bias(const float* a, const float* b, const float* c, float* out,
                                std::size_t m, std::size_t n, std::size_t k) {
        if (!a || !b || !c || !out || !m || !n || !k)
            return E_INVALIDARG;
        if (m > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || n > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            k > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;

        ComPtr<ID3D11VertexShader> vs;
        ComPtr<ID3D11PixelShader> ps;
        HRESULT hr = compile_shaders(BinaryOp::GemmTransposeBias, DXGI_FORMAT_R32_FLOAT, vs, ps);
        if (FAILED(hr))
            return hr;

        const UINT rows = static_cast<UINT>(m), cols = static_cast<UINT>(n),
                   inner = static_cast<UINT>(k);
        std::shared_ptr<gpuMemBase> ma, mb, mc, mout;
        hr = arena_texture.upload_matrix_float(context.Get(), a, inner, rows, ma);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.upload_matrix_float(context.Get(), b, inner, cols, mb);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.upload_matrix_float(context.Get(), c, cols, 1, mc);
        if (FAILED(hr))
            return hr;
        hr = arena_texture.acquire_matrix_float(cols, rows, mout);
        if (FAILED(hr))
            return hr;

        const FusedGemmConstants constants{rows, cols, inner, inner, inner, cols, cols, 0};
        hr = update_fused_gemm_constant_buffer(constants);
        if (FAILED(hr))
            return hr;

        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {ma->srv.Get(), mb->srv.Get(), mc->srv.Get()};
        context->PSSetShaderResources(0, 3, srvs);
        ID3D11Buffer* cbs[] = {fused_gemm_constant_buffer.Get()};
        context->PSSetConstantBuffers(0, 1, cbs);
        ID3D11RenderTargetView* rtv = mout->rtv.Get();
        context->OMSetRenderTargets(1, &rtv, nullptr);

        constexpr UINT rows_per_draw = 8;
        for (UINT start = 0; start < rows; start += rows_per_draw) {
            const UINT batch = std::min(rows_per_draw, rows - start);
            D3D11_VIEWPORT vp{};
            vp.TopLeftY = static_cast<float>(start);
            vp.Width = static_cast<float>(cols);
            vp.Height = static_cast<float>(batch);
            vp.MinDepth = 0;
            vp.MaxDepth = 1;
            context->RSSetViewports(1, &vp);
            context->Draw(3, 0);
        }

        ID3D11ShaderResourceView* null_srvs[] = {nullptr, nullptr, nullptr};
        context->PSSetShaderResources(0, 3, null_srvs);
        ID3D11Buffer* null_cbs[] = {nullptr};
        context->PSSetConstantBuffers(0, 1, null_cbs);
        context->PSSetShader(nullptr, nullptr, 0);
        context->VSSetShader(nullptr, nullptr, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        return arena_texture.readback_float(context.Get(), mout, out, m * n);
    }

    D3D_DRIVER_TYPE driver_type;
    UINT device_flags;
    D3D_FEATURE_LEVEL feature_level{};

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Buffer> gemm_constant_buffer;
    ComPtr<ID3D11Buffer> transpose_constant_buffer;
    ComPtr<ID3D11Buffer> add_vector_constant_buffer;
    ComPtr<ID3D11Buffer> fused_gemm_constant_buffer;
    D3D11ShaderManager shader_manager;
    ArenaTexture arena_texture;
};

D3D11Backend::D3D11Backend(D3D_DRIVER_TYPE driver_type, UINT device_flags)
    : impl_(std::make_unique<Impl>(driver_type, device_flags)) {
}

D3D11Backend::~D3D11Backend() = default;
D3D11Backend::D3D11Backend(D3D11Backend&&) noexcept = default;
D3D11Backend& D3D11Backend::operator=(D3D11Backend&&) noexcept = default;

HRESULT D3D11Backend::initialize() {
    return impl_->step_one_creating_the_device();
}

HRESULT D3D11Backend::sum(DXGI_FORMAT format, const void* a, const void* b, void* out,
                          std::size_t count) {
    return impl_->sum(format, a, b, out, count);
}

HRESULT D3D11Backend::mul(DXGI_FORMAT format, const void* a, const void* b, void* out,
                          std::size_t count) {
    return impl_->mul(format, a, b, out, count);
}

HRESULT D3D11Backend::gemm(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                           std::size_t k) {
    return impl_->gemm(a, b, c, m, n, k);
}

HRESULT D3D11Backend::transpose(const float* a, float* out, std::size_t rows, std::size_t cols) {
    return impl_->transpose(a, out, rows, cols);
}

HRESULT D3D11Backend::add_vector(const float* matrix, const float* vector, float* out,
                                 std::size_t rows, std::size_t cols) {
    return impl_->add_vector(matrix, vector, out, rows, cols);
}

HRESULT D3D11Backend::gemm_transpose_bias(const float* a, const float* b, const float* c,
                                          float* out, std::size_t m, std::size_t n, std::size_t k) {
    return impl_->gemm_transpose_bias(a, b, c, out, m, n, k);
}

HRESULT D3D11Backend::copy_gpu_mem(gpuMemBase& a, std::size_t count) {
    return impl_->copy_gpu_mem(a, count);
}

HRESULT D3D11Backend::copy_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out,
                                   std::size_t count) {
    return impl_->copy_gpu_mem(a, b, out, count);
}

HRESULT D3D11Backend::readback_gpu_mem(gpuMemBase& source, void* out, std::size_t count) {
    return impl_->readback_gpu_mem(source, out, count);
}

HRESULT D3D11Backend::readback_gpu_mem_batch(const std::vector<GpuMemReadbackRequest>& requests) {
    return impl_->readback_gpu_mem_batch(requests);
}

HRESULT D3D11Backend::sum_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out,
                                  std::size_t count) {
    return impl_->sum_gpu_mem(a, b, out, count);
}
HRESULT D3D11Backend::mul_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out,
                                  std::size_t count) {
    return impl_->mul_gpu_mem(a, b, out, count);
}
HRESULT D3D11Backend::gemm_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& out, std::size_t m,
                                   std::size_t n, std::size_t k) {
    return impl_->gemm_gpu_mem(a, b, out, m, n, k);
}
HRESULT D3D11Backend::transpose_gpu_mem(gpuMemBase& a, gpuMemBase& out, std::size_t rows,
                                        std::size_t cols) {
    return impl_->transpose_gpu_mem(a, out, rows, cols);
}
HRESULT D3D11Backend::add_vector_gpu_mem(gpuMemBase& matrix, gpuMemBase& vector, gpuMemBase& out,
                                         std::size_t rows, std::size_t cols) {
    return impl_->add_vector_gpu_mem(matrix, vector, out, rows, cols);
}
HRESULT D3D11Backend::gemm_transpose_bias_gpu_mem(gpuMemBase& a, gpuMemBase& b, gpuMemBase& c,
                                                  gpuMemBase& out, std::size_t m, std::size_t n,
                                                  std::size_t k) {
    return impl_->gemm_transpose_bias_gpu_mem(a, b, c, out, m, n, k);
}

HRESULT D3D11Backend::upload_gpu(DXGI_FORMAT format, const void* data, std::size_t count,
                                 std::shared_ptr<void>& out) {
    return impl_->upload_gpu(format, data, count, out);
}
HRESULT D3D11Backend::sum_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b,
                              std::size_t count, std::shared_ptr<void>& out) {
    return impl_->binary_gpu(BinaryOp::Add, a, b, count, out);
}
HRESULT D3D11Backend::mul_gpu(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b,
                              std::size_t count, std::shared_ptr<void>& out) {
    return impl_->binary_gpu(BinaryOp::Mul, a, b, count, out);
}
HRESULT D3D11Backend::readback_gpu(const std::shared_ptr<void>& array, void* out,
                                   std::size_t count) {
    return impl_->readback_gpu(array, out, count);
}

HRESULT D3D11Backend::get_native(ID3D11Device** device, ID3D11DeviceContext** context) const {
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
