#include "arena_texture.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace dxblas::detail {

namespace {

HRESULT create_texture(
    ID3D11Device* device,
    UINT width,
    UINT height,
    DXGI_FORMAT format,
    UINT bind_flags,
    D3D11_USAGE usage,
    UINT cpu_access_flags,
    Microsoft::WRL::ComPtr<ID3D11Texture2D>& texture)
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

    return device->CreateTexture2D(&desc, nullptr, &texture);
}

HRESULT create_srv(
    ID3D11Device* device,
    ID3D11Texture2D* texture,
    DXGI_FORMAT format,
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& srv)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MostDetailedMip = 0;
    desc.Texture2D.MipLevels = 1;
    return device->CreateShaderResourceView(texture, &desc, &srv);
}

HRESULT create_rtv(
    ID3D11Device* device,
    ID3D11Texture2D* texture,
    DXGI_FORMAT format,
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView>& rtv)
{
    D3D11_RENDER_TARGET_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipSlice = 0;
    return device->CreateRenderTargetView(texture, &desc, &rtv);
}

// Upload a tightly packed CPU array without building a padded temporary buffer.
// UpdateSubresource reads only the source region described by each box, so the
// final partial row never reads beyond the caller's allocation.
HRESULT layout_1d_with_max_width(
    std::size_t count, UINT max_width, Layout1D& out)
{
    if (count == 0 || max_width == 0) return E_INVALIDARG;

    const UINT max_dim = static_cast<UINT>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION);
    max_width = std::min(max_width, max_dim);
    const UINT width = static_cast<UINT>(std::min<std::size_t>(count, max_width));
    if (width == 0) return E_INVALIDARG;

    const std::size_t height = count / width + (count % width != 0 ? 1u : 0u);
    if (height == 0 || height > max_dim) return E_INVALIDARG;

    out.width = width;
    out.height = static_cast<UINT>(height);
    return S_OK;
}

void upload_contiguous_rows(
    ID3D11DeviceContext* context,
    ID3D11Texture2D* texture,
    const void* data,
    std::size_t count,
    UINT width,
    UINT height,
    UINT bytes_per_element)
{
    const std::size_t full_rows = count / width;
    const std::size_t tail_elements = count % width;
    const UINT row_pitch = width * bytes_per_element;
    const auto* source = static_cast<const std::uint8_t*>(data);

    if (full_rows != 0) {
        const D3D11_BOX full_rows_box{
            0, 0, 0, width, static_cast<UINT>(full_rows), 1};
        context->UpdateSubresource(
            texture, 0, &full_rows_box, source, row_pitch, 0);
    }

    if (tail_elements != 0) {
        const UINT tail_bytes = static_cast<UINT>(tail_elements) * bytes_per_element;
        const UINT tail_y = static_cast<UINT>(full_rows);
        const D3D11_BOX tail_row_box{
            0, tail_y, 0, static_cast<UINT>(tail_elements), tail_y + 1, 1};
        context->UpdateSubresource(
            texture, 0, &tail_row_box,
            source + full_rows * static_cast<std::size_t>(row_pitch),
            tail_bytes, 0);
    }
}

// Copy a mapped, row-pitched texture region into a tightly packed CPU array.
// The caller must Unmap the resource after this helper returns, on both success
// and failure paths.
HRESULT copy_mapped_to_cpu(
    const D3D11_MAPPED_SUBRESOURCE& mapped,
    void* destination,
    std::size_t count,
    UINT row_elements,
    UINT rows,
    UINT bytes)
{
    if (!mapped.pData || !destination || count == 0 || row_elements == 0 ||
        rows == 0 || bytes == 0 ||
        count > static_cast<std::size_t>(row_elements) * rows)
        return E_INVALIDARG;

    const std::size_t total_bytes = count * bytes;
    auto* dst = static_cast<std::uint8_t*>(destination);
    const auto* src = static_cast<const std::uint8_t*>(mapped.pData);

    if (rows == 1 || mapped.RowPitch == static_cast<std::size_t>(row_elements) * bytes) {
        std::memcpy(dst, src, total_bytes);
        return S_OK;
    }

    std::size_t remaining = count;
    for (UINT row = 0; row < rows && remaining != 0; ++row) {
        const std::size_t row_count = std::min<std::size_t>(remaining, row_elements);
        std::memcpy(dst, src + static_cast<std::size_t>(row) * mapped.RowPitch,
                    row_count * bytes);
        dst += row_count * bytes;
        remaining -= row_count;
    }

    return remaining == 0 ? S_OK : E_FAIL;
}

} // namespace

UINT bytes_per_element(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT:
        return 1;
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R16_FLOAT:
        return 2;
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R32_FLOAT:
        return 4;
    default:
        return 0;
    }
}

DXGI_FORMAT staging_format_for(DXGI_FORMAT format)
{
    switch (bytes_per_element(format)) {
    case 1: return DXGI_FORMAT_R8_TYPELESS;
    case 2: return DXGI_FORMAT_R16_TYPELESS;
    case 4: return DXGI_FORMAT_R32_TYPELESS;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

HRESULT layout_1d(std::size_t count, Layout1D& out)
{
    return layout_1d_with_max_width(
        count, static_cast<UINT>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION), out);
}

HRESULT readback_staging_dimensions(
    UINT current_width, UINT current_height, UINT request_width, UINT request_height,
    UINT& allocation_width, UINT& allocation_height)
{
    if (request_width == 0 || request_height == 0 ||
        request_width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        request_height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        current_width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        current_height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return E_INVALIDARG;

    const auto area = [](UINT width, UINT height, std::size_t& result) -> bool {
        if (width == 0 || height == 0) {
            result = 0;
            return true;
        }
        if (static_cast<std::size_t>(height) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(width))
            return false;
        result = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        return true;
    };

    std::size_t current_area = 0;
    std::size_t request_area = 0;
    if (!area(current_width, current_height, current_area) ||
        !area(request_width, request_height, request_area))
        return E_INVALIDARG;

    const UINT combined_width = std::max(current_width, request_width);
    const UINT combined_height = std::max(current_height, request_height);
    std::size_t combined_area = 0;
    if (!area(combined_width, combined_height, combined_area))
        return E_INVALIDARG;

    const std::size_t larger_area = std::max(current_area, request_area);
    const std::size_t max_area = std::numeric_limits<std::size_t>::max();
    const std::size_t area_limit = larger_area > max_area / 2
        ? max_area : larger_area * 2;

    // Keep independent maxima only when their combined rectangle is at most
    // about 2x the larger existing/requested area. Otherwise size for this
    // request alone; future requests can grow the staging texture again.
    if (combined_area > area_limit) {
        allocation_width = request_width;
        allocation_height = request_height;
    } else {
        allocation_width = combined_width;
        allocation_height = combined_height;
    }
    return S_OK;
}

HRESULT GpuTimer::initialize(ID3D11Device* device, ID3D11DeviceContext* context)
{
    if (!device || !context) return E_INVALIDARG;

    D3D11_QUERY_DESC desc{};
    desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
    HRESULT hr = device->CreateQuery(&desc, &disjoint);
    if (FAILED(hr)) return hr;

    desc.Query = D3D11_QUERY_TIMESTAMP;
    Microsoft::WRL::ComPtr<ID3D11Query> start;
    hr = device->CreateQuery(&desc, &start);
    if (FAILED(hr)) return hr;
    Microsoft::WRL::ComPtr<ID3D11Query> end;
    hr = device->CreateQuery(&desc, &end);
    if (FAILED(hr)) return hr;

    context_ = context;
    disjoint_query_ = std::move(disjoint);
    start_query_ = std::move(start);
    end_query_ = std::move(end);
    active_ = false;
    pending_ = false;
    return S_OK;
}

HRESULT GpuTimer::begin()
{
    if (!context_ || !disjoint_query_ || !start_query_ || !end_query_)
        return E_UNEXPECTED;
    if (active_ || pending_) return E_UNEXPECTED;

    context_->Begin(disjoint_query_.Get());
    context_->End(start_query_.Get());
    active_ = true;
    return S_OK;
}

HRESULT GpuTimer::end()
{
    if (!context_ || !active_) return E_UNEXPECTED;

    context_->End(end_query_.Get());
    context_->End(disjoint_query_.Get());
    active_ = false;
    pending_ = true;
    return S_OK;
}

double GpuTimer::elapsed_ms()
{
    if (!context_ || !pending_) return std::numeric_limits<double>::quiet_NaN();

    const auto wait_for_query = [this](ID3D11Query* query, void* data, UINT size) -> HRESULT {
        HRESULT hr = S_FALSE;
        while (hr == S_FALSE) {
            hr = context_->GetData(query, data, size, 0);
            if (hr == S_FALSE) std::this_thread::yield();
        }
        return hr;
    };

    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint{};
    UINT64 start = 0;
    UINT64 end = 0;
    HRESULT hr = wait_for_query(disjoint_query_.Get(), &disjoint, sizeof(disjoint));
    if (SUCCEEDED(hr)) hr = wait_for_query(start_query_.Get(), &start, sizeof(start));
    if (SUCCEEDED(hr)) hr = wait_for_query(end_query_.Get(), &end, sizeof(end));

    pending_ = false;
    if (FAILED(hr) || disjoint.Disjoint || disjoint.Frequency == 0 || end < start)
        return std::numeric_limits<double>::quiet_NaN();

    return static_cast<double>(end - start) * 1000.0 /
        static_cast<double>(disjoint.Frequency);
}

HRESULT ArenaTexture::initialize(ID3D11Device* device)
{
    if (!device) return E_INVALIDARG;
    device_ = device;
    return S_OK;
}

HRESULT ArenaTexture::initialize_adaptive(ID3D11Device* device)
{
    return initialize(device);
}

HRESULT ArenaTexture::ensure(
    DXGI_FORMAT input_format,
    DXGI_FORMAT output_format,
    UINT width,
    UINT height)
{
    if (!device_ || width == 0 || height == 0 ||
        width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return E_INVALIDARG;

    const bool format_matches =
        input_format_ == input_format &&
        output_format_ == output_format;

    const bool capacity_matches =
        capacity_width_ >= width &&
        capacity_height_ >= height;

    if (format_matches && capacity_matches)
        return S_OK;

    const auto round_up_power_of_two = [](UINT value, UINT& rounded) -> HRESULT {
        const UINT max_dimension = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        if (value == 0 || value > max_dimension)
            return E_INVALIDARG;

        UINT power = 1;
        while (power < value && power < max_dimension)
            power <<= 1;
        if (power < value)
            return E_INVALIDARG;

        rounded = power;
        return S_OK;
    };

    UINT new_width = 0;
    UINT new_height = 0;
    HRESULT hr = round_up_power_of_two(width, new_width);
    if (FAILED(hr)) return hr;
    hr = round_up_power_of_two(height, new_height);
    if (FAILED(hr)) return hr;

    // allocate() creates all replacement textures and views before swapping
    // them into the arena, so a failed growth leaves the old resources intact.
    return allocate(input_format, output_format, new_width, new_height);
}

HRESULT ArenaTexture::allocate(
    DXGI_FORMAT input_format,
    DXGI_FORMAT output_format,
    UINT width,
    UINT height)
{
    Microsoft::WRL::ComPtr<ID3D11Texture2D> input_a;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> input_b;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> output;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv_a;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv_b;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv;

    HRESULT hr = create_texture(
        device_.Get(), width, height, input_format,
        D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT, 0, input_a);
    if (FAILED(hr)) return hr;

    hr = create_texture(
        device_.Get(), width, height, input_format,
        D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT, 0, input_b);
    if (FAILED(hr)) return hr;

    hr = create_texture(
        device_.Get(), width, height, output_format,
        D3D11_BIND_RENDER_TARGET, D3D11_USAGE_DEFAULT, 0, output);
    if (FAILED(hr)) return hr;

    hr = create_srv(device_.Get(), input_a.Get(), input_format, srv_a);
    if (FAILED(hr)) return hr;

    hr = create_srv(device_.Get(), input_b.Get(), input_format, srv_b);
    if (FAILED(hr)) return hr;

    hr = create_rtv(device_.Get(), output.Get(), output_format, rtv);
    if (FAILED(hr)) return hr;

    input_a_ = std::move(input_a);
    input_b_ = std::move(input_b);
    output_ = std::move(output);
    input_a_srv_ = std::move(srv_a);
    input_b_srv_ = std::move(srv_b);
    output_rtv_ = std::move(rtv);

    input_format_ = input_format;
    output_format_ = output_format;
    capacity_width_ = width;
    capacity_height_ = height;

    return S_OK;
}

void ArenaTexture::upload(
    ID3D11DeviceContext* context,
    const void* a,
    const void* b,
    UINT row_pitch,
    UINT width,
    UINT height)
{
    const D3D11_BOX box{
        0, 0, 0,
        width, height, 1
    };

    context->UpdateSubresource(
        input_a_.Get(), 0, &box, a, row_pitch, 0);
    context->UpdateSubresource(
        input_b_.Get(), 0, &box, b, row_pitch, 0);
}

HRESULT ArenaTexture::readback(
    ID3D11DeviceContext* context,
    void* destination,
    UINT width,
    UINT height,
    UINT elements,
    UINT bytes_per_element)
{
    if (!context || !destination || !output_ ||
        width == 0 || height == 0 || elements == 0 ||
        width > capacity_width_ || height > capacity_height_ ||
        elements > static_cast<std::size_t>(width) * height)
        return E_INVALIDARG;

    HRESULT hr = ensure_readback_staging(output_format_, width, height);
    if (FAILED(hr)) return hr;

    // Copy only the rendered rectangle into the shared staging resource.
    const D3D11_BOX used_region{0, 0, 0, width, height, 1};
    context->CopySubresourceRegion(
        shared_staging_.Get(), 0, 0, 0, 0, output_.Get(), 0, &used_region);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context->Map(
        shared_staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr))
        return hr;

    const HRESULT copy_hr = copy_mapped_to_cpu(
        mapped, destination, elements, width, height, bytes_per_element);
    context->Unmap(shared_staging_.Get(), 0);
    return copy_hr;
}


HRESULT ArenaTexture::acquire_float(std::size_t count, std::shared_ptr<gpuMemBase>& out)
{
    if (!device_ || count == 0 ||
        count > static_cast<std::size_t>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) *
                static_cast<std::size_t>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION))
        return E_INVALIDARG;

    Layout1D layout{};
    HRESULT hr = layout_1d_with_max_width(count, 4096u, layout);
    if (FAILED(hr)) return hr;
    hr = acquire_matrix_float(layout.width, layout.height, out);
    if (SUCCEEDED(hr)) out->count = count;
    return hr;
}

HRESULT ArenaTexture::acquire_matrix_float(
    UINT width, UINT height, std::shared_ptr<gpuMemBase>& out)
{
    if (!device_ || width == 0 || height == 0 ||
        width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return E_INVALIDARG;

    const std::size_t count = static_cast<std::size_t>(width) * height;
    for (auto& slot : gpu_slots_) {
        if (slot.use_count() == 1 && slot->width == width && slot->height == height) {
            slot->count = count;
            out = slot;
            return S_OK;
        }
    }

    auto slot = std::make_shared<gpuMemBase>();
    slot->width = width;
    slot->height = height;
    slot->count = count;
    slot->storage_format = DXGI_FORMAT_R32_FLOAT;
    HRESULT hr = create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
        D3D11_USAGE_DEFAULT, 0, slot->texture);
    if (FAILED(hr)) return hr;
    hr = create_srv(device_.Get(), slot->texture.Get(), DXGI_FORMAT_R32_FLOAT, slot->srv);
    if (FAILED(hr)) return hr;
    hr = create_rtv(device_.Get(), slot->texture.Get(), DXGI_FORMAT_R32_FLOAT, slot->rtv);
    if (FAILED(hr)) return hr;
    gpu_slots_.push_back(slot);
    out = std::move(slot);
    return S_OK;
}

HRESULT ArenaTexture::upload_matrix_float(
    ID3D11DeviceContext* context, const float* data, UINT width, UINT height,
    std::shared_ptr<gpuMemBase>& out)
{
    if (!context || !data || width == 0 || height == 0) return E_INVALIDARG;
    HRESULT hr = acquire_matrix_float(width, height, out);
    if (FAILED(hr)) return hr;
    context->UpdateSubresource(out->texture.Get(), 0, nullptr, data,
        width * static_cast<UINT>(sizeof(float)), 0);
    return S_OK;
}

HRESULT ArenaTexture::upload_float(
    ID3D11DeviceContext* context, const void* data, std::size_t count,
    std::shared_ptr<gpuMemBase>& out)
{
    if (!context || !data || count == 0) return E_INVALIDARG;
    HRESULT hr = acquire_float(count, out);
    if (FAILED(hr)) return hr;

    upload_contiguous_rows(context, out->texture.Get(), data, count,
        out->width, out->height, static_cast<UINT>(sizeof(float)));
    return S_OK;
}

HRESULT ArenaTexture::readback_float(
    ID3D11DeviceContext* context, const std::shared_ptr<gpuMemBase>& source,
    void* destination, std::size_t count)
{
    if (!context || !source || !destination || count != source->count ||
        !source->texture || count == 0 || source->width == 0)
        return E_INVALIDARG;

    const UINT rows = static_cast<UINT>((count + source->width - 1u) / source->width);
    const UINT copied_width = static_cast<UINT>(std::min<std::size_t>(count, source->width));
    assert(source->storage_format != DXGI_FORMAT_UNKNOWN);
    HRESULT hr = ensure_readback_staging(source->storage_format, copied_width, rows);
    if (FAILED(hr)) return hr;

    const D3D11_BOX used_region{0, 0, 0, copied_width, rows, 1};
    context->CopySubresourceRegion(
        shared_staging_.Get(), 0, 0, 0, 0, source->texture.Get(), 0, &used_region);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context->Map(shared_staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return hr;

    const HRESULT copy_hr = copy_mapped_to_cpu(
        mapped, destination, count, copied_width, rows, static_cast<UINT>(sizeof(float)));
    context->Unmap(shared_staging_.Get(), 0);
    return copy_hr;
}


HRESULT ArenaTexture::prepare_gpu_mem(gpuMemBase& memory, DXGI_FORMAT format, std::size_t count)
{
    if (!device_ || count == 0) return E_INVALIDARG;
    const UINT bytes = bytes_per_element(format);
    if (!bytes) return E_INVALIDARG;
    const UINT max_dim = static_cast<UINT>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION);

    Layout1D layout{};
    const HRESULT layout_hr = layout_1d(count, layout);
    if (FAILED(layout_hr)) return layout_hr;

    UINT width = layout.width;
    UINT height = layout.height;
    const bool has_explicit_dimensions = memory.width != 0 && memory.height != 0 &&
        memory.width <= max_dim && memory.height <= max_dim &&
        static_cast<std::size_t>(memory.width) * memory.height >= count;
    // Preserve shapes marked by matrix/GEMM helpers. Also honor dimensions set
    // by callers before the first upload (the gpuMem GEMM sample uses this
    // compatibility path); once a resource exists, stale dimensions alone do
    // not override the canonical 1D layout after a reassignment.
    const bool valid_explicit_shape = memory.storage_format == format &&
        has_explicit_dimensions &&
        (memory.shape_is_user_defined || !memory.texture.Get());

    if (valid_explicit_shape) {
        width = memory.width;
        height = memory.height;
    }

    // Ordinary 1D buffers may reuse a taller allocation, but their row width
    // must match the canonical layout. This keeps pixel coordinates compatible
    // with freshly allocated buffers after a shrink/reassignment.
    if (memory.storage_format == format &&
        memory.width == width && memory.height >= height &&
        memory.texture && memory.srv && memory.rtv) {
        memory.count = count;
        memory.shape_is_user_defined = valid_explicit_shape;
        return S_OK;
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv;
    HRESULT hr = create_texture(device_.Get(), width, height, format,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, D3D11_USAGE_DEFAULT, 0, texture);
    if (FAILED(hr)) return hr;
    hr = create_srv(device_.Get(), texture.Get(), format, srv);
    if (FAILED(hr)) return hr;
    hr = create_rtv(device_.Get(), texture.Get(), format, rtv);
    if (FAILED(hr)) return hr;

    memory.texture = std::move(texture);
    memory.srv = std::move(srv);
    memory.rtv = std::move(rtv);
    memory.width = width;
    memory.height = height;
    memory.count = count;
    memory.storage_format = format;
    memory.shape_is_user_defined = valid_explicit_shape;
    return S_OK;
}
HRESULT ArenaTexture::prepare_gpu_mem_matrix(gpuMemBase& memory, DXGI_FORMAT format, UINT width, UINT height)
{
    if (!device_ || format == DXGI_FORMAT_UNKNOWN || width == 0 || height == 0 ||
        width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION || height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return E_INVALIDARG;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (memory.storage_format == format && memory.width == width && memory.height == height &&
        memory.texture && memory.srv && memory.rtv) {
        memory.count = count;
        memory.shape_is_user_defined = true;
        return S_OK;
    }
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv;
    HRESULT hr = create_texture(device_.Get(), width, height, format,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, D3D11_USAGE_DEFAULT, 0, texture);
    if (FAILED(hr)) return hr;
    hr = create_srv(device_.Get(), texture.Get(), format, srv);
    if (FAILED(hr)) return hr;
    hr = create_rtv(device_.Get(), texture.Get(), format, rtv);
    if (FAILED(hr)) return hr;
    memory.texture = std::move(texture);
    memory.srv = std::move(srv);
    memory.rtv = std::move(rtv);
    memory.width = width;
    memory.height = height;
    memory.count = count;
    memory.storage_format = format;
    memory.shape_is_user_defined = true;
    return S_OK;
}

HRESULT ArenaTexture::upload_gpu_mem(ID3D11DeviceContext* context, gpuMemBase& memory,
                                     const void* data, std::size_t count)
{
    if (!context || !data || count == 0) return E_INVALIDARG;
    HRESULT hr = prepare_gpu_mem(memory, memory.storage_format, count);
    if (FAILED(hr)) return hr;
    const UINT bytes = bytes_per_element(memory.storage_format);
    upload_contiguous_rows(context, memory.texture.Get(), data, count,
        memory.width, memory.height, bytes);
    return S_OK;
}

HRESULT ArenaTexture::ensure_readback_staging(DXGI_FORMAT format, UINT width, UINT height)
{
    if (!device_ || width == 0 || height == 0)
        return E_INVALIDARG;

    const DXGI_FORMAT staging_format = staging_format_for(format);
    if (staging_format == DXGI_FORMAT_UNKNOWN)
        return E_INVALIDARG;

    const bool reusable = shared_staging_ && shared_staging_format_ == staging_format &&
        shared_staging_width_ >= width && shared_staging_height_ >= height;
    if (reusable) return S_OK;

    // Typed formats with the same element size share a typeless staging resource.
    // Retain independent dimension maxima only when their area stays within the
    // bounded-growth policy; otherwise allocate for this request alone.
    UINT allocation_width = width;
    UINT allocation_height = height;
    if (shared_staging_format_ == staging_format) {
        const HRESULT dimensions_hr = readback_staging_dimensions(
            shared_staging_width_, shared_staging_height_, width, height,
            allocation_width, allocation_height);
        if (FAILED(dimensions_hr)) return dimensions_hr;
    }

    // Build the replacement separately. If allocation fails, the previous
    // staging texture and its recorded dimensions remain valid and untouched.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    HRESULT hr = create_texture(device_.Get(), allocation_width, allocation_height,
        staging_format, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ, staging);
    if (FAILED(hr)) return hr;

    shared_staging_ = std::move(staging);
    shared_staging_format_ = staging_format;
    shared_staging_width_ = allocation_width;
    shared_staging_height_ = allocation_height;
    return S_OK;
}

HRESULT ArenaTexture::readback_gpu_mem(ID3D11DeviceContext* context, gpuMemBase& memory,
                                       void* out, std::size_t count)
{
    if (!context || !memory.texture.Get() || !out || count == 0 ||
        count > memory.count || memory.width == 0 || memory.height == 0)
        return E_INVALIDARG;
    assert(memory.storage_format != DXGI_FORMAT_UNKNOWN);

    const UINT rows = static_cast<UINT>((count + memory.width - 1u) / memory.width);
    const UINT copied_width = static_cast<UINT>(std::min<std::size_t>(count, memory.width));
    HRESULT hr = ensure_readback_staging(memory.storage_format, copied_width, rows);
    if (FAILED(hr)) return hr;

    const D3D11_BOX used_region{0, 0, 0, copied_width, rows, 1};
    context->CopySubresourceRegion(
        shared_staging_.Get(), 0, 0, 0, 0, memory.texture.Get(), 0, &used_region);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context->Map(shared_staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return hr;
    const UINT bytes = bytes_per_element(memory.storage_format);
    const HRESULT copy_hr = bytes
        ? copy_mapped_to_cpu(mapped, out, count, copied_width, rows, bytes)
        : E_INVALIDARG;
    context->Unmap(shared_staging_.Get(), 0);
    return copy_hr;
}
HRESULT ArenaTexture::readback_gpu_mem_batch(
    ID3D11DeviceContext* context,
    const std::vector<GpuMemReadbackRequest>& requests)
{
    if (!context) return E_POINTER;
    if (requests.empty()) return S_OK;

    struct PendingReadback {
        const GpuMemReadbackRequest* request = nullptr;
        UINT copied_width = 0;
        UINT rows = 0;
        UINT bytes_per_element = 0;
    };
    struct ReadbackGroup {
        DXGI_FORMAT staging_format = DXGI_FORMAT_UNKNOWN;
        // Keep a typed representative format for ensure_readback_staging;
        // staging_format_for() maps it to this group's typeless format.
        DXGI_FORMAT representative_format = DXGI_FORMAT_UNKNOWN;
        std::vector<PendingReadback> items;
    };

    // Validate and prepare every request before issuing any GPU copy. This also
    // ensures a bad later request cannot leave earlier copies queued.
    std::vector<ReadbackGroup> groups;
    groups.reserve(requests.size());
    const UINT max_dimension = static_cast<UINT>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION);

    for (const auto& request : requests) {
        if (request.count == 0) continue;
        if (!request.source || !request.destination || !request.source->texture.Get() ||
            request.count > request.source->count || request.source->width == 0 ||
            request.source->height == 0)
            return E_INVALIDARG;

        const UINT bytes = bytes_per_element(request.source->storage_format);
        const DXGI_FORMAT staging_format = staging_format_for(request.source->storage_format);
        if (bytes == 0 || staging_format == DXGI_FORMAT_UNKNOWN)
            return E_INVALIDARG;

        const std::size_t source_width = request.source->width;
        const std::size_t row_count = request.count / source_width +
            (request.count % source_width != 0 ? 1u : 0u);
        if (row_count == 0 || row_count > request.source->height || row_count > max_dimension)
            return E_INVALIDARG;

        PendingReadback item;
        item.request = &request;
        item.copied_width = static_cast<UINT>(std::min<std::size_t>(request.count, source_width));
        item.rows = static_cast<UINT>(row_count);
        item.bytes_per_element = bytes;

        auto group_it = std::find_if(groups.begin(), groups.end(),
            [staging_format](const ReadbackGroup& group) {
                return group.staging_format == staging_format;
            });
        if (group_it == groups.end()) {
            ReadbackGroup group;
            group.staging_format = staging_format;
            group.representative_format = request.source->storage_format;
            group.items.push_back(item);
            groups.push_back(std::move(group));
        } else {
            group_it->items.push_back(item);
        }
    }

    if (groups.empty()) return S_OK;

    // A pass packs requests vertically into one shared staging texture. Split
    // a group when the sum of its row counts would exceed the D3D11 dimension.
    for (const auto& group : groups) {
        std::size_t pass_begin = 0;
        while (pass_begin < group.items.size()) {
            std::size_t pass_end = pass_begin;
            UINT staging_width = 0;
            UINT staging_height = 0;
            while (pass_end < group.items.size()) {
                const PendingReadback& item = group.items[pass_end];
                if (item.rows > max_dimension) return E_INVALIDARG;
                if (staging_height != 0 && item.rows > max_dimension - staging_height)
                    break;
                staging_width = std::max(staging_width, item.copied_width);
                staging_height += item.rows;
                ++pass_end;
            }
            if (pass_end == pass_begin || staging_width == 0 || staging_height == 0)
                return E_INVALIDARG;

            HRESULT hr = ensure_readback_staging(
                group.representative_format, staging_width, staging_height);
            if (FAILED(hr)) return hr;

            UINT dst_y = 0;
            for (std::size_t i = pass_begin; i < pass_end; ++i) {
                const PendingReadback& item = group.items[i];
                const D3D11_BOX used_region{
                    0, 0, 0, item.copied_width, item.rows, 1};
                context->CopySubresourceRegion(
                    shared_staging_.Get(), 0, 0, dst_y, 0,
                    item.request->source->texture.Get(), 0, &used_region);
                dst_y += item.rows;
            }

            D3D11_MAPPED_SUBRESOURCE mapped{};
            hr = context->Map(shared_staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
            if (FAILED(hr)) return hr;

            HRESULT copy_result = S_OK;
            UINT src_y = 0;
            for (std::size_t i = pass_begin; i < pass_end; ++i) {
                const PendingReadback& item = group.items[i];
                D3D11_MAPPED_SUBRESOURCE request_mapped = mapped;
                request_mapped.pData = static_cast<std::uint8_t*>(mapped.pData) +
                    static_cast<std::size_t>(src_y) * mapped.RowPitch;
                const HRESULT item_result = copy_mapped_to_cpu(
                    request_mapped, item.request->destination, item.request->count,
                    item.copied_width, item.rows, item.bytes_per_element);
                if (FAILED(item_result)) {
                    copy_result = item_result;
                    break;
                }
                src_y += item.rows;
            }
            context->Unmap(shared_staging_.Get(), 0);
            if (FAILED(copy_result)) return copy_result;

            pass_begin = pass_end;
        }
    }

    return S_OK;
}
HRESULT ArenaTexture::copy_gpu_mem_buffers(ID3D11DeviceContext* context, gpuMemBase& a,
                                             gpuMemBase& b, gpuMemBase& out, std::size_t count)
{
    if (!context || !a.cpu_data() || !b.cpu_data() || count == 0 || a.storage_format != b.storage_format ||
        a.storage_format != out.storage_format) return E_INVALIDARG;
    HRESULT hr = upload_gpu_mem(context, a, a.cpu_data(), count);
    if (FAILED(hr)) return hr;
    hr = upload_gpu_mem(context, b, b.cpu_data(), count);
    if (FAILED(hr)) return hr;
    return prepare_gpu_mem(out, out.storage_format, count);
}

HRESULT ArenaTexture::create_gpu_buffer(
    ID3D11DeviceContext* context, DXGI_FORMAT format, const void* data,
    std::size_t count, std::shared_ptr<gpuMemBase>& out)
{
    if (!context || count == 0) return E_INVALIDARG;
    const UINT bytes = bytes_per_element(format);
    if (bytes == 0) return E_INVALIDARG;
    Layout1D layout{};
    HRESULT hr = layout_1d(count, layout);
    if (FAILED(hr)) return hr;
    const UINT width = layout.width;
    const UINT height = layout.height;
    auto texture = std::make_shared<gpuMemBase>();
    hr = create_texture(device_.Get(), width, height, format,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, D3D11_USAGE_DEFAULT, 0, texture->texture);
    if (FAILED(hr)) return hr;
    hr = create_srv(device_.Get(), texture->texture.Get(), format, texture->srv);
    if (FAILED(hr)) return hr;
    hr = create_rtv(device_.Get(), texture->texture.Get(), format, texture->rtv);
    if (FAILED(hr)) return hr;
    texture->width = width; texture->height = height; texture->count = count;
    texture->storage_format = format;
    if (data) {
        upload_contiguous_rows(context, texture->texture.Get(), data, count,
            width, height, bytes);
    }
    out = std::move(texture);
    return S_OK;
}

HRESULT ArenaTexture::readback_gpu_buffer(
    ID3D11DeviceContext* context, const std::shared_ptr<void>& opaque,
    void* out, std::size_t count)
{
    if (!context || !opaque || !out || count == 0) return E_INVALIDARG;
    auto texture = std::static_pointer_cast<gpuMemBase>(opaque);
    if (count > texture->count) return E_INVALIDARG;
    return readback_gpu_mem(context, *texture, out, count);
}

std::shared_ptr<gpuMemBase> ArenaTexture::texture_from_handle(
    const std::shared_ptr<void>& handle) const noexcept
{
    if (!handle) return {};
    return std::static_pointer_cast<gpuMemBase>(handle);
}

} // namespace dxblas::detail
