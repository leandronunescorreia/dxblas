#include "arena_texture.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <dxgi1_4.h>

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

} // namespace

HRESULT ArenaTexture::initialize(
    ID3D11Device* device,
    UINT initial_width,
    UINT initial_height)
{
    if (!device || initial_width == 0 || initial_height == 0)
        return E_INVALIDARG;

    // Initialization records device capabilities only. GPU resources are created
    // on demand by ensure() for the first actual operation.
    device_ = device;
    switch (device->GetFeatureLevel()) {
    case D3D_FEATURE_LEVEL_11_1:
    case D3D_FEATURE_LEVEL_11_0:
        max_dimension_ = 16384;
        break;
    case D3D_FEATURE_LEVEL_10_1:
    case D3D_FEATURE_LEVEL_10_0:
        max_dimension_ = 8192;
        break;
    case D3D_FEATURE_LEVEL_9_3:
        max_dimension_ = 4096;
        break;
    case D3D_FEATURE_LEVEL_9_2:
    case D3D_FEATURE_LEVEL_9_1:
        max_dimension_ = 2048;
        break;
    default:
        max_dimension_ = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        break;
    }

    // Retain the parameters for API compatibility and reject unsupported sizes,
    // but do not allocate or prewarm anything here.
    if (initial_width > max_dimension_ || initial_height > max_dimension_)
        return E_INVALIDARG;
    return S_OK;
}

HRESULT ArenaTexture::initialize_adaptive(ID3D11Device* device)
{
    if (!device) return E_INVALIDARG;

    // Do not allocate based on the adapter's memory budget at startup. The
    // budget is shared with other applications and is not a reliable estimate
    // of a safe single allocation. ensure() allocates only the requested region.
    return initialize(device, 1, 1);
}

HRESULT ArenaTexture::ensure(
    DXGI_FORMAT input_format,
    DXGI_FORMAT output_format,
    UINT width,
    UINT height)
{
    if (!device_ || width == 0 || height == 0 ||
        width > max_dimension_ || height > max_dimension_)
        return E_INVALIDARG;

    const bool format_matches =
        input_format_ == input_format &&
        output_format_ == output_format;

    const bool capacity_matches =
        capacity_width_ >= width &&
        capacity_height_ >= height;

    // Reuse existing resources when both format and requested dimensions fit.
    if (format_matches && capacity_matches)
        return S_OK;

    // Allocate exactly for this request. In particular, a format change must
    // not preserve a previous 4096x4096 capacity when the new format only needs
    // a small buffer. allocate() builds a complete replacement before swapping
    // resources, so a failed allocation leaves the previous arena usable.
    return allocate(input_format, output_format, width, height);
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
    Microsoft::WRL::ComPtr<ID3D11Texture2D> readback;
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

    hr = create_texture(
        device_.Get(), width, height, output_format,
        0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ, readback);
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
    readback_ = std::move(readback);
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
    UINT bytes_per_element,
    bool packed_integer)
{
    if (!context || !destination || !readback_ || !output_ ||
        width == 0 || height == 0 || elements == 0 ||
        width > capacity_width_ || height > capacity_height_)
        return E_INVALIDARG;

    // Copy only the rendered rectangle. The staging texture may be much larger
    // than this operation (for example, a 4096x4096 arena for a tiny sum).
    const D3D11_BOX used_region{0, 0, 0, width, height, 1};
    context->CopySubresourceRegion(
        readback_.Get(), 0, 0, 0, 0, output_.Get(), 0, &used_region);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = context->Map(
        readback_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr))
        return hr;

    const UINT last_row_elements = elements - (height - 1u) * width;
    auto* dst = static_cast<std::uint8_t*>(destination);
    const auto* src = static_cast<const std::uint8_t*>(mapped.pData);

    for (UINT y = 0; y < height; ++y) {
        const UINT row_elements =
            (y + 1u == height) ? last_row_elements : width;

        auto* row_dst =
            dst + static_cast<std::size_t>(y) * width * bytes_per_element;
        const auto* row_src =
            src + static_cast<std::size_t>(y) * mapped.RowPitch;

        if (!packed_integer) {
            std::memcpy(
                row_dst,
                row_src,
                static_cast<std::size_t>(row_elements) * bytes_per_element);
            continue;
        }

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

    context->Unmap(readback_.Get(), 0);
    return S_OK;
}


HRESULT ArenaTexture::acquire_float(std::size_t count, std::shared_ptr<gpuMemBase>& out)
{
    if (!device_ || count == 0 ||
        count > static_cast<std::size_t>(max_dimension_) *
                static_cast<std::size_t>(max_dimension_))
        return E_INVALIDARG;

    const UINT width = static_cast<UINT>(std::min<std::size_t>(count, max_dimension_));
    const UINT height = static_cast<UINT>((count + width - 1u) / width);
    HRESULT hr = acquire_matrix_float(width, height, out);
    if (SUCCEEDED(hr)) out->count = count;
    return hr;
}

HRESULT ArenaTexture::acquire_matrix_float(
    UINT width, UINT height, std::shared_ptr<gpuMemBase>& out)
{
    if (!device_ || width == 0 || height == 0 ||
        width > max_dimension_ ||
        height > max_dimension_)
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
    HRESULT hr = create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
        D3D11_USAGE_DEFAULT, 0, slot->texture);
    if (FAILED(hr)) return hr;
    hr = create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT,
        0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ, slot->staging);
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

    const std::size_t capacity = static_cast<std::size_t>(out->width) * out->height;
    if (capacity == count) {
        context->UpdateSubresource(out->texture.Get(), 0, nullptr, data,
            out->width * static_cast<UINT>(sizeof(float)), 0);
    } else {
        std::vector<float> padded(capacity, 0.0f);
        std::memcpy(padded.data(), data, count * sizeof(float));
        context->UpdateSubresource(out->texture.Get(), 0, nullptr, padded.data(),
            out->width * static_cast<UINT>(sizeof(float)), 0);
    }
    return S_OK;
}

HRESULT ArenaTexture::readback_float(
    ID3D11DeviceContext* context, const std::shared_ptr<gpuMemBase>& source,
    void* destination, std::size_t count)
{
    if (!context || !source || !destination || count != source->count ||
        !source->staging || !source->texture || count == 0 || source->width == 0)
        return E_INVALIDARG;

    const UINT rows = static_cast<UINT>((count + source->width - 1u) / source->width);
    const UINT copied_width = static_cast<UINT>(std::min<std::size_t>(count, source->width));
    const D3D11_BOX used_region{0, 0, 0, copied_width, rows, 1};
    context->CopySubresourceRegion(
        source->staging.Get(), 0, 0, 0, 0, source->texture.Get(), 0, &used_region);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = context->Map(source->staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return hr;

    const auto* src = static_cast<const std::uint8_t*>(mapped.pData);
    auto* dst = static_cast<std::uint8_t*>(destination);
    std::size_t copied = 0;
    for (UINT y = 0; y < source->height && copied < count; ++y) {
        const std::size_t row_count = std::min<std::size_t>(source->width, count - copied);
        std::memcpy(dst + copied * sizeof(float),
                    src + static_cast<std::size_t>(y) * mapped.RowPitch,
                    row_count * sizeof(float));
        copied += row_count;
    }
    context->Unmap(source->staging.Get(), 0);
    return S_OK;
}



namespace {
UINT gpu_mem_bytes(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT: return 1;
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT: case DXGI_FORMAT_R16_FLOAT: return 2;
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT: case DXGI_FORMAT_R32_FLOAT: return 4;
    default: return 0;
    }
}
}

HRESULT ArenaTexture::prepare_gpu_mem(gpuMemBase& memory, DXGI_FORMAT format, std::size_t count)
{
    if (!device_ || count == 0) return E_INVALIDARG;
    const UINT bytes = gpu_mem_bytes(format);
    if (!bytes) return E_INVALIDARG;
    const UINT max_dim = max_dimension_;
    // Preserve a caller-provided matrix shape when it can contain the payload;
    // otherwise use the standard contiguous 1D-array packing.
    UINT width = memory.width;
    UINT height = memory.height;
    if (width == 0 || height == 0 || width > max_dim || height > max_dim ||
        static_cast<std::size_t>(width) * height < count) {
        width = static_cast<UINT>(std::min<std::size_t>(count, max_dim));
        const std::size_t height_size = (count + width - 1u) / width;
        if (height_size == 0 || height_size > max_dim) return E_INVALIDARG;
        height = static_cast<UINT>(height_size);
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv;
    HRESULT hr = create_texture(device_.Get(), width, height, format,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, D3D11_USAGE_DEFAULT, 0, texture);
    if (FAILED(hr)) return hr;
    hr = create_srv(device_.Get(), texture.Get(), format, srv);
    if (FAILED(hr)) return hr;
    hr = create_rtv(device_.Get(), texture.Get(), format, rtv);
    if (FAILED(hr)) return hr;
    hr = create_texture(device_.Get(), width, height, format, 0,
        D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ, staging);
    if (FAILED(hr)) return hr;

    memory.texture = std::move(texture);
    memory.staging = std::move(staging);
    memory.srv = std::move(srv);
    memory.rtv = std::move(rtv);
    memory.width = width;
    memory.height = height;
    memory.count = count;
    memory.storage_format = format;
    return S_OK;
}

HRESULT ArenaTexture::upload_gpu_mem(ID3D11DeviceContext* context, gpuMemBase& memory,
                                     const void* data, std::size_t count)
{
    if (!context || !data || count == 0) return E_INVALIDARG;
    HRESULT hr = prepare_gpu_mem(memory, memory.storage_format, count);
    if (FAILED(hr)) return hr;
    const UINT bytes = gpu_mem_bytes(memory.storage_format);
    const std::size_t capacity = static_cast<std::size_t>(memory.width) * memory.height;
    if (capacity == count) {
        // Common case: the caller's payload exactly covers the texture, so
        // UpdateSubresource can upload directly without an intermediate CPU copy.
        context->UpdateSubresource(memory.texture.Get(), 0, nullptr, data, memory.width * bytes, 0);
    } else {
        // D3D11 reads the full final row, so pad only partial textures to avoid
        // reading past the caller's allocation.
        std::vector<std::uint8_t> padded(capacity * bytes, 0);
        std::memcpy(padded.data(), data, count * bytes);
        context->UpdateSubresource(memory.texture.Get(), 0, nullptr, padded.data(), memory.width * bytes, 0);
    }
    return S_OK;
}

HRESULT ArenaTexture::readback_gpu_mem(ID3D11DeviceContext* context, gpuMemBase& memory,
                                       void* out, std::size_t count)
{
    if (!context || !memory.texture.Get() || !memory.staging.Get() || !out ||
        count == 0 || count > memory.count || memory.width == 0)
        return E_INVALIDARG;

    // Transfer only rows containing live elements; do not copy the entire arena.
    const UINT rows = static_cast<UINT>((count + memory.width - 1u) / memory.width);
    const UINT copied_width = static_cast<UINT>(std::min<std::size_t>(count, memory.width));
    const D3D11_BOX used_region{0, 0, 0, copied_width, rows, 1};
    context->CopySubresourceRegion(
        memory.staging.Get(), 0, 0, 0, 0, memory.texture.Get(), 0, &used_region);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = context->Map(memory.staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return hr;
    const UINT bytes = gpu_mem_bytes(memory.storage_format);
    if (!bytes) { context->Unmap(memory.staging.Get(), 0); return E_INVALIDARG; }
    auto* dst = static_cast<std::uint8_t*>(out);
    const auto* src = static_cast<const std::uint8_t*>(mapped.pData);
    std::size_t remaining = count;
    for (UINT row = 0; remaining && row < memory.height; ++row) {
        const std::size_t n = std::min<std::size_t>(remaining, memory.width);
        std::memcpy(dst, src + static_cast<std::size_t>(row) * mapped.RowPitch, n * bytes);
        dst += n * bytes;
        remaining -= n;
    }
    context->Unmap(memory.staging.Get(), 0);
    return remaining == 0 ? S_OK : E_FAIL;
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
    UINT bytes = 0;
    switch (format) {
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT: bytes = 1; break;
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT: case DXGI_FORMAT_R16_FLOAT: bytes = 2; break;
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT: case DXGI_FORMAT_R32_FLOAT: bytes = 4; break;
    default: return E_INVALIDARG;
    }
    const UINT max_dim = max_dimension_;
    const UINT width = static_cast<UINT>(std::min<std::size_t>(count, max_dim));
    const std::size_t h = (count + width - 1u) / width;
    if (h > max_dim) return E_INVALIDARG;
    auto texture = std::make_shared<gpuMemBase>();
    HRESULT hr = create_texture(device_.Get(), width, static_cast<UINT>(h), format,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, D3D11_USAGE_DEFAULT, 0, texture->texture);
    if (FAILED(hr)) return hr;
    hr = create_srv(device_.Get(), texture->texture.Get(), format, texture->srv);
    if (FAILED(hr)) return hr;
    hr = create_rtv(device_.Get(), texture->texture.Get(), format, texture->rtv);
    if (FAILED(hr)) return hr;
    texture->width = width; texture->height = static_cast<UINT>(h); texture->count = count;
    if (data) {
        const std::size_t capacity = static_cast<std::size_t>(width) * h;
        std::vector<std::uint8_t> padded(capacity * bytes, 0);
        std::memcpy(padded.data(), data, count * bytes);
        context->UpdateSubresource(texture->texture.Get(), 0, nullptr, padded.data(), width * bytes, 0);
    }
    out = std::move(texture);
    return S_OK;
}

HRESULT ArenaTexture::copy_gpu_buffers(
    ID3D11DeviceContext* context, DXGI_FORMAT format, const void* a, const void* b,
    std::size_t count, std::shared_ptr<void>& gpu_a, std::shared_ptr<void>& gpu_b,
    std::shared_ptr<void>& gpu_out)
{
    if (!a || !b || count == 0) return E_INVALIDARG;
    std::shared_ptr<gpuMemBase> ta, tb, to;
    HRESULT hr = create_gpu_buffer(context, format, a, count, ta); if (FAILED(hr)) return hr;
    hr = create_gpu_buffer(context, format, b, count, tb); if (FAILED(hr)) return hr;
    hr = create_gpu_buffer(context, format, nullptr, count, to); if (FAILED(hr)) return hr;
    gpu_a = std::move(ta); gpu_b = std::move(tb); gpu_out = std::move(to);
    return S_OK;
}

HRESULT ArenaTexture::readback_gpu_buffer(
    ID3D11DeviceContext* context, const std::shared_ptr<void>& opaque,
    void* out, std::size_t count)
{
    if (!context || !opaque || !out || count == 0) return E_INVALIDARG;
    auto texture = std::static_pointer_cast<gpuMemBase>(opaque);
    if (count > texture->count || texture->width == 0) return E_INVALIDARG;
    D3D11_TEXTURE2D_DESC desc{}; texture->texture->GetDesc(&desc);
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.MiscFlags = 0;
    HRESULT hr = device_->CreateTexture2D(&desc, nullptr, &staging);
    if (FAILED(hr)) return hr;
    const UINT rows = static_cast<UINT>((count + texture->width - 1u) / texture->width);
    const UINT copied_width = static_cast<UINT>(std::min<std::size_t>(count, texture->width));
    const D3D11_BOX used_region{0, 0, 0, copied_width, rows, 1};
    context->CopySubresourceRegion(
        staging.Get(), 0, 0, 0, 0, texture->texture.Get(), 0, &used_region);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return hr;
    UINT bytes = desc.Format == DXGI_FORMAT_R8_UINT || desc.Format == DXGI_FORMAT_R8_SINT ? 1u :
        (desc.Format == DXGI_FORMAT_R16_UINT || desc.Format == DXGI_FORMAT_R16_SINT || desc.Format == DXGI_FORMAT_R16_FLOAT ? 2u : 4u);
    auto* dst = static_cast<std::uint8_t*>(out);
    const auto* src = static_cast<const std::uint8_t*>(mapped.pData);
    std::size_t remaining = count;
    for (UINT row = 0; remaining && row < texture->height; ++row) {
        const std::size_t n = std::min<std::size_t>(remaining, texture->width);
        std::memcpy(dst, src + static_cast<std::size_t>(row) * mapped.RowPitch, n * bytes);
        dst += n * bytes; remaining -= n;
    }
    context->Unmap(staging.Get(), 0);
    return remaining == 0 ? S_OK : E_FAIL;
}

std::shared_ptr<gpuMemBase> ArenaTexture::texture_from_handle(const std::shared_ptr<void>& handle) const noexcept
{
    if (!handle) return {};
    return std::static_pointer_cast<gpuMemBase>(handle);
}

} // namespace dxblas::detail
