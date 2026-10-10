#include "arena_texture.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>\n#include <vector>

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

    device_ = device;

    // Reserve the first arena as a common float working set. Other formats
    // are lazily promoted/reallocated by ensure() when first used.
    HRESULT hr = ensure(
        DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32_FLOAT,
        initial_width,
        initial_height);
    if (FAILED(hr)) return hr;

    // Prewarm one GPU-resident slot during Context::initialize().
    std::shared_ptr<GpuTexture> initial_slot;
    hr = acquire_float(static_cast<std::size_t>(initial_width) * initial_height, initial_slot);
    initial_slot.reset(); // return it to the arena pool for the first operation
    return hr;
}

HRESULT ArenaTexture::ensure(
    DXGI_FORMAT input_format,
    DXGI_FORMAT output_format,
    UINT width,
    UINT height)
{
    if (!device_ || width == 0 || height == 0)
        return E_INVALIDARG;

    const bool format_matches =
        input_format_ == input_format &&
        output_format_ == output_format;

    const bool capacity_matches =
        capacity_width_ >= width &&
        capacity_height_ >= height;

    if (format_matches && capacity_matches)
        return S_OK;

    // Grow geometrically. The old resources stay alive until all commands
    // referencing them are complete; D3D11 keeps the command stream valid.
    UINT new_width = std::max(width, capacity_width_);
    UINT new_height = std::max(height, capacity_height_);

    if (new_width == 0)
        new_width = width;
    if (new_height == 0)
        new_height = height;

    if (format_matches) {
        new_width = std::max(new_width, capacity_width_ * 2u);
        new_height = std::max(new_height, capacity_height_ * 2u);
    } else {
        new_width = std::max(new_width, 2048u);
        new_height = std::max(new_height, height);
    }

    // Never exceed D3D11's Texture2D dimension limit.
    new_width = std::min(new_width, static_cast<UINT>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION));
    new_height = std::min(new_height, static_cast<UINT>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION));

    if (new_width < width || new_height < height)
        return E_INVALIDARG;

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
    context->CopyResource(readback_.Get(), output_.Get());

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


HRESULT ArenaTexture::acquire_float(std::size_t count, std::shared_ptr<GpuTexture>& out)
{
    if (!device_ || count == 0 ||
        count > static_cast<std::size_t>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) *
                static_cast<std::size_t>(D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION))
        return E_INVALIDARG;

    const UINT width = static_cast<UINT>(std::min<std::size_t>(count, 4096));
    const UINT height = static_cast<UINT>((count + width - 1u) / width);
    for (auto& slot : gpu_slots_) {
        if (slot.use_count() == 1 && slot->width == width && slot->height == height) {
            slot->count = count;
            out = slot;
            return S_OK;
        }
    }

    auto slot = std::make_shared<GpuTexture>();
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

HRESULT ArenaTexture::upload_float(
    ID3D11DeviceContext* context, const void* data, std::size_t count,
    std::shared_ptr<GpuTexture>& out)
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
    ID3D11DeviceContext* context, const std::shared_ptr<GpuTexture>& source,
    void* destination, std::size_t count)
{
    if (!context || !source || !destination || count != source->count)
        return E_INVALIDARG;
    context->CopyResource(source->staging.Get(), source->texture.Get());
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


} // namespace dxblas::detail
