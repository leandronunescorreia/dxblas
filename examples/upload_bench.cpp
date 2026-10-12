#include <dxblas/dxblas.hpp>
#include <dxblas/device_selection.hpp>

#include <wrl/client.h>

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
constexpr UINT kTextureWidth = 16384;
constexpr int kWarmupIterations = 20;
constexpr int kMeasuredIterations = 100;

struct UploadTarget {
    ComPtr<ID3D11Texture2D> destination;
    ComPtr<ID3D11Texture2D> staging;
    UINT height = 0;
    std::size_t count = 0;
};

struct Result {
    double median_ms = std::numeric_limits<double>::quiet_NaN();
    double gb_s = std::numeric_limits<double>::quiet_NaN();
};

std::string adapter_name(const DXGI_ADAPTER_DESC1& desc)
{
    return dxblas::detail::adapter_name_utf8(desc.Description);
}

double median(std::vector<double> values)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    const double upper = *middle;
    if (values.size() % 2 != 0) return upper;
    return (upper + *std::max_element(values.begin(), middle)) * 0.5;
}

double throughput_gb_s(std::size_t bytes, double milliseconds)
{
    if (!(milliseconds > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(bytes) / (milliseconds * 1.0e6);
}

void print_number(double value, int width = 12)
{
    if (value == value && value != std::numeric_limits<double>::infinity() &&
        value != -std::numeric_limits<double>::infinity()) {
        std::cout << std::setw(width) << std::fixed << std::setprecision(3) << value;
    } else {
        std::cout << std::setw(width) << "n/a";
    }
}

HRESULT create_texture_pair(ID3D11Device* device,
                            std::size_t count,
                            bool with_staging,
                            UploadTarget& out)
{
    if (!device || count == 0 || count % kTextureWidth != 0)
        return E_INVALIDARG;

    const std::size_t rows = count / kTextureWidth;
    if (rows == 0 || rows > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return E_INVALIDARG;

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = kTextureWidth;
    desc.Height = static_cast<UINT>(rows);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &out.destination);
    if (FAILED(hr)) return hr;

    if (with_staging) {
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateTexture2D(&desc, nullptr, &out.staging);
        if (FAILED(hr)) return hr;
    }

    out.height = static_cast<UINT>(rows);
    out.count = count;
    return S_OK;
}

// Strategy A: mirror the library's UpdateSubresource upload: a full-row box
// plus a separate tail box if the element count does not end on a row boundary.
void upload_update_subresource(ID3D11DeviceContext* context,
                               const UploadTarget& target,
                               const float* data)
{
    const std::size_t full_rows = target.count / kTextureWidth;
    const std::size_t tail_elements = target.count % kTextureWidth;
    constexpr UINT row_bytes = kTextureWidth * sizeof(float);

    if (full_rows != 0) {
        const D3D11_BOX full_rows_box{
            0, 0, 0, kTextureWidth, static_cast<UINT>(full_rows), 1};
        context->UpdateSubresource(target.destination.Get(), 0, &full_rows_box,
                                   data, row_bytes, 0);
    }

    if (tail_elements != 0) {
        const UINT tail_y = static_cast<UINT>(full_rows);
        const UINT tail_bytes = static_cast<UINT>(tail_elements * sizeof(float));
        const D3D11_BOX tail_box{
            0, tail_y, 0, static_cast<UINT>(tail_elements), tail_y + 1, 1};
        context->UpdateSubresource(
            target.destination.Get(), 0, &tail_box,
            data + full_rows * static_cast<std::size_t>(kTextureWidth),
            tail_bytes, 0);
    }
}

// Strategy B: persistently allocated staging texture, mapped and filled per
// iteration, then copied into the DEFAULT texture. Respect RowPitch on all rows.
HRESULT upload_staging_copy(ID3D11DeviceContext* context,
                            const UploadTarget& target,
                            const float* data)
{
    if (!context || !target.staging || !target.destination || !data)
        return E_INVALIDARG;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = context->Map(target.staging.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    if (FAILED(hr)) return hr;

    constexpr std::size_t row_bytes = static_cast<std::size_t>(kTextureWidth) * sizeof(float);
    if (mapped.RowPitch == row_bytes) {
        std::memcpy(mapped.pData, data, target.count * sizeof(float));
    } else {
        auto* destination = static_cast<std::uint8_t*>(mapped.pData);
        const auto* source = reinterpret_cast<const std::uint8_t*>(data);
        for (UINT row = 0; row < target.height; ++row) {
            std::memcpy(destination + static_cast<std::size_t>(row) * mapped.RowPitch,
                        source + static_cast<std::size_t>(row) * row_bytes,
                        row_bytes);
        }
    }
    context->Unmap(target.staging.Get(), 0);

    context->CopySubresourceRegion(target.destination.Get(), 0, 0, 0, 0,
                                   target.staging.Get(), 0, nullptr);
    return S_OK;
}

HRESULT wait_for_gpu(ID3D11DeviceContext* context, ID3D11Query* event_query)
{
    if (!context || !event_query) return E_POINTER;
    context->End(event_query);
    context->Flush();

    BOOL completed = FALSE;
    for (;;) {
        const HRESULT hr = context->GetData(event_query, &completed, sizeof(completed), 0);
        if (hr == S_OK && completed) return S_OK;
        if (FAILED(hr)) return hr;
        SwitchToThread();
    }
}

HRESULT run_iterations(ID3D11DeviceContext* context,
                       ID3D11Query* event_query,
                       UploadTarget& target,
                       const float* data,
                       bool use_staging,
                       Result& result)
{
    auto perform_one = [&]() -> HRESULT {
        if (use_staging) return upload_staging_copy(context, target, data);
        upload_update_subresource(context, target, data);
        return S_OK;
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        HRESULT hr = perform_one();
        if (FAILED(hr)) return hr;
        hr = wait_for_gpu(context, event_query);
        if (FAILED(hr)) return hr;
    }

    std::vector<double> times;
    times.reserve(kMeasuredIterations);
    for (int i = 0; i < kMeasuredIterations; ++i) {
        const auto start = Clock::now();
        HRESULT hr = perform_one();
        if (FAILED(hr)) return hr;
        hr = wait_for_gpu(context, event_query);
        const auto stop = Clock::now();
        if (FAILED(hr)) return hr;
        times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    result.median_ms = median(std::move(times));
    result.gb_s = throughput_gb_s(target.count * sizeof(float), result.median_ms);
    return S_OK;
}

HRESULT acquire_default_device(ComPtr<ID3D11Device>& device,
                               ComPtr<ID3D11DeviceContext>& context)
{
    dxblas::Context dx_context;
    HRESULT hr = dx_context.initialize();
    if (FAILED(hr)) return hr;

    dxblas::NativeD3D11 native{};
    hr = dxblas::get_native_d3d11(dx_context, native);
    if (FAILED(hr)) return hr;

    // get_native_d3d11 returns AddRef'd pointers. They remain valid after the
    // short-lived Context is destroyed because these ComPtrs own the references.
    device.Attach(native.device);
    context.Attach(native.context);
    return S_OK;
}

HRESULT acquire_indexed_device(UINT requested_index,
                               ComPtr<ID3D11Device>& device,
                               ComPtr<ID3D11DeviceContext>& context,
                               std::string& name)
{
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapters1(requested_index, &adapter);
    if (FAILED(hr)) return hr;

    DXGI_ADAPTER_DESC1 desc{};
    hr = adapter->GetDesc1(&desc);
    if (FAILED(hr)) return hr;
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
        std::cerr << "Refusing software adapter index " << requested_index << ".\n";
        return E_INVALIDARG;
    }

    D3D_FEATURE_LEVEL feature_level{};
    hr = dxblas::detail::create_device_for_adapter(
        adapter.Get(), 0, device, feature_level, context);
    if (SUCCEEDED(hr)) name = adapter_name(desc);
    return hr;
}

void print_adapters()
{
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return;
    std::cout << "DXGI adapters (use --adapter-index N to select a hardware adapter):\n";
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT hr = factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr)) break;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        const bool software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        std::cout << "  " << i << ": " << adapter_name(desc)
                  << (software ? " [software]" : " [hardware]") << '\n';
    }
}

HRESULT benchmark_size(ID3D11Device* device,
                       ID3D11DeviceContext* context,
                       ID3D11Query* event_query,
                       const char* label,
                       std::size_t count)
{
    UploadTarget update_target;
    UploadTarget staging_target;
    HRESULT hr = create_texture_pair(device, count, false, update_target);
    if (FAILED(hr)) return hr;
    hr = create_texture_pair(device, count, true, staging_target);
    if (FAILED(hr)) return hr;

    // A deterministic payload is sufficient: this benchmark measures upload
    // mechanics, not shader output. Allocate it once outside measured intervals.
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i)
        data[i] = static_cast<float>(static_cast<int>(i % 251) - 125) * 0.01f;

    Result update_result;
    Result staging_result;
    hr = run_iterations(context, event_query, update_target, data.data(), false, update_result);
    if (FAILED(hr)) return hr;
    hr = run_iterations(context, event_query, staging_target, data.data(), true, staging_result);
    if (FAILED(hr)) return hr;

    std::cout << std::left << std::setw(10) << label << " | ";
    print_number(update_result.median_ms); std::cout << " | ";
    print_number(update_result.gb_s); std::cout << " | ";
    print_number(staging_result.median_ms); std::cout << " | ";
    print_number(staging_result.gb_s); std::cout << " | ";
    const double ratio = staging_result.median_ms > 0.0
        ? update_result.median_ms / staging_result.median_ms
        : std::numeric_limits<double>::quiet_NaN();
    print_number(ratio); std::cout << "x\n";
    return S_OK;
}

} // namespace

int main(int argc, char** argv)
{
    bool list_only = false;
    bool has_adapter_index = false;
    UINT adapter_index = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--list-adapters") {
            list_only = true;
        } else if (arg == "--adapter-index" && i + 1 < argc) {
            try {
                const unsigned long parsed = std::stoul(argv[++i]);
                if (parsed > std::numeric_limits<UINT>::max()) throw std::out_of_range("adapter index");
                adapter_index = static_cast<UINT>(parsed);
                has_adapter_index = true;
            } catch (...) {
                std::cerr << "Invalid --adapter-index value.\n";
                return 2;
            }
        } else {
            std::cerr << "Usage: dxblas_upload_bench [--list-adapters] [--adapter-index N]\n";
            return 2;
        }
    }

    if (list_only) {
        print_adapters();
        if (argc == 2) return 0;
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::string selected_name;
    HRESULT hr = S_OK;
    if (has_adapter_index) {
        hr = acquire_indexed_device(adapter_index, device, context, selected_name);
    } else {
        hr = acquire_default_device(device, context);
        if (SUCCEEDED(hr)) {
            ComPtr<IDXGIDevice> dxgi_device;
            ComPtr<IDXGIAdapter> adapter;
            DXGI_ADAPTER_DESC desc{};
            if (SUCCEEDED(device.As(&dxgi_device)) &&
                SUCCEEDED(dxgi_device->GetAdapter(&adapter)) &&
                SUCCEEDED(adapter->GetDesc(&desc))) {
                selected_name = dxblas::detail::adapter_name_utf8(desc.Description);
            }
        }
    }
    if (FAILED(hr)) {
        std::cerr << "D3D11 device initialization failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }

    ComPtr<ID3D11Query> event_query;
    D3D11_QUERY_DESC query_desc{};
    query_desc.Query = D3D11_QUERY_EVENT;
    hr = device->CreateQuery(&query_desc, &event_query);
    if (FAILED(hr)) {
        std::cerr << "CreateQuery(D3D11_QUERY_EVENT) failed: HRESULT 0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << '\n';
        return 1;
    }

    constexpr std::size_t kib = 1024;
    constexpr std::size_t mib = 1024 * kib;
    const std::size_t sizes[] = {64 * kib / sizeof(float), mib / sizeof(float),
                                 16 * mib / sizeof(float), 256 * mib / sizeof(float)};
    const char* labels[] = {"64 KiB", "1 MiB", "16 MiB", "256 MiB"};

    std::cout << "dxblas_upload_bench: UpdateSubresource vs persistent staging upload\n"
              << "Adapter: " << (selected_name.empty() ? "unknown" : selected_name) << '\n'
              << "Texture: DXGI_FORMAT_R32_FLOAT, width=" << kTextureWidth
              << " elements (" << (kTextureWidth * sizeof(float)) << " bytes/row)\n"
              << "Each measured iteration waits for a D3D11 EVENT query; timings include completion.\n"
              << "Warm-up iterations: " << kWarmupIterations
              << ", measured iterations: " << kMeasuredIterations
              << ", statistic: median, throughput: decimal GB/s\n\n";
    std::cout << std::left << std::setw(10) << "Size" << " | "
              << std::right << std::setw(12) << "A ms" << " | "
              << std::setw(12) << "A GB/s" << " | "
              << std::setw(12) << "B ms" << " | "
              << std::setw(12) << "B GB/s" << " | "
              << std::setw(10) << "A/B speed" << '\n'
              << std::string(77, '-') << '\n';

    for (std::size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        hr = benchmark_size(device.Get(), context.Get(), event_query.Get(), labels[i], sizes[i]);
        if (FAILED(hr)) {
            std::cerr << "Benchmark failed for " << labels[i] << ": HRESULT 0x" << std::hex
                      << static_cast<unsigned long>(hr) << std::dec << '\n';
            return 1;
        }
    }

    std::cout << "\nDecision rule: consider switching library uploads to B only if B is >=20% faster at both 16 MiB and 256 MiB,\n"
              << "and B is no more than 10% slower at 64 KiB on the primary GPU.\n";
    return 0;
}
