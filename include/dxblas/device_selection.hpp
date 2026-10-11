#pragma once

// Shared D3D11 adapter selection for the library and standalone samples.
// Hardware adapters are ranked by dedicated video memory; this is a practical
// default heuristic, not a guarantee of peak benchmark performance.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace dxblas::detail {

inline std::string adapter_name_utf8(const wchar_t* name)
{
    if (!name || !*name) return "Unknown adapter";
    const int required = WideCharToMultiByte(CP_UTF8, 0, name, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) return "Unknown adapter";
    std::string result(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, 0, name, -1, result.data(), required, nullptr, nullptr);
    result.resize(static_cast<std::size_t>(required - 1));
    return result;
}

inline const char* feature_level_name(D3D_FEATURE_LEVEL level) noexcept
{
    switch (level) {
    case D3D_FEATURE_LEVEL_11_1: return "11.1";
    case D3D_FEATURE_LEVEL_11_0: return "11.0";
    case D3D_FEATURE_LEVEL_10_1: return "10.1";
    case D3D_FEATURE_LEVEL_10_0: return "10.0";
    case D3D_FEATURE_LEVEL_9_3: return "9.3";
    case D3D_FEATURE_LEVEL_9_2: return "9.2";
    case D3D_FEATURE_LEVEL_9_1: return "9.1";
    default: return "unknown";
    }
}

struct D3D11AdapterCandidate {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 description{};
    UINT enumeration_index = 0;
    bool software = false;
};

inline HRESULT create_device_for_adapter(
    IDXGIAdapter* adapter,
    UINT requested_flags,
    Microsoft::WRL::ComPtr<ID3D11Device>& device,
    D3D_FEATURE_LEVEL& feature_level,
    Microsoft::WRL::ComPtr<ID3D11DeviceContext>& immediate_context)
{
    const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    const D3D_FEATURE_LEVEL fallback_levels[] = { D3D_FEATURE_LEVEL_11_0 };

    auto attempt = [&](UINT flags, const D3D_FEATURE_LEVEL* requested_levels, UINT level_count) {
        device.Reset();
        immediate_context.Reset();
        feature_level = {};
        return D3D11CreateDevice(
            adapter,
            adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            requested_levels,
            level_count,
            D3D11_SDK_VERSION,
            &device,
            &feature_level,
            &immediate_context);
    };

    HRESULT hr = attempt(requested_flags, levels, ARRAYSIZE(levels));
    if (hr == E_INVALIDARG)
        hr = attempt(requested_flags, fallback_levels, ARRAYSIZE(fallback_levels));

    if (FAILED(hr) && (requested_flags & D3D11_CREATE_DEVICE_DEBUG)) {
        const UINT flags_without_debug = requested_flags & ~D3D11_CREATE_DEVICE_DEBUG;
        hr = attempt(flags_without_debug, levels, ARRAYSIZE(levels));
        if (hr == E_INVALIDARG)
            hr = attempt(flags_without_debug, fallback_levels, ARRAYSIZE(fallback_levels));
    }
    return hr;
}

inline HRESULT create_best_d3d11_device(
    UINT device_flags,
    Microsoft::WRL::ComPtr<ID3D11Device>& device,
    D3D_FEATURE_LEVEL& feature_level,
    Microsoft::WRL::ComPtr<ID3D11DeviceContext>& immediate_context,
    std::string* selected_adapter_name = nullptr)
{
    using Microsoft::WRL::ComPtr;

    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::cerr << "[dxBLAS] Could not enumerate DXGI adapters (HRESULT 0x"
                  << std::hex << static_cast<unsigned long>(hr) << std::dec << ").\n";
        return hr;
    }

    std::vector<D3D11AdapterCandidate> adapters;
    for (UINT index = 0;; ++index) {
        D3D11AdapterCandidate candidate;
        hr = factory->EnumAdapters1(index, &candidate.adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr)) return hr;
        hr = candidate.adapter->GetDesc1(&candidate.description);
        if (FAILED(hr)) continue;
        candidate.enumeration_index = index;
        candidate.software = (candidate.description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        adapters.push_back(candidate);
    }

    std::cout << "[dxBLAS] DirectX GPU adapters detected: " << adapters.size() << '\n';
    for (const auto& adapter : adapters) {
        const double dedicated_mib = static_cast<double>(adapter.description.DedicatedVideoMemory) /
                                     (1024.0 * 1024.0);
        std::cout << "  GPU " << adapter.enumeration_index << ": "
                  << adapter_name_utf8(adapter.description.Description)
                  << " | Dedicated VRAM: " << static_cast<unsigned long long>(dedicated_mib)
                  << " MiB | " << (adapter.software ? "software adapter" : "hardware adapter")
                  << '\n';
    }

    // Prefer real hardware, then the largest dedicated video-memory budget.
    // Enumeration order breaks ties (e.g. integrated GPUs with 0 dedicated VRAM).
    std::stable_sort(adapters.begin(), adapters.end(), [](const auto& left, const auto& right) {
        if (left.software != right.software) return !left.software;
        return left.description.DedicatedVideoMemory > right.description.DedicatedVideoMemory;
    });

    for (const auto& candidate : adapters) {
        if (candidate.software) continue;
        hr = create_device_for_adapter(
            candidate.adapter.Get(), device_flags, device, feature_level, immediate_context);
        if (FAILED(hr)) {
            std::cerr << "[dxBLAS] Could not initialize GPU '"
                      << adapter_name_utf8(candidate.description.Description)
                      << "' (HRESULT 0x" << std::hex << static_cast<unsigned long>(hr)
                      << std::dec << "); trying the next hardware adapter.\n";
            continue;
        }

        const std::string name = adapter_name_utf8(candidate.description.Description);
        if (selected_adapter_name) *selected_adapter_name = name;
        std::cout << "[dxBLAS] Selected GPU: " << name
                  << " | Feature level: " << feature_level_name(feature_level)
                  << " | Dedicated VRAM: "
                  << static_cast<unsigned long long>(candidate.description.DedicatedVideoMemory / (1024u * 1024u))
                  << " MiB\n";
        return S_OK;
    }

    std::cerr << "[dxBLAS] No usable hardware GPU adapter was found. WARP/software fallback is disabled.\n";
    return DXGI_ERROR_NOT_FOUND;
}

} // namespace dxblas::detail
