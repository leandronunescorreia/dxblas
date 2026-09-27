#pragma once

#include <cstddef>

#include <dxgiformat.h>
#include <windows.h>

namespace dxblas {

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual HRESULT initialize() = 0;

    virtual HRESULT sum(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count) = 0;

    virtual HRESULT mul(
        DXGI_FORMAT format,
        const void* a,
        const void* b,
        void* out,
        std::size_t count) = 0;
};

} // namespace dxblas
