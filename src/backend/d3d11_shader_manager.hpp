#pragma once

#include <d3d11.h>
#include <dxgiformat.h>
#include <wrl/client.h>

#include <cstddef>
#include <string>
#include <unordered_map>

namespace dxblas::detail {

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

class D3D11ShaderManager {
public:
    D3D11ShaderManager() = default;
    ~D3D11ShaderManager() = default;

    D3D11ShaderManager(const D3D11ShaderManager&) = delete;
    D3D11ShaderManager& operator=(const D3D11ShaderManager&) = delete;
    D3D11ShaderManager(D3D11ShaderManager&&) noexcept = default;
    D3D11ShaderManager& operator=(D3D11ShaderManager&&) noexcept = default;

    HRESULT get_or_compile(
        ID3D11Device* device,
        BinaryOp op,
        DXGI_FORMAT format,
        Microsoft::WRL::ComPtr<ID3D11ComputeShader>& out_shader);

    const std::string& last_error() const noexcept { return last_error_; }
    void clear() noexcept { cache_.clear(); }

private:
    struct ShaderKeyHash {
        std::size_t operator()(const ShaderKey& key) const noexcept
        {
            return (static_cast<std::size_t>(key.op) << 16) ^ static_cast<std::size_t>(key.format);
        }
    };

    std::unordered_map<ShaderKey, Microsoft::WRL::ComPtr<ID3D11ComputeShader>, ShaderKeyHash> cache_;
    std::string last_error_;
};

} // namespace dxblas::detail
