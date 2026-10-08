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
        Microsoft::WRL::ComPtr<ID3D11PixelShader>& out_shader);

    HRESULT get_or_compile_fullscreen_vertex_shader(
        ID3D11Device* device,
        Microsoft::WRL::ComPtr<ID3D11VertexShader>& out_shader,
        Microsoft::WRL::ComPtr<ID3DBlob>& out_bytecode);

    const std::string& last_error() const noexcept { return last_error_; }
    void clear() noexcept;

private:
    struct ShaderKeyHash {
        std::size_t operator()(const ShaderKey& key) const noexcept
        {
            return (static_cast<std::size_t>(key.op) << 16) ^ static_cast<std::size_t>(key.format);
        }
    };

    std::unordered_map<ShaderKey, Microsoft::WRL::ComPtr<ID3D11PixelShader>, ShaderKeyHash> pixel_cache_;
    Microsoft::WRL::ComPtr<ID3D11VertexShader> fullscreen_vertex_shader_;
    Microsoft::WRL::ComPtr<ID3DBlob> fullscreen_vertex_bytecode_;
    std::string last_error_;
};

} // namespace dxblas::detail
