#include "backend/d3d11_shader_manager.hpp"

#include <d3dcompiler.h>

#include <fstream>
#include <sstream>
#include <string>

namespace dxblas::detail {

namespace {

constexpr const char* kSumShaderSource = R"(
#ifndef DTYPE
#define DTYPE float
#endif

Buffer<DTYPE> input_a : register(t0);
Buffer<DTYPE> input_b : register(t1);
RWBuffer<DTYPE> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] + input_b[id.x];
}
)";

constexpr const char* kMulShaderSource = R"(
#ifndef DTYPE
#define DTYPE float
#endif

Buffer<DTYPE> input_a : register(t0);
Buffer<DTYPE> input_b : register(t1);
RWBuffer<DTYPE> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] * input_b[id.x];
}
)";

const char* get_dtype_string(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R32_UINT:
        return "uint";
    case DXGI_FORMAT_R32_FLOAT:
        return "float";
    default:
        return nullptr;
    }
}

std::string load_shader_source(BinaryOp op, const char*& source_name)
{
    const char* filename = nullptr;
    const char* fallback_source = nullptr;

    switch (op) {
    case BinaryOp::Add:
        filename = "sum.hlsl";
        source_name = "sum.hlsl";
        fallback_source = kSumShaderSource;
        break;
    case BinaryOp::Mul:
        filename = "mul.hlsl";
        source_name = "mul.hlsl";
        fallback_source = kMulShaderSource;
        break;
    default:
        source_name = "unknown.hlsl";
        return {};
    }

    // Attempt to load from disk (supports live editing and debugging)
    const std::string search_paths[] = {
        std::string("src/shaders/") + filename,
        std::string("../src/shaders/") + filename,
        std::string("shaders/") + filename,
        std::string("./") + filename,
    };

    for (const auto& path : search_paths) {
        std::ifstream file(path, std::ios::in | std::ios::binary);
        if (file.is_open()) {
            std::ostringstream ss;
            ss << file.rdbuf();
            std::string content = ss.str();
            if (!content.empty()) {
                return content;
            }
        }
    }

    // Gracefully fall back to embedded HLSL source if not available on disk
    return std::string(fallback_source);
}

} // namespace

HRESULT D3D11ShaderManager::get_or_compile(
    ID3D11Device* device,
    BinaryOp op,
    DXGI_FORMAT format,
    Microsoft::WRL::ComPtr<ID3D11ComputeShader>& out_shader)
{
    if (!device)
        return E_POINTER;

    // Check cache first
    const ShaderKey key{op, format};
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        out_shader = it->second;
        return S_OK;
    }

    // Resolve preprocessor configuration
    const char* dtype = get_dtype_string(format);
    if (!dtype) {
        last_error_ = "Unsupported DXGI_FORMAT for compute shader.";
        return E_INVALIDARG;
    }

    const char* source_name = nullptr;
    std::string shader_code = load_shader_source(op, source_name);
    if (shader_code.empty()) {
        last_error_ = "Failed to load shader source.";
        return E_FAIL;
    }

    D3D_SHADER_MACRO defines[] = {
        {"DTYPE", dtype},
        {nullptr, nullptr}
    };

    // Configure compilation flags following Direct3D best practices
    UINT compile_flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG) || defined(DEBUG)
    compile_flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    compile_flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;

    HRESULT hr = D3DCompile(
        shader_code.c_str(),
        shader_code.size(),
        source_name,
        defines,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "cs_5_0",
        compile_flags,
        0,
        &bytecode,
        &errors);

    if (FAILED(hr)) {
        if (errors) {
            last_error_.assign(
                static_cast<const char*>(errors->GetBufferPointer()),
                errors->GetBufferSize());
            OutputDebugStringA(last_error_.c_str());
        } else {
            last_error_ = "D3DCompile failed without error blob.";
        }
        return hr;
    }

    hr = device->CreateComputeShader(
        bytecode->GetBufferPointer(),
        bytecode->GetBufferSize(),
        nullptr,
        &out_shader);

    if (FAILED(hr)) {
        last_error_ = "CreateComputeShader failed with HRESULT: " + std::to_string(hr);
        return hr;
    }

    cache_[key] = out_shader;
    return S_OK;
}

} // namespace dxblas::detail
