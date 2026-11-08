#include "backend/d3d11_shader_manager.hpp"

#include <d3dcompiler.h>

#include <fstream>
#include <sstream>
#include <string>

namespace dxblas::detail {

namespace {

constexpr const char* kFullscreenVertexShaderSource = R"(
struct VSOutput
{
    float4 position : SV_Position;
};

VSOutput main(uint vertex_id : SV_VertexID)
{
    VSOutput output;

    // Full-screen triangle. No vertex/index buffer is required.
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0)
    };

    output.position = float4(positions[vertex_id], 0.0, 1.0);
    return output;
}
)";

constexpr const char* kSumPixelShaderSource = R"(
#ifndef DTYPE
#define DTYPE float
#endif
Texture2D<DTYPE> input_a : register(t0);
Texture2D<DTYPE> input_b : register(t1);
struct PSInput { float4 position : SV_Position; };
#if defined(UNSIGNED_INTEGER_OUTPUT)
uint main(PSInput input) : SV_Target { int2 p=int2(input.position.xy); return input_a.Load(int3(p,0))+input_b.Load(int3(p,0)); }
#elif defined(SIGNED_INTEGER_OUTPUT)
int main(PSInput input) : SV_Target { int2 p=int2(input.position.xy); return input_a.Load(int3(p,0))+input_b.Load(int3(p,0)); }
#else
DTYPE main(PSInput input) : SV_Target { int2 p=int2(input.position.xy); return input_a.Load(int3(p,0))+input_b.Load(int3(p,0)); }
#endif
)";

constexpr const char* kMulPixelShaderSource = R"(
#ifndef DTYPE
#define DTYPE float
#endif
Texture2D<DTYPE> input_a : register(t0);
Texture2D<DTYPE> input_b : register(t1);
struct PSInput { float4 position : SV_Position; };
#if defined(UNSIGNED_INTEGER_OUTPUT)
uint main(PSInput input) : SV_Target { int2 p=int2(input.position.xy); return input_a.Load(int3(p,0))*input_b.Load(int3(p,0)); }
#elif defined(SIGNED_INTEGER_OUTPUT)
int main(PSInput input) : SV_Target { int2 p=int2(input.position.xy); return input_a.Load(int3(p,0))*input_b.Load(int3(p,0)); }
#else
DTYPE main(PSInput input) : SV_Target { int2 p=int2(input.position.xy); return input_a.Load(int3(p,0))*input_b.Load(int3(p,0)); }
#endif
)";

// Embedded fallback so GEMM does not depend on the process working directory.
// The external src/shaders/gemm.hlsl remains the editable source of record.
constexpr const char* kGemmPixelShaderSource = R"(
// Row-major FP32 GEMM using flattened gpuMem texture storage. The source
// arrays may have texture widths unrelated to the logical matrix dimensions.
Texture2D<float> matrix_a : register(t0);
Texture2D<float> matrix_b : register(t1);
cbuffer GemmConstants : register(b0)
{
    uint M;
    uint N;
    uint K;
    uint AWidth;
    uint BWidth;
    uint OutWidth;
    uint Padding0;
    uint Padding1;
};
struct PSInput { float4 position : SV_Position; };
float main(PSInput input) : SV_Target
{
    const uint outputIndex = (uint)input.position.y * OutWidth + (uint)input.position.x;
    if (outputIndex >= M * N) return 0.0f;
    const uint row = outputIndex / N;
    const uint column = outputIndex - row * N;
    float accumulator = 0.0f;
    [loop]
    for (uint inner = 0; inner < K; ++inner)
    {
        const uint aIndex = row * K + inner;
        const uint bIndex = inner * N + column;
        const float av = matrix_a.Load(int3((int)(aIndex % AWidth), (int)(aIndex / AWidth), 0));
        const float bv = matrix_b.Load(int3((int)(bIndex % BWidth), (int)(bIndex / BWidth), 0));
        accumulator += av * bv;
    }
    return accumulator;
}
)";

const char* get_dtype_string(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R32_UINT:
        return "uint";
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R32_SINT:
        return "int";
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
        return "float";
    default:
        return nullptr;
    }
}

std::string load_shader_source(BinaryOp op, const char*& source_name) {
    const char* filename = nullptr;
    const char* fallback_source = nullptr;

    switch (op) {
    case BinaryOp::Add:
        filename = "sum.hlsl";
        source_name = "sum.hlsl";
        fallback_source = kSumPixelShaderSource;
        break;
    case BinaryOp::Mul:
        filename = "mul.hlsl";
        source_name = "mul.hlsl";
        fallback_source = kMulPixelShaderSource;
        break;
    case BinaryOp::Gemm:
        filename = "gemm.hlsl";
        source_name = "gemm.hlsl";
        fallback_source = kGemmPixelShaderSource;
        break;
    default:
        source_name = "unknown.hlsl";
        return {};
    }

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
            if (!content.empty())
                return content;
        }
    }

    return fallback_source ? std::string(fallback_source) : std::string{};
}

HRESULT compile_shader(const std::string& source, const char* source_name,
                       const D3D_SHADER_MACRO* defines, const char* profile,
                       Microsoft::WRL::ComPtr<ID3DBlob>& bytecode, std::string& last_error) {
    UINT compile_flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG) || defined(DEBUG)
    compile_flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    compile_flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

    Microsoft::WRL::ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(source.c_str(), source.size(), source_name, defines,
                            D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", profile, compile_flags, 0,
                            &bytecode, &errors);

    if (FAILED(hr)) {
        if (errors) {
            last_error.assign(static_cast<const char*>(errors->GetBufferPointer()),
                              errors->GetBufferSize());
            OutputDebugStringA(last_error.c_str());
        } else {
            last_error = "D3DCompile failed without error blob.";
        }
    }

    return hr;
}

} // namespace

HRESULT D3D11ShaderManager::get_or_compile(ID3D11Device* device, BinaryOp op, DXGI_FORMAT format,
                                           Microsoft::WRL::ComPtr<ID3D11PixelShader>& out_shader) {
    if (!device)
        return E_POINTER;

    const ShaderKey key{op, format, format};
    auto it = pixel_cache_.find(key);
    if (it != pixel_cache_.end()) {
        out_shader = it->second;
        return S_OK;
    }

    const char* dtype = get_dtype_string(format);
    if (!dtype) {
        last_error_ = "Unsupported DXGI_FORMAT for pixel shader.";
        return E_INVALIDARG;
    }

    const char* source_name = nullptr;
    std::string shader_code = load_shader_source(op, source_name);
    if (shader_code.empty()) {
        last_error_ = "Failed to load pixel shader source.";
        return E_FAIL;
    }

    const bool unsigned_integer = format == DXGI_FORMAT_R8_UINT || format == DXGI_FORMAT_R16_UINT ||
                                  format == DXGI_FORMAT_R32_UINT;
    const bool signed_integer = format == DXGI_FORMAT_R8_SINT || format == DXGI_FORMAT_R16_SINT ||
                                format == DXGI_FORMAT_R32_SINT;

    D3D_SHADER_MACRO defines[3]{};
    defines[0] = {"DTYPE", dtype};
    if (unsigned_integer)
        defines[1] = {"UNSIGNED_INTEGER_OUTPUT", "1"};
    else if (signed_integer)
        defines[1] = {"SIGNED_INTEGER_OUTPUT", "1"};

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    HRESULT hr = compile_shader(shader_code, source_name, defines, "ps_5_0", bytecode, last_error_);
    if (FAILED(hr))
        return hr;

    hr = device->CreatePixelShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr,
                                   &out_shader);

    if (FAILED(hr)) {
        last_error_ = "CreatePixelShader failed with HRESULT: " + std::to_string(hr);
        return hr;
    }

    pixel_cache_[key] = out_shader;
    return S_OK;
}

HRESULT D3D11ShaderManager::get_or_compile_fullscreen_vertex_shader(
    ID3D11Device* device, Microsoft::WRL::ComPtr<ID3D11VertexShader>& out_shader,
    Microsoft::WRL::ComPtr<ID3DBlob>& out_bytecode) {
    if (!device)
        return E_POINTER;

    if (fullscreen_vertex_shader_) {
        out_shader = fullscreen_vertex_shader_;
        out_bytecode = fullscreen_vertex_bytecode_;
        return S_OK;
    }

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    HRESULT hr = compile_shader(kFullscreenVertexShaderSource, "fullscreen_triangle.hlsl", nullptr,
                                "vs_5_0", bytecode, last_error_);
    if (FAILED(hr))
        return hr;

    hr = device->CreateVertexShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
                                    nullptr, &fullscreen_vertex_shader_);
    if (FAILED(hr)) {
        last_error_ = "CreateVertexShader failed with HRESULT: " + std::to_string(hr);
        return hr;
    }

    fullscreen_vertex_bytecode_ = bytecode;
    out_shader = fullscreen_vertex_shader_;
    out_bytecode = fullscreen_vertex_bytecode_;
    return S_OK;
}

void D3D11ShaderManager::clear() noexcept {
    pixel_cache_.clear();
    fullscreen_vertex_shader_.Reset();
    fullscreen_vertex_bytecode_.Reset();
}

} // namespace dxblas::detail
