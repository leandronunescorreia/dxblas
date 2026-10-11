#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxblas/device_selection.hpp>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;

namespace {
constexpr UINT kDefaultN = 4096;
constexpr UINT kTile = 16;

struct MatrixConstants {
    UINT n;
    UINT padding[3]{};
};

double milliseconds(Clock::duration elapsed)
{
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

bool check_hr(HRESULT hr, const char* operation)
{
    if (SUCCEEDED(hr))
        return true;

    std::cerr << operation << " failed (HRESULT 0x"
              << std::hex << static_cast<unsigned long>(hr)
              << std::dec << ").\n";
    return false;
}

std::filesystem::path find_shader()
{
    const std::filesystem::path candidates[] = {
        L"shaders/matrix_multiply.hlsl",
        L"examples/shaders/matrix_multiply.hlsl",
        L"../examples/shaders/matrix_multiply.hlsl",
        L"H:\\dev\\gpu\\dxblas\\build\\shaders\\matrix_multiply.hlsl"
    };

    for (const auto& path : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec) && !ec)
            return path;
    }
    return {};
}

bool wait_for_query(ID3D11DeviceContext* context, ID3D11Query* query)
{
    for (;;) {
        const HRESULT hr = context->GetData(query, nullptr, 0, 0);
        if (hr == S_OK)
            return true;
        if (hr != S_FALSE)
            return check_hr(hr, "ID3D11DeviceContext::GetData");
        Sleep(0);
    }
}

bool create_structured_buffer(
    ID3D11Device* device,
    UINT byteWidth,
    UINT bindFlags,
    UINT stride,
    ID3D11Buffer** output)
{
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = byteWidth;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bindFlags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    return check_hr(device->CreateBuffer(&desc, nullptr, output),
                    "Create structured buffer");
}

bool create_timestamp_query(
    ID3D11Device* device,
    D3D11_QUERY queryType,
    ID3D11Query** output)
{
    D3D11_QUERY_DESC desc{};
    desc.Query = queryType;
    return check_hr(device->CreateQuery(&desc, output), "CreateQuery");
}

void cpu_matrix_multiply_blocked(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    UINT n)
{
    std::fill(c.begin(), c.end(), 0.0f);

    // i-k-j order keeps B and C accesses contiguous in row-major storage.
    // Blocking improves cache locality without requiring third-party BLAS.
    constexpr UINT block = 32;
    for (UINT ii = 0; ii < n; ii += block) {
        const UINT iEnd = std::min(ii + block, n);
        for (UINT jj = 0; jj < n; jj += block) {
            const UINT jEnd = std::min(jj + block, n);
            for (UINT kk = 0; kk < n; kk += block) {
                const UINT kEnd = std::min(kk + block, n);
                for (UINT i = ii; i < iEnd; ++i) {
                    float* cRow = c.data() + static_cast<std::size_t>(i) * n;
                    for (UINT k = kk; k < kEnd; ++k) {
                        const float av = a[static_cast<std::size_t>(i) * n + k];
                        const float* bRow = b.data() + static_cast<std::size_t>(k) * n;
                        for (UINT j = jj; j < jEnd; ++j)
                            cRow[j] += av * bRow[j];
                    }
                }
            }
        }
    }
}

bool validate_samples(
    const std::vector<float>& cpu,
    const std::vector<float>& gpu,
    UINT n)
{
    // Compare all cells when the matrix is small; otherwise deterministic samples
    // keep validation overhead low after the full CPU benchmark.
    const std::size_t count = cpu.size();
    const std::size_t checks = std::min<std::size_t>(count, 4096);
    float maxAbsError = 0.0f;
    for (std::size_t i = 0; i < checks; ++i) {
        const std::size_t index = count <= checks ? i : (i * count / checks);
        const float expected = cpu[index];
        const float actual = gpu[index];
        const float error = std::abs(expected - actual);
        maxAbsError = std::max(maxAbsError, error);
        const float tolerance = 2.0e-3f + 2.0e-4f * std::abs(expected);
        if (!std::isfinite(actual) || error > tolerance) {
            std::cerr << "Validation failed at (" << index / n << ", "
                      << index % n << "): CPU=" << expected
                      << ", GPU=" << actual << ", abs error=" << error << '\n';
            return false;
        }
    }

    std::cout << "Validation: " << checks
              << " cells checked; max sampled absolute error = "
              << std::scientific << maxAbsError << std::defaultfloat << '\n';
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    UINT n = kDefaultN;
    if (argc > 1) {
        try {
            const unsigned long parsed = std::stoul(argv[1]);
            if (parsed == 0 || parsed > 8192) {
                std::cerr << "Matrix size must be in [1, 8192].\n";
                return 2;
            }
            n = static_cast<UINT>(parsed);
        } catch (...) {
            std::cerr << "Usage: dxblas_matrix_multiply_4096 [matrix-size]\n";
            return 2;
        }
    }

    const std::uint64_t elements64 = static_cast<std::uint64_t>(n) * n;
    const std::uint64_t bytes64 = elements64 * sizeof(float);
    if (bytes64 > UINT_MAX) {
        std::cerr << "Matrix exceeds the Direct3D 11 buffer-size limit.\n";
        return 2;
    }

    const std::size_t elements = static_cast<std::size_t>(elements64);
    const UINT bytes = static_cast<UINT>(bytes64);

    std::cout << "dxBLAS sample: FP32 matrix multiplication benchmark\n"
              << "Matrix size: " << n << " x " << n << '\n'
              << "Operation count: approximately "
              << std::fixed << std::setprecision(3)
              << (2.0 * static_cast<double>(n) * n * n / 1.0e9)
              << " GFLOP\n"
              << "Each matrix: " << (bytes64 / (1024.0 * 1024.0))
              << " MiB\n\n";

    std::vector<float> a(elements), b(elements), cpuResult(elements), gpuResult(elements);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        a[i] = value(rng);
        b[i] = value(rng);
    }

    // CPU benchmark: single-threaded blocked FP32 reference implementation.
    std::cout << "Running CPU multiplication (single-threaded, blocked)...\n";
    const auto cpuStart = Clock::now();
    cpu_matrix_multiply_blocked(a, b, cpuResult, n);
    const auto cpuEnd = Clock::now();
    const double cpuMs = milliseconds(cpuEnd - cpuStart);
    std::cout << "CPU time: " << cpuMs << " ms ("
              << (2.0 * static_cast<double>(n) * n * n / (cpuMs * 1.0e6))
              << " GFLOP/s)\n\n";

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel{};
    HRESULT hr = dxblas::detail::create_best_d3d11_device(
        0, device, featureLevel, context);
    if (!check_hr(hr, "create_best_d3d11_device")) {
        std::cerr << "No WARP fallback is used: the benchmark must measure a hardware GPU.\n";
        return 1;
    }

    const auto shaderPath = find_shader();
    if (shaderPath.empty()) {
        std::cerr << "Could not find shaders/matrix_multiply.hlsl. "
                     "Run from the build directory or project root.\n";
        return 1;
    }

    ComPtr<ID3DBlob> shaderBlob, errors;
    hr = D3DCompileFromFile(
        shaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main", "cs_5_0",
#if defined(_DEBUG) || defined(DEBUG)
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
#else
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
#endif
        0, &shaderBlob, &errors);
    if (FAILED(hr)) {
        if (errors)
            std::cerr << static_cast<const char*>(errors->GetBufferPointer()) << '\n';
        check_hr(hr, "D3DCompileFromFile");
        return 1;
    }

    ComPtr<ID3D11ComputeShader> shader;
    if (!check_hr(device->CreateComputeShader(
            shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(),
            nullptr, &shader), "CreateComputeShader"))
        return 1;

    ComPtr<ID3D11Buffer> bufferA, bufferB, bufferC;
    if (!create_structured_buffer(device.Get(), bytes, D3D11_BIND_SHADER_RESOURCE,
                                  sizeof(float), &bufferA) ||
        !create_structured_buffer(device.Get(), bytes, D3D11_BIND_SHADER_RESOURCE,
                                  sizeof(float), &bufferB) ||
        !create_structured_buffer(device.Get(), bytes, D3D11_BIND_UNORDERED_ACCESS,
                                  sizeof(float), &bufferC))
        return 1;

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = static_cast<UINT>(elements);
    ComPtr<ID3D11ShaderResourceView> srvA, srvB;
    if (!check_hr(device->CreateShaderResourceView(bufferA.Get(), &srvDesc, &srvA),
                  "CreateShaderResourceView(A)") ||
        !check_hr(device->CreateShaderResourceView(bufferB.Get(), &srvDesc, &srvB),
                  "CreateShaderResourceView(B)"))
        return 1;

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = static_cast<UINT>(elements);
    ComPtr<ID3D11UnorderedAccessView> uavC;
    if (!check_hr(device->CreateUnorderedAccessView(bufferC.Get(), &uavDesc, &uavC),
                  "CreateUnorderedAccessView(C)"))
        return 1;

    // Upload input data. Upload and readback are reported separately from kernel time.
    const auto uploadStart = Clock::now();
    context->UpdateSubresource(bufferA.Get(), 0, nullptr, a.data(), 0, 0);
    context->UpdateSubresource(bufferB.Get(), 0, nullptr, b.data(), 0, 0);
    context->Flush();

    ComPtr<ID3D11Query> uploadFence;
    if (!create_timestamp_query(device.Get(), D3D11_QUERY_EVENT, &uploadFence))
        return 1;
    context->End(uploadFence.Get());
    if (!wait_for_query(context.Get(), uploadFence.Get()))
        return 1;
    const double uploadMs = milliseconds(Clock::now() - uploadStart);

    MatrixConstants constants{};
    constants.n = n;
    D3D11_BUFFER_DESC cbDesc{};
    cbDesc.ByteWidth = sizeof(MatrixConstants);
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ComPtr<ID3D11Buffer> constantBuffer;
    if (!check_hr(device->CreateBuffer(&cbDesc, nullptr, &constantBuffer),
                  "CreateBuffer(constants)"))
        return 1;
    context->UpdateSubresource(constantBuffer.Get(), 0, nullptr, &constants, 0, 0);

    ComPtr<ID3D11Query> disjoint, startQuery, endQuery;
    if (!create_timestamp_query(device.Get(), D3D11_QUERY_TIMESTAMP_DISJOINT, &disjoint) ||
        !create_timestamp_query(device.Get(), D3D11_QUERY_TIMESTAMP, &startQuery) ||
        !create_timestamp_query(device.Get(), D3D11_QUERY_TIMESTAMP, &endQuery))
        return 1;

    ID3D11ShaderResourceView* srvs[] = {srvA.Get(), srvB.Get()};
    ID3D11UnorderedAccessView* uavs[] = {uavC.Get()};
    ID3D11Buffer* cbs[] = {constantBuffer.Get()};

    auto dispatch = [&]() {
        context->CSSetShader(shader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 2, srvs);
        context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, cbs);
        context->Dispatch((n + kTile - 1) / kTile, (n + kTile - 1) / kTile, 1);

        ID3D11ShaderResourceView* nullSrvs[] = {nullptr, nullptr};
        ID3D11UnorderedAccessView* nullUavs[] = {nullptr};
        context->CSSetShaderResources(0, 2, nullSrvs);
        context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
    };

    // One untimed warm-up ensures compilation and initial scheduling are not counted.
    dispatch();
    context->Flush();
    ComPtr<ID3D11Query> warmupFence;
    if (!create_timestamp_query(device.Get(), D3D11_QUERY_EVENT, &warmupFence))
        return 1;
    context->End(warmupFence.Get());
    if (!wait_for_query(context.Get(), warmupFence.Get()))
        return 1;

    context->Begin(disjoint.Get());
    context->End(startQuery.Get());
    const auto gpuWallStart = Clock::now();
    dispatch();
    context->End(endQuery.Get());
    context->End(disjoint.Get());
    context->Flush();

    if (!wait_for_query(context.Get(), disjoint.Get()) ||
        !wait_for_query(context.Get(), startQuery.Get()) ||
        !wait_for_query(context.Get(), endQuery.Get()))
        return 1;

    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjointData{};
    UINT64 gpuStart = 0, gpuEnd = 0;
    if (!check_hr(context->GetData(disjoint.Get(), &disjointData,
                                  sizeof(disjointData), 0), "GetData(disjoint)") ||
        !check_hr(context->GetData(startQuery.Get(), &gpuStart,
                                  sizeof(gpuStart), 0), "GetData(start timestamp)") ||
        !check_hr(context->GetData(endQuery.Get(), &gpuEnd,
                                  sizeof(gpuEnd), 0), "GetData(end timestamp)"))
        return 1;

    const double gpuMs = (!disjointData.Disjoint && disjointData.Frequency != 0)
        ? (1000.0 * static_cast<double>(gpuEnd - gpuStart) /
           static_cast<double>(disjointData.Frequency))
        : -1.0;
    const double gpuWallMs = milliseconds(Clock::now() - gpuWallStart);

    D3D11_BUFFER_DESC stagingDesc{};
    stagingDesc.ByteWidth = bytes;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;
    stagingDesc.StructureByteStride = 0;
    ComPtr<ID3D11Buffer> staging;
    if (!check_hr(device->CreateBuffer(&stagingDesc, nullptr, &staging),
                  "CreateBuffer(staging readback)"))
        return 1;

    const auto readbackStart = Clock::now();
    context->CopyResource(staging.Get(), bufferC.Get());
    context->Flush();
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (!check_hr(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped),
                  "Map(staging readback)"))
        return 1;
    std::copy_n(static_cast<const float*>(mapped.pData), elements, gpuResult.data());
    context->Unmap(staging.Get(), 0);
    const double readbackMs = milliseconds(Clock::now() - readbackStart);

    std::cout << "GPU device: hardware D3D11, feature level 0x"
              << std::hex << static_cast<unsigned>(featureLevel) << std::dec << '\n'
              << "GPU upload + synchronization: " << uploadMs << " ms\n"
              << "GPU kernel (timestamp query): "
              << (gpuMs >= 0.0 ? gpuMs : gpuWallMs) << " ms\n"
              << "GPU dispatch wall time: " << gpuWallMs << " ms\n"
              << "GPU readback: " << readbackMs << " ms\n";

    if (!validate_samples(cpuResult, gpuResult, n))
        return 1;

    if (gpuMs > 0.0) {
        const double gpuGflops =
            2.0 * static_cast<double>(n) * n * n / (gpuMs * 1.0e6);
        std::cout << "GPU kernel throughput: " << gpuGflops << " GFLOP/s\n";
        std::cout << "CPU/GPU kernel speedup: " << cpuMs / gpuMs << "x\n";
    }
    std::cout << "Note: CPU is single-threaded. GPU kernel timing excludes upload/readback.\n";
    return 0;
}
