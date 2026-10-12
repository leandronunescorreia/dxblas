# dxBLAS D3D11 — baby-step refactor

This version deliberately goes back to a very small first implementation.

The goals are:

- Direct3D 11 instead of Direct3D 9.
- No library-defined GPU state/enumeration layer.
- Native Direct3D types are used directly (`DXGI_FORMAT`, `D3D_DRIVER_TYPE`,
  `D3D11_CREATE_DEVICE_*`, `D3D_FEATURE_LEVEL`, `HRESULT`, etc.).
- A backend interface separates the public API from the graphics API.
- Each backend implementation encapsulates platform-specific types using the PImpl idiom.
- Shaders are cached per operation and format so compilation only occurs once on demand.
- Offscreen `Texture2D` render targets instead of compute buffers.
- The first example is intentionally small and debug-friendly.

## Project Structure

```text
dxblas/
├── CMakeLists.txt                  # Build configuration (C++17, D3D11, DXGI, D3DCompiler)
├── cmake/                          # CMake modules & future backend configuration
│   └── README.md
├── include/
│   └── dxblas/
│       ├── backend.hpp             # Abstract IBackend interface (initialize, sum, mul)
│       └── dxblas.hpp              # Public Context API and native interop helpers
├── src/
│   ├── dxblas.cpp                  # Context implementation (owns std::unique_ptr<IBackend>)
│   ├── d3d11_backend.cpp           # Direct3D 11 classic graphics pipeline implementation (PImpl)
│   └── backend/
│       └── d3d11_backend.hpp       # Internal D3D11Backend declaration
└── examples/
    ├── fist_step.cpp               # Introductory sample demonstrating sum, mul, and native handles
    └── big_array.cpp               # Sample testing 512-element arrays through an offscreen pixel pass
```

### Component Breakdown

- **`include/dxblas/backend.hpp`**: Defines the abstract `IBackend` interface with `initialize()`, `sum()`, and `mul()`. Decouples linear algebra algorithms from specific graphics APIs.
- **`include/dxblas/dxblas.hpp`**: Public library facade. Declares `dxblas::Context`, which manages backend lifetime via `std::unique_ptr<IBackend>`, and provides native handle queries (`get_native_d3d11`).
- **`src/dxblas.cpp`**: Implements `Context`, forwarding BLAS operations directly to the active backend.
- **`src/backend/d3d11_backend.hpp`**: Internal declaration of `D3D11Backend`, utilizing PImpl to prevent leaking `<d3d11.h>` into public headers.
- **`src/d3d11_backend.cpp`**: Full Direct3D 11 classic graphics pipeline. Implements device creation with automatic debug-layer fallback, Texture2D allocation/upload, runtime HLSL compilation with shader caching, full-screen-triangle pixel rendering, and staging texture readback.
- **`examples/fist_step.cpp`**: Educational sample verifying element-wise vector addition (`sum`) and multiplication (`mul`) on `uint32` and `float` data.
- **`examples/big_array.cpp`**: Benchmark/verification sample computing `sum` and `mul` on 512 elements through offscreen Texture2D pixel passes with CPU validation.

## Baby steps

The D3D11 backend is intentionally organized as:

```text
step_one_creating_the_device()
step_two_creating_input_texture()
step_three_creating_output_texture()
step_four_compiling_shaders()      <-- cached per (op, format)
step_five_rendering_the_shader()
step_six_reading_back_texture()
```

The current public operations wire these steps together for element-wise
1D buffer addition (`sum`) and multiplication (`mul`).

Supported formats in this first sample:

```text
DXGI_FORMAT_R8_UINT
DXGI_FORMAT_R16_UINT
DXGI_FORMAT_R32_UINT
DXGI_FORMAT_R32_FLOAT
```

There is intentionally no custom `dxblas_data_type`, `dxblas_status`,
`dxblas_shader_model`, sampler state enum, or graphics-state enum.

The format passed to the API is the native `DXGI_FORMAT`, and failures are
reported as the native Windows `HRESULT`.

## Architecture

```text
dxblas::Context (Facade, holds std::unique_ptr<IBackend>)
        |
        v
dxblas::IBackend (Interface)
        |
        +-- dxblas::detail::D3D11Backend
        |    +-- Impl (PImpl encapsulating <d3d11.h>)
        |         +-- ID3D11Device
        |         +-- ID3D11DeviceContext
        |         +-- shader_cache (std::unordered_map)
        |         +-- Texture2D inputs / render target / staging texture
        |
        +-- (Future) OpenGLBackend
        +-- (Future) VulkanBackend
        +-- (Future) MetalBackend
```

Future backends implement the same `IBackend` interface without changing the public
algorithm API:

```text
IBackend
  +-- D3D11Backend (PImpl)
  +-- D3D12Backend (PImpl)
  +-- OpenGLBackend (PImpl)
  +-- VulkanBackend (PImpl)
  +-- MetalBackend (PImpl)
```

## DX11 backend from the compute pipeline to a classic graphics pipeline
```text
C++ input arrays
      │
      ▼
Texture2D A ─────┐
                 ├──► Pixel Shader ───► offscreen Texture2D
Texture2D B ─────┘       ▲
                         │
                 Full-screen triangle
                         │
                         ▼
                  staging Texture2D
                         │
                         ▼
                    C++ output
```

## Build & Run

```powershell
Remove-Item -Recurse -Force build
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run samples
.\build\dxblas_first_step.exe
.\build\dxblas_big_array.exe
```

## Memory ownership (`gpuMem<T>`)

`gpuMem<T>` is a non-owning upload descriptor: the pointer passed to its constructor or `assign(ptr, n)` must remain valid until `Context::copy` returns. A successful `Context::copy` clears the pending host pointer, so the descriptor does not retain or copy the CPU payload. To upload new data, call `assign(ptr, n)` and then `copy` again. The existing GPU texture is reused when its shape and storage format still match. `gpuMem<T>` is move-only because copying it would create multiple objects that appear to own the same GPU resources; moving transfers the texture/SRV/RTV and leaves the source empty. Do not pass a temporary vector or otherwise let the pointed-to storage expire before `Context::copy` returns.

## D3D11 pixel pipeline texture arena

The D3D11 backend uses a persistent `ArenaTexture` owned by the backend context.
The arena keeps reusable input SRVs, an output RTV texture, and a CPU-readable
staging texture alive across operations. It grows when a requested operation
does not fit and only reallocates when the format or capacity must change.

Each operation therefore follows the hot path:

1. `UpdateSubresource` the two input arrays into the persistent arena textures.
2. Draw the fullscreen triangle and execute the pixel shader.
3. `CopyResource` the persistent output texture into its staging texture.
4. Map/read the staging texture back into the caller's C++ output buffer.

No `ID3D11ComputeShader` or `Dispatch` is used.



## GPU-resident arrays (pixel-shader backend)

The context also exposes a GPU-resident path for `DXGI_FORMAT_R32_FLOAT`:

```cpp
dxblas::GpuArray a_gpu, b_gpu, tmp_gpu, result_gpu;
context.upload_gpu(DXGI_FORMAT_R32_FLOAT, a.data(), count, a_gpu);
context.upload_gpu(DXGI_FORMAT_R32_FLOAT, b.data(), count, b_gpu);
context.sum_gpu(a_gpu, b_gpu, tmp_gpu);
context.mul_gpu(tmp_gpu, b_gpu, result_gpu);
context.readback_gpu(result_gpu, output.data(), count);
```

Uploads happen once, intermediate arrays remain in arena-managed textures, and
readback is explicit. GPU-resident texture slots are reused after the last
`GpuArray` handle releases them. The current GPU-resident API intentionally
supports float32 only; the synchronous API retains the existing integer paths.
The backend uses the classic D3D11 vertex/pixel-shader pipeline and never uses
compute shaders or `Dispatch()`.


## Function-level typed operations

`Context` is not templated. The C++17 typed convenience overloads select the element format per call, so one context can mix supported element types:

```cpp
std::vector<float> af{1.f, 2.f}, bf{3.f, 4.f}, cf;
std::vector<std::uint32_t> ai{1, 2}, bi{3, 4}, ci;

context.sum(af, bf, cf); // R32_FLOAT shader variant
context.sum(ai, bi, ci); // R32_UINT shader variant
```

The typed overloads map C++ element types through `dxblas::dxgi_format<T>`. The runtime-format overloads remain available for dynamically typed buffers. The shader manager specializes and caches concrete HLSL programs at runtime. Supported typed C++ mappings are `float`, `uint8_t`, `uint16_t`, `uint32_t`, `int8_t`, `int16_t`, and `int32_t`; corresponding DXGI formats are used. FP16 storage (`DXGI_FORMAT_R16_FLOAT`) is supported by the runtime-format API, but a standard C++17 `float16` type is not provided.

Integer variants now return native `uint`/`int` shader outputs to compatible integer render targets instead of packing integer bytes into an UNORM texture. The GPU-resident `GpuArray` path remains float-only in this version; typed `std::vector` operations use the synchronous upload/execute/readback path.

## GEMM matrix multiplication sample

`dxblas_gemm` adds row-major FP32 GEMM using the existing D3D11 texture arena and a pixel shader. It computes `C[M,N] = A[M,K] * B[K,N]` using render-to-texture; it does **not** use Direct3D compute shaders, UAVs, or structured buffers.

Build on Windows with the Windows SDK and CMake, then run:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --target dxblas_gemm
.\build\Release\dxblas_gemm.exe
```

The sample defaults to a 4096 x 4096 square multiplication. An optional argument selects a smaller square size for smoke tests, for example `dxblas_gemm.exe 256`. It reports a multi-threaded CPU reference time and an end-to-end GPU time including texture uploads and result readback, and validates sampled output elements. The first shader compilation is warmed up before timing.

**Performance note:** this first implementation is a straightforward pixel-shader GEMM, with one K-length dot product per output pixel and row-batched draws to reduce individual draw duration. It is a correctness/reference implementation rather than a tiled, compute-optimized GEMM. At 4096 x 4096 it can be slow and may be affected by Windows GPU timeout (TDR), depending on the GPU and driver. Run a smaller size first.

### GEMM shader loading

`dxblas_gemm` can run regardless of its process working directory: the GEMM HLSL has an embedded fallback. CMake also copies editable shaders beside the `dxblas_gemm` executable for development. If a shader is edited on disk, ensure the executable's working directory allows the runtime loader to find the corresponding `shaders/gemm.hlsl`; otherwise the embedded built-in shader is used.


## Upload strategy benchmark (T11)

`dxblas_upload_bench` compares the current `UpdateSubresource` path (A) with a
persistent CPU-write staging texture followed by `CopySubresourceRegion` (B).
Both strategies upload `DXGI_FORMAT_R32_FLOAT` textures with 16,384 elements per
row. Every iteration is followed by a D3D11 event-query wait so the reported
median latency includes GPU completion, rather than measuring command submission
alone. The staging texture is reused across warm-up and measured iterations for
each size. The benchmark runs 20 warm-up iterations and records the median of 100
measured iterations for 64 KiB, 1 MiB, 16 MiB, and 256 MiB.

Build and run on Windows:

```bat
cmake --build build --config Release --target dxblas_upload_bench
build\Release\dxblas_upload_bench.exe --list-adapters
build\Release\dxblas_upload_bench.exe
```

The default run uses DXBLAS's normal best-hardware-adapter selection. To measure
another hardware adapter, pass its DXGI enumeration index (for example, index 0):

```bat
build\Release\dxblas_upload_bench.exe --adapter-index 0
```

Record one table per GPU below. **No results have been recorded yet**; this source
archive was not run on a Windows GPU during implementation. Do not switch the
library's upload strategy until measurements on the primary GPU satisfy the
project's decision rule: B must be at least 20% faster at both 16 MiB and 256 MiB,
and no more than 10% slower at 64 KiB.

| GPU / adapter | Size | A: UpdateSubresource median ms | A GB/s | B: staging median ms | B GB/s | A/B speed ratio |
|---|---:|---:|---:|---:|---:|---:|
| Pending Windows run | 64 KiB | — | — | — | — | — |
| Pending Windows run | 1 MiB | — | — | — | — | — |
| Pending Windows run | 16 MiB | — | — | — | — | — |
| Pending Windows run | 256 MiB | — | — | — | — | — |
