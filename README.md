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
