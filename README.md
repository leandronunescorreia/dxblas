# dxBLAS D3D11 — baby-step refactor

This version deliberately goes back to a very small first implementation.

The goals are:

- Direct3D 11 instead of Direct3D 9.
- No library-defined GPU state/enumeration layer.
- Native Direct3D types are used directly (`DXGI_FORMAT`, `D3D_DRIVER_TYPE`,
  `D3D11_CREATE_DEVICE_*`, `D3D_FEATURE_LEVEL`, `HRESULT`, etc.).
- A backend interface separates the public API from the graphics API.
- Each backend implementation encapsulates platform-specific types using the PImpl idiom.
- Shaders are cached per-format so compilation only occurs once on demand.
- Native 1D GPU buffers (`ID3D11Buffer`) instead of 2D textures.
- The first example is intentionally small and debug-friendly.

## Baby steps

The D3D11 backend is intentionally organized as:

```text
step_one_creating_the_device()
step_two_creating_the_buffer()
step_three_uploading_the_buffer()
step_four_compiling_the_shader()   <-- cached per-format
step_five_dispatching_the_shader()
step_six_reading_back_the_buffer()
```

The current public operation wires these steps together for a simple element-wise
1D buffer addition (`Buffer<T>` + `Buffer<T>` -> `RWBuffer<T>`).

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
        |         +-- buffers
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

## build

```powershell
Remove-Item -Recurse -Force build
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Debug
cmake --build build
.\build\dxblas_step_by_step.exe
```