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

## D3D11 pixel pipeline texture arena and readback

The backend owns a lazy `ArenaTexture`: it does not reserve a maximum-sized
4096×4096 arena at initialization. Textures are allocated on first use and
recreated only when the requested dimensions or format require more capacity.
The arena keeps reusable input SRVs and an output RTV texture for the classic
vertex/pixel-shader/render-to-texture pipeline.

Uploads continue to use `UpdateSubresource`, including a full-rows box and a
tail box when the input does not fill the final row. T11 measured an alternative
persistent CPU-write staging upload, but the measured 16 MiB result did not meet
the project's 20% improvement threshold, so the production upload path was not
changed.

Readback staging is shared and cached by element-size-compatible format rather
than recreated for each request. The staging capacity grows as needed. The
single-readback path uses a contiguous `memcpy` when the mapped pitch is tight;
otherwise it copies row by row. `copy_to_cpu_batch` queues all requested regions
into one shared staging texture and maps it once for the batch. Successful
`Map` calls are paired with `Unmap` before returning. GPU memory resources retain
their `storage_format` so readback does not need to guess a fallback format.

No `ID3D11ComputeShader`, UAV, structured buffer, or `Dispatch` is used.



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


## Copy timing benchmark (T10)

`dxblas_copy_timing` measures CPU time spent in `Context::copy`, a GPU timestamp
region around that call, and CPU wall time for readback. The values below were
reported by the Release run on the NVIDIA GeForce 840M. Each size used one warm-up
and five measured iterations; the displayed statistic is the median. Upload
throughput counts both input buffers, and all throughput figures use decimal GB/s.
The CPU upload number is submission/call time, while the GPU timestamp is a
separate GPU timeline measurement; they are not additive measurements of the
same interval.

| Size | Upload CPU ms | Upload CPU GB/s | Upload GPU ms | Upload GPU GB/s | Readback CPU ms | Readback GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 KiB | 0.024 | 0.086 | 0.004 | 0.508 | 0.181 | 0.006 |
| 1 MiB | 0.664 | 3.156 | 1.345 | 1.560 | 1.060 | 0.989 |
| 16 MiB | 6.475 | 5.182 | 21.394 | 1.568 | 13.898 | 1.207 |
| 256 MiB | 292.345 | 1.836 | 442.040 | 1.215 | 206.963 | 1.297 |

The large readback times justify investigating overlap in a future change, but
T10's optional asynchronous readback ticket API was not implemented. The library
remains synchronous and its behavior was not changed by the timer helper.

## Upload strategy benchmark (T11)

`dxblas_upload_bench` compares Strategy A (`UpdateSubresource`, matching the
production upload path) with Strategy B (a persistent `D3D11_USAGE_STAGING`
texture mapped for CPU write, followed by `CopySubresourceRegion`). The results
below are from the NVIDIA GeForce 840M only. The run used 20 warm-up iterations
and 100 measured iterations per size, forced GPU completion after each iteration
with an EVENT query, and reported median latency. Throughput is decimal GB/s.

| Size | A median ms | A GB/s | B median ms | B GB/s | A/B speed ratio |
|---:|---:|---:|---:|---:|---:|
| 64 KiB | 0.238 | 0.275 | 0.173 | 0.379 | 1.378× |
| 1 MiB | 0.939 | 1.117 | 0.932 | 1.125 | 1.007× |
| 16 MiB | 15.584 | 1.077 | 14.428 | 1.163 | 1.080× |
| 256 MiB | 499.479 | 0.537 | 233.712 | 1.149 | 2.137× |

**Decision:** keep Strategy A in the library. Strategy B is substantially faster
at 256 MiB and faster at 64 KiB, but only about 7.4% faster at 16 MiB, below the
required 20% improvement. The integrated Intel adapter has not yet been measured,
so the all-hardware-adapters portion of T11 remains outstanding.

Build and run the benchmarks on Windows:

```bat
cmake --build build --config Release --target dxblas_copy_timing dxblas_upload_bench
build\Release\dxblas_copy_timing.exe
build\Release\dxblas_upload_bench.exe --list-adapters
build\Release\dxblas_upload_bench.exe --adapter-index 0
```

Use the adapter indices reported by `--list-adapters` to run the upload benchmark
on each hardware adapter; skip software adapters. Do not infer Intel results from
the NVIDIA measurements.

## T01–T11 implementation and verification report

The following file lists describe the task patches in this hand-off. They identify
files changed for each task; where a task reuses earlier changes, the list is the
set changed by that task's patch, not the full cumulative diff.

| Task | Files changed | Result / deviation |
|---|---|---|
| T01 | `src/d3d11_backend.cpp` | Restored best hardware-adapter selection. |
| T02 | `examples/copy_tests.cpp`, `CMakeLists.txt`, `run_all_samples.bat` | Added copy test harness. |
| T03 | `src/backend/arena_texture.hpp`, `src/backend/arena_texture.cpp`, `src/d3d11_backend.cpp` | Centralized format-size and 1D-layout helpers. |
| T04 | `src/backend/arena_texture.hpp`, `src/backend/arena_texture.cpp`, `include/dxblas/gpu_mem.hpp` | Added shared staging keyed by element size. |
| T05 | `src/backend/arena_texture.cpp` | Added tight-pitch single-`memcpy` readback path. |
| T06 | `src/backend/arena_texture.hpp`, `src/backend/arena_texture.cpp` | Made arena allocation lazy. |
| T07 | `src/backend/arena_texture.cpp`, `src/backend/arena_texture.hpp` | Added batched readback using one staging texture and one `Map`. |
| T08 | `src/backend/arena_texture.cpp`, `src/backend/arena_texture.hpp`, `src/backend/d3d11_backend.hpp`, `src/d3d11_backend.cpp`, `include/dxblas/backend.hpp` | Consolidated arithmetic validation, rejected output/input texture aliasing, enforced format invariants, and removed unused wrappers. `texture_from_handle` was retained because it remains used by the GPU-array path. |
| T09 | `include/dxblas/dxblas.hpp`, `examples/copy_tests.cpp`, `README.md` | Made `gpuMem<T>` move-only, rejected temporary-vector assignment, tested resource moves and same-size reuploads, and documented pointer lifetime. |
| T10 | `CMakeLists.txt`, `src/backend/arena_texture.hpp`, `src/backend/arena_texture.cpp`, `examples/copy_timing.cpp` | Added timestamp helper and timing sample. Optional asynchronous readback was deferred; the synchronous API remains unchanged. |
| T11 | `CMakeLists.txt`, `README.md`, `examples/upload_bench.cpp` | Added A/B upload benchmark. No production upload change because the 16 MiB threshold was not met. Intel measurements remain outstanding. |

### Runtime evidence provided

- T09: Release `run_all_samples.bat` reported 9 passed and 0 failed; the copy tests included second-data reupload, same-size texture reuse, and move-resource checks.
- T10: Release run reported 10 passed and 0 failed, including `dxblas_copy_timing.exe`; measurements are recorded above.
- T11: Release run reported 11 passed and 0 failed, including `dxblas_upload_bench.exe`; NVIDIA measurements and the decision are recorded above.
- Not evidenced in the supplied logs: a complete Debug run, D3D11 debug-layer run with zero new warnings, or Intel upload benchmark results. These checks must not be treated as passed until run on Windows.
