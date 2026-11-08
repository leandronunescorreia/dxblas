# dxblas — copy CPU→GPU / GPU→CPU task list

Scope: only the copy paths (upload, readback, staging, batch readback) and the small fixes found around them.
Written for both a human and a coding agent. **Do one task at a time, in order.** Do not start a task before the previous one is checked off.

## Global rules (read first, apply to every task)

1. **Never use compute shaders.** No `ID3D11ComputeShader`, `Dispatch`, UAVs, structured buffers or `RWTexture`. Vertex shader + pixel shader + render-to-texture only.
2. **One task = one change set.** Touch only the files listed in the task. No drive-by refactors, no renames outside the task, no GEMM changes.
3. **Keep the public API source-compatible** unless the task says otherwise (`Context::copy`, `copy_to_cpu`, `copy_to_cpu_batch`, `sum`, `mul`, `gemm`, typed vector overloads, `GpuArray` path).
4. **Every task ends with the same loop:** build → run the verification listed in the task → fill the report → tick the box.
5. **If something in a task contradicts the code you see, stop and report it** instead of guessing.
6. **Do not weaken validation** (`E_INVALIDARG` checks) to make something pass.
7. Run with the D3D11 debug layer on (`D3D11_CREATE_DEVICE_DEBUG`) at least once per task; the output must have no new `D3D11 ERROR` / `WARNING` lines.

Build (Windows, from repo root):

```powershell
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Report template (paste at the end of each task)

```text
Task: Txx
Files changed: ...
What was done: ...
Verification run: <commands> -> <result>
Debug layer messages: none / <list>
Deviations from the task: none / <list>
Open questions: none / <list>
```

## Progress

- [x] T01 Restore best-adapter device selection
- [x] T02 Add a copy test harness (safety net)
- [x] T03 Central helpers: bytes-per-format and 1D layout
- [x] T04 Shared staging keyed by element size
- [x] T05 Reliable single-memcpy readback fast path
- [x] T06 Lazy arena (remove the 4096² pre-allocation)
- [x] T07 Batch readback with one staging texture and one Map
- [x] T08 Error-path, validation and leftover cleanups
- [x] T09 `gpuMem` lifetime semantics hardening
- [x] T10 GPU timestamps + optional async readback (Part A complete; optional Part B deferred)
- [ ] T11 Upload strategy benchmark (NVIDIA measured; Intel measurement pending)
- [ ] T12 Wrap-up: README, full run, hand-off (Debug + debug-layer + Intel verification pending)

---

## T01 — Restore best-adapter device selection

**Why.** The new `step_one_creating_the_device` calls `D3D11CreateDevice(nullptr, driver_type, ...)`, which takes the default adapter. On hybrid laptops that is often the integrated GPU. The old code used `create_best_d3d11_device` (largest dedicated VRAM, hardware only) and retried with feature level 11_0 only on `E_INVALIDARG`. Both behaviors were lost.

**Files:** `src/d3d11_backend.cpp` (and `include/dxblas/device_selection.hpp` only if a fix is needed).

**Steps**
1. Re-add `#include <dxblas/device_selection.hpp>` in `src/d3d11_backend.cpp`.
2. In `step_one_creating_the_device`: if `driver_type == D3D_DRIVER_TYPE_HARDWARE`, call `create_best_d3d11_device(device_flags, device, created_level, context)`.
3. Otherwise (WARP/reference/software) call `D3D11CreateDevice` with the 11_1/11_0 list. If it returns `E_INVALIDARG`, retry with `{ D3D_FEATURE_LEVEL_11_0 }`. If it still fails and the debug flag is set, clear the flag and repeat both attempts.
4. Keep `feature_level = created_level;` and the `arena_texture.initialize_adaptive(...)` call as they are (T06 changes that later).

**Done when**
- Console prints the adapter list and "Selected GPU: ..." for the hardware path.
- On a machine with two GPUs, the one with more dedicated VRAM is selected.
- `dxblas_gpumem_roundtrip` still passes.

**Do not:** change adapter ranking rules or add WARP fallback to the hardware path.

---

## T02 — Add a copy test harness (safety net)

**Why.** The next tasks change upload/readback internals. We need a test that proves data survives the roundtrip for awkward sizes before touching anything.

**Files:** new `examples/copy_tests.cpp`, `CMakeLists.txt` (new target `dxblas_copy_tests`, same pattern as the other examples).

**Steps**
1. Create `examples/copy_tests.cpp` using `dxblas::Context`, `gpuMem<T>`, `Context::copy`, `Context::copy_to_cpu`, `Context::copy_to_cpu_batch`.
2. Sizes to test for `float` and `std::int32_t` (and `std::uint8_t` / `std::int16_t` where `gpuMem` supports them): `1, 2, 5, 255, 4095, 4096, 4097, 16383, 16384, 16385, 32768, 32769, 1'000'003`.
   Note: elements are laid out in rows of `min(count, 16384)`, so 16385 and 32769 exercise the partial-last-row path.
3. For each size: fill with a deterministic pattern (index-based, e.g. `i * 2654435761u` cast to type, or `float(i) * 0.25f`), `copy`, read back, compare **bit-exact** (use `memcmp`). Print PASS/FAIL per case. Return non-zero on any failure.
4. Extra cases:
   - Upload the same `gpuMem` twice with different data (via `assign` then `copy`) and verify the second data set, not the first.
   - Upload size N, then a smaller size M < N into the same `gpuMem`, verify M elements.
   - Read back a prefix (`element_count < size()`), verify prefix only.
   - `copy_to_cpu_batch` with 2 and 3 sources of different sizes.
   - Mixed formats back to back: float, then int32, then uint8, then float again (this stresses staging reallocation in T04).
5. Add the target to `CMakeLists.txt`; add it to `run_all_samples.bat`.

**Done when**
- The harness builds and **passes on the current code** (this is the baseline).
- If a case fails on the current code, report it; do not fix it in this task.

**Do not:** modify library code.

---

## T03 — Central helpers: bytes-per-format and 1D layout

**Why.** The bytes-per-format `switch` exists in several places (`gpu_mem_bytes`, `create_gpu_buffer`, `binary_op`), and the "width = min(count, 16384); height = ceil(count / width)" math is repeated. Later tasks need one source of truth.

**Files:** `src/backend/arena_texture.hpp`, `src/backend/arena_texture.cpp`, `src/d3d11_backend.cpp`.

**Steps**
1. In `arena_texture.hpp` (namespace `dxblas::detail`) declare and in the `.cpp` define:
   ```cpp
   UINT bytes_per_element(DXGI_FORMAT f);            // 0 for unsupported
   struct Layout1D { UINT width, height; };
   HRESULT layout_1d(std::size_t count, Layout1D& out);  // width=min(count,16384), height=ceil; E_INVALIDARG if height>16384 or count==0
   ```
   Supported formats: `R8_UINT/SINT`, `R16_UINT/SINT/FLOAT`, `R32_UINT/SINT/FLOAT`.
2. Replace every local copy of the switch (`gpu_mem_bytes`, the switch in `create_gpu_buffer`, the one in `binary_op`) with `bytes_per_element`.
3. Replace the repeated width/height math in `prepare_gpu_mem`, `create_gpu_buffer`, `acquire_float` with `layout_1d`. **Keep the existing rule in `prepare_gpu_mem` that a caller-provided shape is preserved when `width*height >= count`.** `acquire_float` currently uses 4096 as max width; keep that value there for now (pass it as a parameter or keep a local constant) so behavior does not change in this task.
4. Remove the now-unused static functions.

**Done when**
- `grep -n "case DXGI_FORMAT_R8_UINT"` shows exactly one switch.
- `dxblas_copy_tests` and all other examples behave exactly as before.

**Do not:** change any behavior.

---

## T04 — Shared staging keyed by element size

**Why.** `ensure_readback_staging` reallocates whenever the exact format changes, so alternating float/int32 reads thrash. `CopySubresourceRegion` only needs compatible formats (same typeless family), so one staging per element size is enough. Also remove the leftover per-object staging.

**Files:** `src/backend/arena_texture.hpp/.cpp`, `include/dxblas/gpu_mem.hpp`.

**Steps**
1. Add a helper `DXGI_FORMAT staging_format_for(DXGI_FORMAT f)` returning `DXGI_FORMAT_R32_TYPELESS`, `DXGI_FORMAT_R16_TYPELESS` or `DXGI_FORMAT_R8_TYPELESS` by `bytes_per_element`.
2. Change `ensure_readback_staging` to compare/allocate with `staging_format_for(format)`. Keep the grow-only rule for width and height (max of old and requested).
3. **Verify with the debug layer** that `CopySubresourceRegion` from `R32_FLOAT`/`R32_UINT`/`R32_SINT` textures into the `R32_TYPELESS` staging produces no errors. If the runtime rejects it for any format, fall back to one staging per exact format stored in a small `std::unordered_map<DXGI_FORMAT, ...>` and report this.
4. Remove `ComPtr<ID3D11Texture2D> staging` from `gpuMemBase` (it is unused now). Fix any compile errors that reveal remaining users.
5. Make the sync-path `ArenaTexture::readback` use the shared staging too (copy only the used box from `output_`, then Map the shared staging). Then delete the member `readback_` and its creation in `allocate`.

**Done when**
- `dxblas_copy_tests` passes, including the "mixed formats back to back" case.
- With the debug layer: no format-compatibility errors.
- Instrument once (temporary counter or breakpoint) to confirm the staging texture is **not** reallocated when going float → int32 → float with the same sizes. Remove the instrumentation afterwards.

**Do not:** change upload code or batch readback yet (T07).

---

## T05 — Reliable single-memcpy readback fast path

**Why.** The fast path checks `RowPitch == copied_width * bytes`, but the shared staging is usually wider than the copied region, so the check almost never succeeds. The code is correct but falls back to the per-row loop.

**Files:** `src/backend/arena_texture.cpp`.

**Steps**
1. Create one internal function used by all single readbacks, e.g.
   `HRESULT copy_mapped_to_cpu(const D3D11_MAPPED_SUBRESOURCE& m, void* dst, std::size_t count, UINT row_elements, UINT rows, UINT bytes)`.
2. Rules inside it:
   - If `rows == 1`: one `memcpy(dst, m.pData, count * bytes)`.
   - Else if `m.RowPitch == row_elements * bytes`: one `memcpy` of `count * bytes`.
   - Else: loop rows, `min(remaining, row_elements)` elements per row, stride `m.RowPitch`. Return `E_FAIL` if `remaining != 0` at the end.
3. Use it from: sync `readback`, `readback_float`, `readback_gpu_mem`. (The batch path is replaced in T07.)
4. Make sure `Unmap` is always called after a successful `Map`, on every path, before returning.

**Done when**
- `dxblas_copy_tests` passes.
- A readback of 5 elements, and of 4096 floats, takes the single-`memcpy` branch (check by stepping once in the debugger, then no instrumentation left).

**Do not:** change what region is copied on the GPU.

---

## T06 — Lazy arena (remove the 4096² pre-allocation)

**Why.** `initialize_adaptive` allocates 4096×4096 textures at startup (input A, input B, output, plus a prewarmed slot). That is hundreds of MiB for a 5-element sum, and a format change reallocates all of it.

**Files:** `src/backend/arena_texture.hpp/.cpp`, `src/d3d11_backend.cpp`.

**Steps**
1. Change `initialize(device, ...)` so it only stores `device_`. Remove the `ensure(...)` call and the prewarmed `acquire_float` slot.
2. Make `initialize_adaptive(device)` a thin wrapper that calls `initialize(device)`, or delete it and update the call in `step_one_creating_the_device`. Remove the video-memory probing (`IDXGIAdapter3::QueryVideoMemoryInfo`) since it is no longer needed.
3. Rewrite `ensure(input_format, output_format, width, height)`:
   - If format matches and capacity fits: return `S_OK`.
   - Otherwise allocate exactly what is needed, rounded up to the next power of two per dimension (min 1), capped at `D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION`. On a format change, allocate for the new format (do not force 2048 minimum, do not inherit old capacity).
   - Allocate the new textures first and swap them in only if all creations succeed (the current `allocate` already does this; keep it).
4. Make sure `allocate` no longer creates the staging texture (done in T04). Only input A, input B and output remain.
5. Check `binary_op` in `src/d3d11_backend.cpp`: with `tile_width = 4096` it already asks `ensure` for the tile it needs. Confirm the first call with 5 elements creates textures of 8×1 (or similar), not 4096².

**Done when**
- Task Manager / GPU memory (or `IDXGIAdapter3::QueryVideoMemoryInfo` printed once) shows startup memory is a few MiB, not hundreds.
- `dxblas_copy_tests`, `dxblas_big_array`, `dxblas_typed_operations`, `dxblas_gpumem_sum` all still pass.
- A 4096×4096-element float `sum` still works (arena grows once).

**Do not:** change GPU-resident `GpuArray` slot pooling (only the prewarm is removed).

---

## T07 — Batch readback with one staging texture and one Map

**Why.** `readback_gpu_mem_batch` calls `CreateTexture2D(STAGING)` for every request on every call and maps each one separately. Use one cached staging texture, place each source at a different `DstY`, and map once.

**Files:** `src/backend/arena_texture.cpp`, `src/backend/arena_texture.hpp`.

**Steps**
1. Validate all requests first (null pointers, `count <= source->count`, `bytes_per_element != 0`, texture present). Skip `count == 0` entries. If nothing is left, return `S_OK`. **Do not queue any GPU copy until all requests validate.**
2. Group the remaining requests by `staging_format_for(storage_format)`. (Usually one group.)
3. For each group compute, per request: `copied_width = min(count, source->width)`, `rows = ceil(count / source->width)` (same rule as the single path), and a vertical offset `y` (running sum of rows). Required staging size: `width = max(copied_width)`, `height = sum(rows)`. If `height > 16384`, split the group into several passes.
4. Call `ensure_readback_staging(format, width, height)`.
5. For each request call `CopySubresourceRegion(staging, 0, 0, y, 0, source->texture, 0, &box)` with `box = {0,0,0,copied_width,rows,1}`.
6. `Map` the staging **once**, then for each request read from `m.pData + y * m.RowPitch` using the shared function from T05 (`copy_mapped_to_cpu`, passing a pointer offset), then `Unmap` once.
7. If a source appears twice in the request list, that must still work (each gets its own region).
8. Remove the per-request `PendingReadback` staging textures.

**Done when**
- `dxblas_copy_tests` batch cases pass (2 and 3 sources, different sizes, same source twice).
- `dxblas_gpumem_roundtrip` passes and shows a lower or equal "GPU → CPU" time than before.
- No `CreateTexture2D` happens in the batch path after the first call with the same sizes (confirm by a temporary counter, then remove it).

**Do not:** change `Context::copy_to_cpu_batch`'s public signature.

---

## T08 — Error-path, validation and leftover cleanups

**Why.** Small correctness issues found in review.

**Files:** `src/backend/arena_texture.cpp`, `src/d3d11_backend.cpp`, `include/dxblas/dxblas.hpp`.

**Steps**
1. Audit every `Map` call: each success path must `Unmap` exactly once. Fix any early return after a successful `Map`.
2. `sum_gpu_mem` is missing the `count == 0` guard that `mul_gpu_mem` has. Make both identical in validation (same checks, same order). Better: restore a single `binary_gpu_mem(BinaryOp, ...)` helper and have `sum_gpu_mem` / `mul_gpu_mem` call it, so there is one copy of the draw code.
3. In that helper, reject `out` aliasing `a` or `b` (same `texture` pointer) with `E_INVALIDARG`, because a texture cannot be an SRV and an RTV in the same draw.
4. Make sure every texture created by the pool or `create_gpu_buffer` or `prepare_gpu_mem*` sets `storage_format`. Add a debug assertion in `readback_gpu_mem` that `storage_format != DXGI_FORMAT_UNKNOWN`. Remove the `UNKNOWN → R32_FLOAT` fallback in `readback_float` once the assertion holds.
5. Remove dead code that nothing calls (verify with `grep` first, one name at a time): `copy_gpu`, `readback_copy_gpu`, `copy_gpu_buffers`, `sum_texture_gpu`, `texture_from_handle` if unused. If any is used by an example, keep it and report.
6. Remove the unused `packed_integer` parameter of `ArenaTexture::readback` if it is always `false`.

**Done when**
- All examples and `dxblas_copy_tests` pass.
- Debug layer: no warnings about SRV/RTV hazards in any example.
- `grep` shows no references to removed names.

**Do not:** change any public API signature in `dxblas.hpp` beyond what is stated here.

---

## T09 — `gpuMem` lifetime semantics hardening

**Why.** `gpuMem` now keeps only a pointer to the caller's data (good: no host copy). But the rules are easy to break: dangling pointers, accidental copies that share one GPU texture, and `copy` clearing the pointer so a second `copy` fails.

**Files:** `include/dxblas/dxblas.hpp`, `include/dxblas/gpu_mem.hpp`, `examples/copy_tests.cpp`, `README.md`.

**Steps**
1. Delete copy construction/assignment of `gpuMem<T>` (keep move). Moving must transfer texture/SRV/RTV and leave the source empty.
2. Delete the `std::vector<T>&&` constructor overload (already done) **and** the `assign(std::vector<T>&&)` path if it exists; add `assign(std::vector<T>&&) = delete`.
3. Document in a short comment above `gpuMem`: the pointer must stay valid until `Context::copy` returns; after a successful `copy` the pending pointer is cleared; to upload new data call `assign(ptr, n)` and then `copy` again (the texture is reused when shape and format match).
4. Add test cases to `copy_tests.cpp`: assign → copy → assign (new data, same size) → copy → verify second data; verify the texture object did not change between the two copies (compare `a.texture.Get()` before/after).
5. Add a "Memory ownership" paragraph to `README.md` with the rules above.

**Done when**
- Code that copies a `gpuMem` no longer compiles (check with a throwaway snippet, then remove it).
- New test cases pass.

**Do not:** reintroduce a host-side `std::vector` copy.

---

## T10 — GPU timestamps and optional async readback

**Why.** `Map(READ)` blocks until the draw and copy finish, so timings mix transfer and compute and the CPU cannot overlap work.

**Files:** `src/backend/arena_texture.*`, `src/d3d11_backend.cpp`, `include/dxblas/backend.hpp`, `include/dxblas/dxblas.hpp`, new `examples/copy_timing.cpp`, `CMakeLists.txt`.

**Steps (part A — timing, required)**
1. Add a small internal `GpuTimer` helper: one `D3D11_QUERY_TIMESTAMP_DISJOINT` plus two `D3D11_QUERY_TIMESTAMP` queries. `begin()` / `end()` wrap a region; `elapsed_ms()` polls with `GetData` (loop until ready) and returns `NaN` if the disjoint flag is set.
2. Create `examples/copy_timing.cpp` that, for sizes 1 KiB, 1 MiB, 16 MiB, 256 MiB (float), prints: CPU wall time of `copy` (upload), GPU time of the upload copy region if measurable, CPU wall time of readback, and effective GB/s for each. Run each case several times, report median.
3. Do not change library behavior for part A beyond exposing the timer as an internal helper.

**Steps (part B — async readback, optional, only if A shows readback wait dominates)**
1. Add `ReadbackTicket` (opaque, move-only): holds the staging region info plus an `ID3D11Query` of type `EVENT`.
2. `Context::copy_to_cpu_async(gpuMem<T>&, T* dst, n, ReadbackTicket&)`: queue `CopySubresourceRegion` into the shared staging at a **reserved** region, `End(event_query)`, return immediately.
3. `Context::poll(ticket)` → `S_OK`/`S_FALSE` via `GetData(event, ..., D3D11_ASYNC_GETDATA_DONOTFLUSH)`; `Context::wait(ticket)` does the `Map` and the copy to `dst`.
4. Because one staging region is shared, allow only one outstanding ticket at a time, or give each ticket its own region. Document the choice.

**Done when**
- `dxblas_copy_timing` runs and prints a table; paste it into the report.
- If part B is done: a test starts an async readback, does CPU work, then `wait`, and the data matches.

**Do not:** use `D3D11_MAP_FLAG_DO_NOT_WAIT` without handling `DXGI_ERROR_WAS_STILL_DRAWING`.

---

## T11 — Upload strategy benchmark (decision task)

**Why.** `UpdateSubresource` is fine on most drivers but may do an extra internal copy. An alternative is a persistent `STAGING` texture with `CPU_ACCESS_WRITE` filled via `Map`, then `CopySubresourceRegion` to the default texture. This task **measures first**; it only changes library code if the alternative clearly wins.

**Files:** new `examples/upload_bench.cpp`, `CMakeLists.txt`; library files only if the decision rule below says so.

**Steps**
1. In `examples/upload_bench.cpp` (using `dxblas::get_native_d3d11` for the device and context) implement two upload strategies on float data into a `D3D11_USAGE_DEFAULT` texture of 16384-wide rows:
   - A: `UpdateSubresource` with the full-rows box + tail box (same as the library).
   - B: persistent staging (`D3D11_USAGE_STAGING`, `CPU_ACCESS_WRITE`), `Map(WRITE)`, row `memcpy` (single `memcpy` when pitch is tight), `Unmap`, `CopySubresourceRegion`.
2. Sizes: 64 KiB, 1 MiB, 16 MiB, 256 MiB. For each: 20 warm iterations discarded, 100 measured, force completion each iteration (a tiny readback or an `EVENT` query), report median ms and GB/s.
3. Run on every GPU you have (integrated and discrete) and record results.
4. **Decision rule:** switch the library to B only if B is at least 20% faster at 16 MiB and 256 MiB on the primary GPU and not slower than 10% at 64 KiB. Otherwise keep A and just record the table in the README.
5. If switching: B needs a persistent upload staging texture with grow-only capacity keyed by element size (same idea as T04), and `upload_contiguous_rows` becomes the fallback for small sizes.

**Done when**
- Results table is in the report and in `README.md`.
- Library behavior unchanged unless the rule triggered; if it triggered, `dxblas_copy_tests` passes.

---

## T12 — Wrap-up: README, full run, hand-off

**Files:** `README.md`, `run_all_samples.bat`.

**Steps**
1. Update the README sections that describe the arena and readback so they match the new behavior (lazy arena, shared staging by element size, batch readback with one Map, ownership rules, timing numbers).
2. Run `run_all_samples.bat` in Debug and Release. All must pass.
3. Run once with the debug layer and confirm zero new warnings.
4. Produce a final report: for each task T01–T11 list files changed, any deviation, and the measured numbers from T10/T11.
5. Create the hand-off zip of the repo (without `build/`) and give it to the reviewer.

**Done when:** the reviewer has the zip plus the final report, and every box in "Progress" is ticked.


## T12 hand-off status (2026-10-12)

README and `run_all_samples.bat` were updated, benchmark tables and the T01–T11 file-change report were added, and a source hand-off ZIP was created without `build/`. The supplied Release logs show 9/9 samples after T09, 10/10 after T10, and 11/11 after T11. T12 remains unchecked because no Debug run, no D3D11 debug-layer warning audit, and no Intel integrated-GPU T11 run were supplied. Do not claim those verifications passed until they are performed on Windows.
