# DXBLAS Corrective Work Plan — Six Follow-up Tasks

**Review baseline:** `dxblas(8).zip`  
**Purpose:** Turn the source review into a bounded, testable follow-up plan. This document describes proposed work; it does not claim the changes have been implemented or runtime-verified.

## Source findings

The following findings are confirmed by inspection of the supplied source:

- `ArenaTexture::prepare_gpu_mem` preserves `memory.width` and `memory.height` whenever the old allocation can contain `count` and the format matches. There is no `shape_is_user_defined` flag in `gpuMemBase`.
- `ArenaTexture::prepare_gpu_mem_matrix` sets an explicit width and height for matrix output, but that intent is not separately recorded.
- `D3D11Backend::binary_gpu_mem` rejects inputs when either widths **or heights** differ, then checks the output against both dimensions and uses `out.height` as the viewport height.
- `ArenaTexture::ensure_readback_staging` grows width and height independently by taking their maxima. This can create a much larger allocation than either individual request needs.
- `ArenaTexture::ensure` returns early only when the current capacity contains both requested dimensions. If it must grow, the new dimensions are rounded from the current request rather than taking the maximum of the request and existing capacity.
- `examples/copy_tests.cpp` has reuse, move, and batch-readback coverage, but the reviewed test names do not cover `sum`/`mul` after a buffer shrinks, sync-arena shape/type alternation, `GpuArray` readback, output alias rejection, or duplicate-source batch readback.
- `examples/copy_timing.cpp` reports CPU time around `Context::copy`, GPU timestamp time around that call, and CPU readback time. It does not report a separate CPU upload-to-completion measurement. It uses one warm-up and five measured iterations.
- `prepare_gpu_mem`, `prepare_gpu_mem_matrix`, `create_gpu_buffer`, and `acquire_matrix_float` each contain texture/view creation sequences that can potentially share a helper.
- `upload_bench.cpp` already measures `UpdateSubresource` versus persistent staging at 64 KiB, 1 MiB, 16 MiB, and 256 MiB. The available NVIDIA results do not meet the current 20% improvement threshold at 16 MiB, so the production upload path should not be switched based on those results alone.

---

## Task 1 — Fix `gpuMem` shape reuse and arithmetic correctness

**Priority:** P0 — correctness  
**Likely files:**
- `include/dxblas/gpu_mem.hpp`
- `src/backend/arena_texture.hpp`
- `src/backend/arena_texture.cpp`
- `src/d3d11_backend.cpp`
- `examples/copy_tests.cpp`

### Problem

`prepare_gpu_mem` can retain an old texture shape after a smaller upload. A fresh buffer containing the same smaller element count can have a different width. `binary_gpu_mem` currently requires both input widths and heights to match, so `sum`/`mul` can return `E_INVALIDARG` even when both buffers contain enough elements.

The shrink/reuse test currently verifies readback but does not then run arithmetic on the reused buffer.

### Planned change

1. Add an explicit shape-intent field such as `shape_is_user_defined` to the internal resource metadata.
2. Mark matrix shapes set by `prepare_gpu_mem_matrix` as explicitly shaped; preserve those shapes for matrix/GEMM use.
3. For ordinary one-dimensional `gpuMem` buffers, compute the canonical `layout_1d(count)` shape. Reuse the texture only if its width matches the required width and its height is sufficient; otherwise recreate the texture/views.
4. In `binary_gpu_mem`, validate the compatible row width rather than requiring equal heights. Calculate the draw height from `ceil(count / width)` and use that height for the viewport. Ensure the output texture can hold the required region and has a compatible width before drawing.
5. Preserve `E_INVALIDARG` for invalid counts, formats, missing resources, incompatible row widths, and output/input texture aliasing.

### Tests / acceptance criteria

- Upload 100 elements to `a`; reassign 50 elements to `a`; upload 50 to fresh `b`; verify `sum(a,b,out)` and `mul(a,b,out)` succeed and return correct values.
- Cover the existing `32769 → 255` shrink case, followed by both arithmetic operations against a fresh 255-element buffer.
- Verify explicit matrix shapes remain stable and GEMM still passes.
- Verify `out.texture == a.texture` and `out.texture == b.texture` continue to return `E_INVALIDARG`.
- No SRV/RTV hazard warnings in the D3D11 debug layer.

---

## Task 2 — Bound shared readback staging growth

**Priority:** P1 — memory use  
**Likely files:**
- `src/backend/arena_texture.cpp`
- `src/backend/arena_texture.hpp`
- `examples/copy_tests.cpp`

### Problem

`ensure_readback_staging` independently retains the maximum width and maximum height for a staging format. A wide request followed by a tall, narrow request can therefore produce an allocation whose area is far larger than either request. For example, combining dimensions near `16384 × 1` and `2 × 16384` could lead to an allocation near `16384 × 16384` pixels (about 1 GiB at four bytes per pixel). The current code does not shrink that staging allocation.

### Planned change

1. Track the area of the current staging texture and the area of each request using overflow-safe size arithmetic.
2. Preserve the current staging allocation when it already contains the requested rectangle.
3. When growth is needed, avoid combining independent dimension maxima if the resulting area is more than approximately **2× the larger of the current allocation area and the requested area**. In that case, allocate for the request instead of the combined dimensions.
4. Keep the staging format compatibility rules and batch-readback layout correct; do not reuse a staging texture that cannot contain all rectangles in a batch.
5. Document the policy in a comment near `ensure_readback_staging`.

### Tests / acceptance criteria

- Exercise a wide-short read followed by a narrow-tall read, and the reverse order.
- Verify the resulting allocation dimensions/area remain within the documented policy.
- Run single and batch readback tests across all supported formats.
- Check allocation failures return the HRESULT without corrupting the previous valid staging resource.

---

## Task 3 — Make arena capacity grow-only and close readback/API test gaps

**Priority:** P1 — reuse stability and regression coverage  
**Likely files:**
- `src/backend/arena_texture.cpp`
- `src/backend/arena_texture.hpp`
- `examples/copy_tests.cpp`
- Possibly a focused existing sample if the `GpuArray` API needs separate coverage.

### Problem

`ArenaTexture::ensure` does not preserve the old capacity dimensions when a same-format request forces growth. Alternating a wide-short request and a narrow-tall request can repeatedly reallocate because the new allocation is sized from the latest request alone.

The reviewed tests also omit several public-path regressions.

### Planned change

1. When the input/output formats are unchanged and capacity must grow, choose each new capacity dimension as the maximum of the existing capacity and the rounded requested dimension.
2. Keep the existing lazy-allocation behavior; do not restore the removed large up-front allocation.
3. Add focused tests for:
   - sync API shape/type sequence: float count 5 → float count 4097 → uint8 → larger float;
   - `GpuArray` upload, arithmetic, and readback;
   - output aliasing (`out == a` and `out == b`) returning `E_INVALIDARG`;
   - batch readback listing the same source twice with distinct destinations;
   - existing differently sized and mixed-format batch cases.
4. Where possible, assert resource/capacity reuse directly rather than inferring it only from elapsed time.

### Tests / acceptance criteria

- The sync arena does not shrink or thrash for same-format alternating dimension requests.
- Format changes remain correct and do not reuse incompatible resources.
- Every new regression test checks both HRESULT and output data.
- All pre-existing copy tests and examples continue to pass.

---

## Task 4 — Separate upload submission, completion, and readback timing

**Priority:** P1 — measurement quality  
**Likely files:**
- `examples/copy_timing.cpp`
- Potentially `src/backend/arena_texture.hpp/.cpp` only if an existing internal timer interface needs a small extension.
- `README.md` for the resulting measurement definitions and results.

### Problem

`copy_timing.cpp` labels CPU time around `Context::copy` as upload CPU time. That interval measures the call/submission path and may not include completion of queued GPU work. The readback call can then pay for outstanding upload work, so the displayed upload and readback timings are not independent.

The benchmark currently uses one warm-up and five measured iterations, which is useful for a quick sample but weak for robust comparisons.

### Planned change

1. Keep the existing CPU submission metric, but label it explicitly as **upload submission CPU ms**.
2. Add **upload-to-completion CPU ms** by ending an EVENT query after the upload commands and waiting for that event before stopping the wall-clock timer. Handle query creation, `End`, `GetData`, and HRESULT failures explicitly.
3. Measure readback only after upload completion has been forced, so the readback interval is not also paying for the preceding upload.
4. For readback, add internal timing boundaries around:
   - `CopySubresourceRegion` submission;
   - `Map(READ)` wait;
   - mapped CPU copy (`memcpy`/row copies);
   - `Unmap`.
   Avoid claiming the CPU duration of `CopySubresourceRegion` equals GPU copy execution time; use GPU timestamps or clearly labelled CPU wall time.
5. Increase repetitions (for example, 20 warm-ups and at least 30 measured iterations) or document why a smaller count is retained. Continue reporting medians.
6. Keep the existing `GpuTimer` timestamp-disjoint handling and NaN behavior for invalid/disjoint results.

### Tests / acceptance criteria

- Output clearly distinguishes submission, completion, GPU timestamp, and readback phases.
- The upload completion query is waited on before the readback timing starts.
- Correctness validation runs outside the measured intervals.
- Results are repeatable enough to compare versions and are documented with GPU/driver and iteration details.

---

## Task 5 — Measure the upload crossover before considering a hybrid strategy

**Priority:** P2 — performance experiment; no production change by default  
**Likely files:**
- `examples/upload_bench.cpp`
- `README.md`
- Production upload files only if the new measurements satisfy an explicitly agreed decision rule.

### Problem

The NVIDIA measurements show:
- At 16 MiB, Strategy B is about 7.4% faster (15.584 ms vs 14.428 ms), below the existing 20% threshold.
- At 256 MiB, Strategy B is about 2.14× faster (499.479 ms vs 233.712 ms).
- At 64 KiB, Strategy B is faster (0.238 ms vs 0.173 ms).

This suggests a possible large-upload crossover, but the required 16 MiB threshold was not met. The available result is only for the NVIDIA GeForce 840M; integrated-GPU measurements have not been supplied.

### Planned change

1. Extend the benchmark with **32 MiB, 64 MiB, and 128 MiB**.
2. Keep the same warm-up count, 100 measured iterations, per-iteration GPU completion wait, and median reporting for both strategies.
3. Run on every available hardware GPU and record adapter identity, driver details where available, median milliseconds, GB/s, and A/B speedup.
4. Determine the crossover from measurements; do not infer it from a single size or one GPU.
5. Before a production switch, define a new explicit rule for a size-based hybrid. Require the hybrid to preserve the existing small-transfer behavior and demonstrate repeatable improvement above the threshold.
6. If a hybrid is justified, use persistent staging with grow-only capacity keyed by element size, retain `upload_contiguous_rows`/Strategy A for smaller transfers, and add a boundary test around the chosen threshold.

### Tests / acceptance criteria

- README contains the actual measured table, not estimated values.
- The selected threshold is supported by results on the primary GPU and checked on the integrated GPU.
- If no clear crossover is demonstrated, leave the library on Strategy A.
- If production behavior changes, all copy tests and examples pass.

---

## Task 6 — Consolidate texture creation and fix small code/documentation issues

**Priority:** P2 — maintainability  
**Likely files:**
- `src/backend/arena_texture.cpp`
- `src/backend/arena_texture.hpp`
- `README.md` if implementation documentation changes.

### Problem

Texture, SRV, and RTV creation is repeated in `prepare_gpu_mem`, `prepare_gpu_mem_matrix`, `create_gpu_buffer`, and `acquire_matrix_float`. The comment immediately above `layout_1d_with_max_width` describes `upload_contiguous_rows`, so it is attached to the wrong function.

### Planned change

1. Introduce a small internal `create_render_texture` helper that creates the texture and the required SRV/RTV views into temporary `ComPtr`s.
2. Commit the new resources to the target object only after every creation step succeeds, preserving the current failure-safe replacement behavior.
3. Keep format, dimensions, usage, bind flags, and CPU access flags explicit at the call site or in the helper parameters. Do not accidentally apply render-target views to staging textures.
4. Reuse the helper only where resource semantics match; retain specialized paths where bind flags or view requirements differ.
5. Move the `upload_contiguous_rows` comment to the function it describes and give `layout_1d_with_max_width` its own accurate comment.
6. Avoid unrelated renames or API changes.

### Tests / acceptance criteria

- All creation call sites preserve their original descriptors and storage-format metadata.
- Injected/observed creation failures do not replace a valid existing resource with a partially created resource.
- Static review shows fewer duplicated creation sequences and accurate comments.
- All examples pass, with no new D3D11 debug-layer warnings.

---

## Recommended implementation order

1. **Task 1 — shape correctness** (blocks trustworthy arithmetic/reuse testing).
2. **Task 2 — readback staging growth** (memory-safety/resource-footprint improvement).
3. **Task 3 — arena capacity and regression coverage**.
4. **Task 4 — timing breakdown** (needed for trustworthy performance diagnosis).
5. **Task 5 — upload crossover experiment** (measurement-only until evidence supports a change).
6. **Task 6 — helper extraction and small cleanup**.

Keep each task as a separate patch/commit. For each task, build in Windows Debug and Release, run the focused test plus `run_all_samples.bat`, and run with the D3D11 debug layer. Do not mark a task complete based on source inspection alone.

## Verification record

| Item | Current evidence from supplied material |
|---|---|
| Shape reuse bug | Confirmed by source inspection; regression repro not yet run against this archive |
| Readback staging over-allocation risk | Confirmed by independent max-width/max-height growth logic |
| Arena capacity thrashing risk | Confirmed by capacity check and request-only growth dimensions |
| Missing timing completion metric | Confirmed in `copy_timing.cpp`; CPU timing surrounds `Context::copy`, with no EVENT wait in that interval |
| Upload crossover | NVIDIA data supplied; 16 MiB threshold not met; 32/64/128 MiB and Intel results pending |
| Texture creation duplication/comment | Confirmed by source inspection |
| Windows Debug/Release and debug-layer validation for these proposed changes | Not performed as part of this review |
