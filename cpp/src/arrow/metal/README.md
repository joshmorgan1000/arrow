# arrow_metal

Metal-coherent buffer / device / memory-manager backend for Apache Arrow.

Apple Silicon has unified memory: CPU and GPU share physical RAM. A Metal
buffer allocated with `MTLResourceStorageModeShared` is simultaneously
CPU-readable (via `.contents`) and GPU-readable (as `MTLBuffer`) with no
copy. `arrow_metal` exposes that property through standard Arrow types
(`arrow::Buffer`, `arrow::Device`, `arrow::MemoryManager`,
`arrow::MemoryPool`) so existing Arrow code can work directly with
GPU-addressable memory.

## Build

```
cmake -DARROW_METAL=ON ..
cmake --build . --target arrow_metal_shared
ctest -R metal
```

`ARROW_METAL` defaults to ON on Apple platforms (`if(APPLE)` in
DefineOptions.cmake) and FATAL-errors elsewhere. There is no software
fallback — Metal is Apple-only.

## Public API

```cpp
#include "arrow/metal/metal_api.h"   // pure C++; do NOT need Objective-C++
```

Five user-facing classes:

| Class                   | Role                                                           |
|-------------------------|----------------------------------------------------------------|
| `MetalDevice`           | Subclass of `arrow::Device`; wraps `id<MTLDevice>`             |
| `MetalDeviceManager`    | Process-wide singleton enumerating Metal devices               |
| `MetalMemoryManager`    | Subclass of `arrow::MemoryManager`; allocates shared MTLBuffer |
| `MetalBuffer`           | Subclass of `arrow::Buffer`; owns / wraps an `id<MTLBuffer>`   |
| `MetalMemoryPool`       | Subclass of `arrow::MemoryPool`; drop-in for `default_memory_pool` |

A MetalBuffer reports `is_cpu() == true` AND `device_type() == kMETAL` —
the unified-memory contract. CPU↔Metal "copies" become memcpy or
zero-copy views, never PCIe transfers.

## Ownership contract for `Wrap()`

`MetalBuffer::Wrap(mm, host_ptr, size, release_fn)` adopts external host
memory via `[device newBufferWithBytesNoCopy:options:deallocator:]`.

- The MetalBuffer owns the MTLBuffer.
- The MTLBuffer's deallocator block owns the host memory.
- `release_fn` runs exactly once when the last MTLBuffer retain drops.
- Do NOT also pin the source through Arrow's `parent_` — that creates
  two paths to free the same memory.

`MetalMemoryManager::ViewBufferFrom` does the inverse: a no-op
deallocator block plus an Arrow `parent_` pointer, so the source Arrow
Buffer remains the sole owner.

## Tests

```
ctest -R metal --output-on-failure
```

Tests skip gracefully via `GTEST_SKIP()` if no Metal device is present
(headless Intel macOS, virtualized hosts).

The `GPUSeesCPUWrites` test compiles a tiny Metal Shading Language
kernel inline and dispatches it against an Arrow buffer to prove
end-to-end CPU↔GPU coherence — it is the load-bearing test for the
"Metal-coherent" claim.

## Benchmarks

Representative numbers on Apple M-series (release build, single thread):

| Benchmark                          | 4 KiB     | 256 MiB   |
|------------------------------------|-----------|-----------|
| `BM_AllocateBuffer`                | ~4 µs     | ~12 µs    |
| `BM_CPUtoMetalCopy`                | ~5 µs     | ~13 ms    |
| `BM_CPUtoMetalView` (zero-copy)    | **~9 ns** | **~10 µs** |
| `BM_SyncEventRecordWait`           | ~14 µs    | n/a       |
| `BM_MetalPoolAllocFree`            | ~1.3 µs   | ~2.5 µs   |

The headline result: `BM_CPUtoMetalView` is ~560-1250x faster than
`BM_CPUtoMetalCopy` because the unified-memory architecture lets us
skip the data move entirely.

## Design constraint: pure-C++ headers

Public headers under `arrow/metal/` MUST NOT include `<Metal/Metal.h>`.
Apple objects are stored as `void*` (with the underlying ObjC type
documented in a comment). The bridge happens only in `.mm` files via
`metal_internal.hh` — see `project_metal_pimpl.md` for rationale.

This means downstream consumers can `#include "arrow/metal/metal_api.h"`
from any plain `.cpp` file without forcing Objective-C++ on their build.
