# arrow_vulkan

Host-coherent Vulkan buffer / device / memory-manager backend for Apache
Arrow. Targets integrated GPUs and CPU rasterizers where
`HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` memory exists. Allocations
from `arrow_vulkan` are simultaneously CPU- and GPU-addressable with no
staging copy. Discrete GPUs without ReBAR are out of scope and rejected
at device-selection time with a clear error.

## Build

```
cmake -DARROW_VULKAN=ON ..
cmake --build . --target arrow_vulkan_shared
ctest -R arrow-vulkan --output-on-failure
```

`ARROW_VULKAN` defaults to OFF; opt in explicitly. Requires a Vulkan
1.2+ loader (the `Vulkan::Vulkan` CMake target via `find_package(Vulkan)`).

Verified runtime targets:

| Platform | Driver | Status |
|---|---|---|
| Apple Silicon (M-series) | MoltenVK 1.3.0 | Allocate / Copy / Sync / Pool work; Wrap path is no-op (MoltenVK advertises but rejects `VK_EXT_external_memory_host` — our honest probe at device init catches it and Wrap returns `Status::NotImplemented`) |
| Linux + AMD APU | RADV (Mesa 25.x) | Full functionality including zero-copy Wrap |
| Linux + Intel iGPU | ANV | Full functionality (untested in this fork) |
| Linux + llvmpipe | Mesa software | Full functionality (slow; useful for headless CI) |

## Public API

```cpp
#include "arrow/vulkan/vulkan_api.h"   // pure C++; do NOT need <vulkan/vulkan.h>
```

Five user-facing classes:

| Class                   | Role                                                              |
|-------------------------|-------------------------------------------------------------------|
| `VulkanDevice`          | Subclass of `arrow::Device`; wraps `(VkPhysicalDevice, VkDevice, VkQueue)` |
| `VulkanDeviceManager`   | Process-wide singleton enumerating + filtering Vulkan devices     |
| `VulkanMemoryManager`   | Subclass of `arrow::MemoryManager`; allocates host-coherent VkBuffer |
| `VulkanBuffer`          | Subclass of `arrow::Buffer`; owns / wraps a `(VkBuffer, VkDeviceMemory)` |
| `VulkanMemoryPool`      | Subclass of `arrow::MemoryPool`; drop-in for `default_memory_pool` |

A VulkanBuffer reports `is_cpu() == true` AND `device_type() == kVULKAN`
on integrated devices — the unified-memory contract. CPU↔Vulkan "copies"
become memcpy or zero-copy views, never PCIe transfers.

## Device picker policy

`VulkanDeviceManager` enumerates physical devices and applies:

1. **Prefer `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`** with a memory type
   matching `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` (the AMD APU /
   Intel iGPU / Mali / MoltenVK path).
2. **Fallback to `VK_PHYSICAL_DEVICE_TYPE_CPU`** (llvmpipe) — useful for
   CI runners without a real GPU; behaves identically since llvmpipe
   exposes unified memory.
3. **Refuse `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`** with
   `Status::NotImplemented`. Matches the project's "zero-copy or refuse"
   promise.

If no acceptable device is found, `VulkanDevice::Default()` returns
`Status::NotImplemented("no Vulkan device with HOST_COHERENT |
DEVICE_LOCAL memory found")`. Tests skip gracefully via `GTEST_SKIP()`.

## Ownership contract for `Wrap()`

`VulkanBuffer::Wrap(mm, host_ptr, size, release_fn, user_data)` adopts
external host memory via `vkAllocateMemory` with
`VkImportMemoryHostPointerInfoEXT`.

- The VulkanBuffer owns the `(VkBuffer, VkDeviceMemory)` pair.
- The caller-supplied `release_fn` owns the host memory.
- Destruction order: `vkUnmapMemory` (for non-imported allocations),
  `vkDestroyBuffer`, `vkFreeMemory`, then `release_fn(host_ptr, size,
  user_data)` exactly once.
- Do NOT also pin the source via Arrow's `parent_` — that creates two
  paths to free the same memory. Use `Wrap` (deallocator owns) OR
  `ViewBufferFrom` (Arrow `parent_` owns), never both.

`host_ptr` and `size` MUST be aligned to
`VulkanDevice::imported_host_pointer_alignment()` (typically the system
page size on real drivers; varies on MoltenVK).

## Tests

```
ctest -R arrow-vulkan --output-on-failure
```

16 tests, structured as a parallel of the Metal suite. The
`GPUSeesCPUWrites` test compiles a tiny GLSL kernel offline (the
SPIR-V bytecode is embedded inline in `vulkan_test.cc`) and dispatches
it against an Arrow buffer to prove end-to-end CPU↔GPU coherence — it is
the load-bearing test.

`NestedSliceGPUDispatch` verifies slice-of-slice GPU dispatch via
`root_vk_buffer()` + `offset()`. `SyncEventConcurrentRecords` fires N
threads × M `Record()` calls and asserts the atomic counter is exact.

Tests skip gracefully on hosts without a Vulkan device or without
`VK_EXT_external_memory_host` working at runtime.

## Benchmarks

Representative numbers on Apple M4 Max (MoltenVK, single thread, release):

| Benchmark                          | 4 KiB     | 256 MiB   |
|------------------------------------|-----------|-----------|
| `BM_AllocateBuffer`                | ~3.7 µs   | ~18 µs    |
| `BM_CPUtoVulkanCopy`               | ~5 µs     | ~13 ms    |
| `BM_CPUtoVulkanView` (zero-copy)   | (skipped on MoltenVK) | — |
| `BM_SyncEventRecordWait`           | ~20 µs    | n/a       |
| `BM_VulkanPoolAllocFree`           | ~3.7 µs   | ~9.4 µs   |

Numbers from Ryzen 890M (RADV, where `BM_CPUtoVulkanView` actually runs)
will be added once measured.

## Design constraint: pure-C++ headers

Public headers under `arrow/vulkan/` MUST NOT include `<vulkan/vulkan.h>`.
Vulkan handles are stored as `void*` (with the underlying type
documented in a comment). The bridge happens only in `.cc` files via
`vulkan_internal.hh` — same pattern as `arrow_metal`'s pImpl story.

This means downstream consumers can `#include "arrow/vulkan/vulkan_api.h"`
from any plain `.cc` file without forcing a Vulkan-aware compile.
