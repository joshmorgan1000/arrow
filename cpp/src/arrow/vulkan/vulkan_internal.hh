// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// arrow_vulkan internal bridge header.
//
// Public arrow/vulkan/*.h headers store Vulkan handles as `void*` so plain
// C++ consumers don't need <vulkan/vulkan.h> in their include graph. This
// .hh is the internal counterpart: it pulls in Vulkan headers and provides
// typed reinterpret bridges so .cc translation units can convert opaque
// pointers to/from real handle types in one obvious place.
//
// Vulkan handles split into two categories on the C ABI level:
//   1. Dispatchable handles  (VkDevice, VkInstance, VkQueue, VkCommandBuffer,
//      VkPhysicalDevice) — already pointer-sized, so void* round-trip is
//      a plain reinterpret_cast.
//   2. Non-dispatchable handles  (VkBuffer, VkDeviceMemory, VkSemaphore,
//      VkCommandPool, etc.) — defined as uint64_t on 32-bit platforms and
//      as opaque pointers on 64-bit platforms via VK_DEFINE_NON_DISPATCHABLE_HANDLE.
//      Arrow only targets 64-bit, so void* round-trip is also safe; we
//      static_assert on sizeof for paranoia.

#pragma once

#ifndef __APPLE__
// Vulkan is multi-platform, but we still keep the include guard so this
// file is recognisable as an "internal" header even on non-Apple builds.
#endif

#include <vulkan/vulkan.h>

#include <cstdint>
#include <type_traits>

#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace vulkan {
namespace internal {

static_assert(sizeof(void*) >= sizeof(VkDevice),
              "arrow_vulkan requires 64-bit pointers (dispatchable handles)");
static_assert(sizeof(void*) >= sizeof(VkBuffer),
              "arrow_vulkan requires VK_USE_64_BIT_PTR_DEFINES (non-dispatchable handles)");
/** ----------------------------------------------------------------- ToOpaque/FromOpaque
 * @brief Symmetric reinterpret bridges between native Vulkan handles and `void*`.
 *
 * Every public arrow/vulkan/ public header stores Vulkan handles as `void*`
 * to keep the include graph clean. These helpers do the cast in one
 * obvious place per direction, so accidental drift between header and
 * impl can be caught at compile time (the `static_assert` below ensures
 * pointer-sized handles).
 */
template <typename Handle>
inline void* ToOpaque(Handle h) {
    return reinterpret_cast<void*>(h);
}
template <typename Handle>
inline Handle FromOpaque(void* p) {
    return reinterpret_cast<Handle>(p);
}
/** ----------------------------------------------------------------- VkResultToString
 * @brief Translate a `VkResult` to a stable short string for diagnostics.
 *
 * Used by the `VK_CHECK` macro below; mirrors what `vk_enum_string_helper.h`
 * would give us, but inlined here so we don't need that header in every TU.
 */
inline const char* VkResultToString(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        default: return "VK_ERROR_<unknown>";
    }
}

}  // namespace internal
}  // namespace vulkan
}  // namespace arrow
/** ----------------------------------------------------------------- ARROW_VK_RETURN_NOT_OK
 * @brief Translate a VkResult into a fail-fast `arrow::Status::Invalid` and propagate.
 *
 * Use at every Vulkan call site where the result is fatal-ish:
 *
 * ```cpp
 * VkResult vr = vkCreateBuffer(dev, &ci, nullptr, &buf);
 * ARROW_VK_RETURN_NOT_OK(vr, "vkCreateBuffer");
 * ```
 *
 * Wraps the result code into a human-readable Status carrying the call
 * site label so failures are diagnosable without rebuilding.
 */
#define ARROW_VK_RETURN_NOT_OK(result, what)                                     \
    do {                                                                         \
        const VkResult _vk_r = (result);                                         \
        if (_vk_r != VK_SUCCESS) {                                               \
            return ::arrow::Status::Invalid(                                     \
                (what), " failed: ",                                             \
                ::arrow::vulkan::internal::VkResultToString(_vk_r));             \
        }                                                                        \
    } while (0)
