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

// Process-wide initialization for arrow_vulkan.
//
// Registers the Vulkan DeviceMapper at static-init time so that
// arrow::ImportDeviceArray / ImportDeviceRecordBatch resolve a
// VulkanMemoryManager when handed an ArrowDeviceArray with
// device_type == ARROW_DEVICE_VULKAN (= 7). The mapper takes the C-Data
// `device_id` (interpreted as VulkanDevice::uuid_hash() — hash of the
// VkPhysicalDevice's pipelineCacheUUID) and resolves to the local
// VulkanDevice with the same UUID. If no match is found, falls back to
// the system default device (covers cross-process IPC where producer
// and consumer share a GPU but the receiver doesn't recognize the ID).

#include "arrow/device.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/logging_internal.h"
#include "arrow/vulkan/visibility.h"
#include "arrow/vulkan/vulkan_device.h"
#include "arrow/vulkan/vulkan_memory.h"

namespace arrow {
namespace vulkan {
namespace internal {
/** --------------------------------------------------------------------------------------------- VulkanDeviceMapper
 * @brief Resolve `device_id` (uuid_hash) → VulkanMemoryManager.
 *
 * Bound at static-init via `RegisterDeviceMapper`. The mapper iterates
 * the cached VulkanDevices looking for a `uuid_hash()` match; failing
 * that, returns the default device's MemoryManager. The C-Data Interface
 * import path consults this when given an ArrowDeviceArray with
 * `device_type == ARROW_DEVICE_VULKAN`.
 */
static Result<std::shared_ptr<MemoryManager>> VulkanDeviceMapper(int64_t device_id) {
    ARROW_ASSIGN_OR_RAISE(auto* mgr, VulkanDeviceManager::Instance());
    for (int i = 0; i < mgr->num_devices(); ++i) {
        ARROW_ASSIGN_OR_RAISE(auto dev, mgr->GetDevice(i));
        if (dev->uuid_hash() == device_id) {
            return dev->default_memory_manager();
        }
    }
    // Fallback: default device. Covers cross-process IPC where producer
    // encoded a non-portable UUID hash but the consumer has only one GPU.
    ARROW_ASSIGN_OR_RAISE(auto fallback, mgr->GetDevice(0));
    return fallback->default_memory_manager();
}
/** --------------------------------------------------------------------------------------------- EnsureVulkanLinked
 * @brief Static-init registration. Idempotent via std::call_once so
 *        repeated dynamic-loads of arrow_vulkan don't double-register.
 */
ARROW_VULKAN_EXPORT void EnsureVulkanLinked();
void EnsureVulkanLinked() {
    static const auto kRegister = []() {
        Status s = RegisterDeviceMapper(DeviceAllocationType::kVULKAN, &VulkanDeviceMapper);
        if (!s.ok()) {
            ARROW_LOG(WARNING) << "arrow_vulkan: RegisterDeviceMapper failed: " << s;
        }
        return 0;
    }();
    (void)kRegister;
}
namespace {
/// Force-trigger registration at library load.
struct VulkanDeviceMapperAutoRegister {
    VulkanDeviceMapperAutoRegister() { EnsureVulkanLinked(); }
};
static VulkanDeviceMapperAutoRegister kVulkanDeviceMapperAutoRegister;
}  // namespace

}  // namespace internal
}  // namespace vulkan
}  // namespace arrow
