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

#pragma once

#include <cstdint>
#include <memory>

#include "arrow/buffer.h"
#include "arrow/device.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/vulkan/visibility.h"
#include "arrow/vulkan/vulkan_buffer.h"
#include "arrow/vulkan/vulkan_device.h"

namespace arrow {
namespace vulkan {
/** --------------------------------------------------------------------------------------------- VulkanMemoryManager
 * @brief MemoryManager that produces host-coherent Vulkan buffers.
 *
 * Allocations go through `vkCreateBuffer` + `vkAllocateMemory` against
 * a memory type satisfying `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL`,
 * persistently mapped via vkMapMemory at construction. The resulting
 * buffer's data() pointer is the same physical memory the GPU accesses
 * via the bound VkBuffer.
 *
 * Copy / View overrides take advantage of the unified-memory architecture
 * on integrated GPUs: CPU↔Vulkan copies become memcpy when storage is
 * shared, and View paths return a no-copy alias whenever device alignment
 * permits. Cross-device transfers between two VulkanMemoryManagers tied
 * to different physical devices fall back to the base-class generic
 * staging path.
 */
class ARROW_VULKAN_EXPORT VulkanMemoryManager : public MemoryManager {
 public:
    Result<std::shared_ptr<io::RandomAccessFile>> GetBufferReader(
        std::shared_ptr<Buffer> buf) override;
    Result<std::shared_ptr<io::OutputStream>> GetBufferWriter(
        std::shared_ptr<Buffer> buf) override;
    Result<std::unique_ptr<Buffer>> AllocateBuffer(int64_t size) override;
    /// \brief VulkanDevice associated with this MemoryManager (typed shorthand).
    std::shared_ptr<VulkanDevice> vulkan_device() const;
    /// \brief Create a wrapped VkTimelineSemaphore for cross-stream synchronization.
    Result<std::shared_ptr<Device::SyncEvent>> MakeDeviceSyncEvent() override;
    /// \brief Wrap an externally created VkSemaphore (must be timeline-typed).
    Result<std::shared_ptr<Device::SyncEvent>> WrapDeviceSyncEvent(
        void* sync_event,
        Device::SyncEvent::release_fn_t release_sync_event) override;
    /** ----------------------------------------------------------------------- Make
     * @brief Construct a VulkanMemoryManager bound to a VulkanDevice.
     */
    static std::shared_ptr<VulkanMemoryManager> Make(
        const std::shared_ptr<VulkanDevice>& device);
 protected:
    using MemoryManager::MemoryManager;
    Result<std::shared_ptr<Buffer>> CopyBufferFrom(
        const std::shared_ptr<Buffer>& buf,
        const std::shared_ptr<MemoryManager>& from) override;
    Result<std::shared_ptr<Buffer>> CopyBufferTo(
        const std::shared_ptr<Buffer>& buf,
        const std::shared_ptr<MemoryManager>& to) override;
    Result<std::unique_ptr<Buffer>> CopyNonOwnedFrom(
        const Buffer& buf, const std::shared_ptr<MemoryManager>& from) override;
    Result<std::unique_ptr<Buffer>> CopyNonOwnedTo(
        const Buffer& buf, const std::shared_ptr<MemoryManager>& to) override;
    Result<std::shared_ptr<Buffer>> ViewBufferFrom(
        const std::shared_ptr<Buffer>& buf,
        const std::shared_ptr<MemoryManager>& from) override;
    Result<std::shared_ptr<Buffer>> ViewBufferTo(
        const std::shared_ptr<Buffer>& buf,
        const std::shared_ptr<MemoryManager>& to) override;
    friend class VulkanDevice;
};
/** --------------------------------------------------------------------------------------------- IsVulkanMemoryManager
 * @brief Whether a MemoryManager instance is a VulkanMemoryManager.
 */
ARROW_VULKAN_EXPORT bool IsVulkanMemoryManager(const MemoryManager& mm);
/** --------------------------------------------------------------------------------------------- AsVulkanMemoryManager
 * @brief Cast a MemoryManager to VulkanMemoryManager or fail with Status::Invalid.
 */
ARROW_VULKAN_EXPORT Result<std::shared_ptr<VulkanMemoryManager>> AsVulkanMemoryManager(
    const std::shared_ptr<MemoryManager>& mm);

}  // namespace vulkan
}  // namespace arrow
