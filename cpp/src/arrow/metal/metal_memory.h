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
#include "arrow/metal/metal_buffer.h"
#include "arrow/metal/metal_device.h"
#include "arrow/metal/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace metal {
/** --------------------------------------------------------------------------------------------- MetalMemoryManager
 * @brief MemoryManager that produces Metal-coherent buffers
 *
 * Allocations go through `[mtl_device newBufferWithLength:options:]`
 * with `MTLResourceStorageModeShared`, so the resulting buffer's
 * `.contents` pointer is the Arrow `data()` pointer and is
 * simultaneously visible to the GPU as an MTLBuffer.
 *
 * Copy and View overrides take advantage of the unified memory
 * architecture: CPU↔Metal copies become memcpy when storage is shared,
 * and View paths return a no-copy alias whenever alignment permits.
 */
class ARROW_METAL_EXPORT MetalMemoryManager : public MemoryManager {
 public:
    Result<std::shared_ptr<io::RandomAccessFile>> GetBufferReader(
        std::shared_ptr<Buffer> buf) override;
    Result<std::shared_ptr<io::OutputStream>> GetBufferWriter(
        std::shared_ptr<Buffer> buf) override;
    Result<std::unique_ptr<Buffer>> AllocateBuffer(int64_t size) override;
    /// \brief MetalDevice associated with this MemoryManager (typed shorthand)
    std::shared_ptr<MetalDevice> metal_device() const;
    /// \brief Create a wrapped MTLSharedEvent for cross-stream synchronisation
    Result<std::shared_ptr<Device::SyncEvent>> MakeDeviceSyncEvent() override;
    /// \brief Wrap an externally created MTLSharedEvent
    Result<std::shared_ptr<Device::SyncEvent>> WrapDeviceSyncEvent(
        void* sync_event,
        Device::SyncEvent::release_fn_t release_sync_event) override;
    /** ----------------------------------------------------------------------- Make
     * @brief Construct a MetalMemoryManager bound to a MetalDevice
     *
     * Used by MetalDevice::default_memory_manager(); also exposed for
     * call sites that need an explicit memory manager handle.
     */
    static std::shared_ptr<MetalMemoryManager> Make(
        const std::shared_ptr<MetalDevice>& device);
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
    friend class MetalDevice;
};
/** --------------------------------------------------------------------------------------------- IsMetalMemoryManager
 * @brief Whether a MemoryManager instance is a MetalMemoryManager
 */
ARROW_METAL_EXPORT bool IsMetalMemoryManager(const MemoryManager& mm);
/** --------------------------------------------------------------------------------------------- AsMetalMemoryManager
 * @brief Cast a MemoryManager to MetalMemoryManager or fail with Status::Invalid
 */
ARROW_METAL_EXPORT Result<std::shared_ptr<MetalMemoryManager>> AsMetalMemoryManager(
    const std::shared_ptr<MemoryManager>& mm);

}  // namespace metal
}  // namespace arrow
