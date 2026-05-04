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

#include "arrow/vulkan/vulkan_memory.h"
#include "arrow/vulkan/vulkan_buffer.h"
#include "arrow/vulkan/vulkan_device.h"
#include "arrow/vulkan/vulkan_internal.hh"

#include <cstring>
#include <utility>

#include "arrow/buffer.h"
#include "arrow/device.h"
#include "arrow/io/memory.h"

namespace arrow {
namespace vulkan {
namespace {
constexpr const char* kVulkanDeviceTypeName = "arrow::vulkan::VulkanDevice";
}  // namespace

std::shared_ptr<VulkanMemoryManager> VulkanMemoryManager::Make(
    const std::shared_ptr<VulkanDevice>& device) {
    struct EnableMakeShared : VulkanMemoryManager {
        explicit EnableMakeShared(const std::shared_ptr<Device>& d)
            : VulkanMemoryManager(d) {}
    };
    return std::make_shared<EnableMakeShared>(device);
}
std::shared_ptr<VulkanDevice> VulkanMemoryManager::vulkan_device() const {
    return std::static_pointer_cast<VulkanDevice>(device_);
}

Result<std::shared_ptr<io::RandomAccessFile>> VulkanMemoryManager::GetBufferReader(
    std::shared_ptr<Buffer> buf) {
    if (!buf) {
        return Status::Invalid("VulkanMemoryManager::GetBufferReader: null buffer");
    }
    return std::make_shared<io::BufferReader>(std::move(buf));
}
Result<std::shared_ptr<io::OutputStream>> VulkanMemoryManager::GetBufferWriter(
    std::shared_ptr<Buffer> buf) {
    if (!buf) {
        return Status::Invalid("VulkanMemoryManager::GetBufferWriter: null buffer");
    }
    if (!buf->is_mutable()) {
        return Status::Invalid("VulkanMemoryManager::GetBufferWriter: read-only buffer");
    }
    return std::make_shared<io::FixedSizeBufferWriter>(std::move(buf));
}
/** --------------------------------------------------------------------------------------------- AllocateBuffer
 * @brief Allocate a host-coherent VkBuffer + VkDeviceMemory pair.
 *
 * Uses the device's pre-cached `coherent_memory_type_index()` so we don't
 * re-walk memoryTypes on every allocation. Persistent map at construction;
 * unmap + free in ~VulkanBuffer.
 */
Result<std::unique_ptr<Buffer>> VulkanMemoryManager::AllocateBuffer(int64_t size) {
    if (size < 0) {
        return Status::Invalid("VulkanMemoryManager::AllocateBuffer: negative size ",
                               size);
    }
    auto vd = vulkan_device();
    VkDevice dev = internal::FromOpaque<VkDevice>(vd->vk_device());
    // Vulkan refuses size == 0 for VkBufferCreateInfo. Pad to 1 byte and
    // pass logical_size=0 through to VulkanBuffer (mirror of the Metal
    // size-0 fix). A 1-byte buffer still consumes memoryRequirements.size,
    // typically rounded up to 256 bytes by the allocator, which is fine.
    const VkDeviceSize alloc_size = size == 0 ? 1 : static_cast<VkDeviceSize>(size);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = alloc_size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf = VK_NULL_HANDLE;
    ARROW_VK_RETURN_NOT_OK(vkCreateBuffer(dev, &bci, nullptr, &buf), "vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = vd->coherent_memory_type_index();

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult r = vkAllocateMemory(dev, &mai, nullptr, &mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(dev, buf, nullptr);
        return Status::OutOfMemory(
            "VulkanMemoryManager::AllocateBuffer: vkAllocateMemory failed: ",
            internal::VkResultToString(r));
    }
    r = vkBindBufferMemory(dev, buf, mem, 0);
    if (r != VK_SUCCESS) {
        vkFreeMemory(dev, mem, nullptr);
        vkDestroyBuffer(dev, buf, nullptr);
        return Status::Invalid(
            "VulkanMemoryManager::AllocateBuffer: vkBindBufferMemory failed: ",
            internal::VkResultToString(r));
    }
    void* mapped = nullptr;
    r = vkMapMemory(dev, mem, 0, alloc_size, 0, &mapped);
    if (r != VK_SUCCESS) {
        vkFreeMemory(dev, mem, nullptr);
        vkDestroyBuffer(dev, buf, nullptr);
        return Status::Invalid(
            "VulkanMemoryManager::AllocateBuffer: vkMapMemory failed: ",
            internal::VkResultToString(r));
    }
    auto mm = std::static_pointer_cast<VulkanMemoryManager>(shared_from_this());
    return std::make_unique<VulkanBuffer>(internal::ToOpaque(buf),
                                          internal::ToOpaque(mem), mapped,
                                          std::move(mm), /*parent=*/nullptr,
                                          /*logical_size=*/size);
}

Result<std::shared_ptr<Device::SyncEvent>> VulkanMemoryManager::MakeDeviceSyncEvent() {
    VkDevice dev = internal::FromOpaque<VkDevice>(vulkan_device()->vk_device());
    VkSemaphoreTypeCreateInfo tci{};
    tci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue = 0;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &tci;
    VkSemaphore sem = VK_NULL_HANDLE;
    ARROW_VK_RETURN_NOT_OK(vkCreateSemaphore(dev, &sci, nullptr, &sem),
                           "vkCreateSemaphore");
    void* opaque = internal::ToOpaque(sem);
    auto vd = vulkan_device();
    auto release = [vd_capture = vd](void* p) {
        if (p == nullptr || !vd_capture) return;
        VkDevice d = internal::FromOpaque<VkDevice>(vd_capture->vk_device());
        VkSemaphore s = internal::FromOpaque<VkSemaphore>(p);
        vkDestroySemaphore(d, s, nullptr);
    };
    struct EnableMakeShared : VulkanDevice::SyncEvent {
        EnableMakeShared(std::shared_ptr<VulkanDevice> d, void* o,
                         Device::SyncEvent::release_fn_t r)
            : VulkanDevice::SyncEvent(std::move(d), o, std::move(r)) {}
    };
    return std::make_shared<EnableMakeShared>(vd, opaque, release);
}
Result<std::shared_ptr<Device::SyncEvent>> VulkanMemoryManager::WrapDeviceSyncEvent(
    void* sync_event, Device::SyncEvent::release_fn_t release_sync_event) {
    if (sync_event == nullptr) {
        return Status::Invalid("VulkanMemoryManager::WrapDeviceSyncEvent: null event");
    }
    struct EnableMakeShared : VulkanDevice::SyncEvent {
        EnableMakeShared(std::shared_ptr<VulkanDevice> d, void* o,
                         Device::SyncEvent::release_fn_t r)
            : VulkanDevice::SyncEvent(std::move(d), o, std::move(r)) {}
    };
    return std::make_shared<EnableMakeShared>(vulkan_device(), sync_event,
                                              std::move(release_sync_event));
}
/** --------------------------------------------------------------------------------------------- CopyBufferFrom
 * @brief CPU → Vulkan copy. Unified memory ⇒ memcpy into the mapped region.
 */
Result<std::shared_ptr<Buffer>> VulkanMemoryManager::CopyBufferFrom(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& from) {
    if (!from->is_cpu()) {
        return nullptr;
    }
    ARROW_ASSIGN_OR_RAISE(auto dst, AllocateBuffer(buf->size()));
    if (buf->size() > 0) {
        std::memcpy(dst->mutable_data(), buf->data(), static_cast<size_t>(buf->size()));
    }
    return std::shared_ptr<Buffer>(std::move(dst));
}
Result<std::shared_ptr<Buffer>> VulkanMemoryManager::CopyBufferTo(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& to) {
    if (!to->is_cpu()) {
        return nullptr;
    }
    ARROW_ASSIGN_OR_RAISE(auto dst, to->AllocateBuffer(buf->size()));
    if (buf->size() > 0) {
        std::memcpy(dst->mutable_data(), buf->data(), static_cast<size_t>(buf->size()));
    }
    return std::shared_ptr<Buffer>(std::move(dst));
}
Result<std::unique_ptr<Buffer>> VulkanMemoryManager::CopyNonOwnedFrom(
    const Buffer& buf, const std::shared_ptr<MemoryManager>& from) {
    if (!from->is_cpu()) return nullptr;
    ARROW_ASSIGN_OR_RAISE(auto dst, AllocateBuffer(buf.size()));
    if (buf.size() > 0) {
        std::memcpy(dst->mutable_data(), buf.data(), static_cast<size_t>(buf.size()));
    }
    return dst;
}
Result<std::unique_ptr<Buffer>> VulkanMemoryManager::CopyNonOwnedTo(
    const Buffer& buf, const std::shared_ptr<MemoryManager>& to) {
    if (!to->is_cpu()) return nullptr;
    ARROW_ASSIGN_OR_RAISE(auto dst, to->AllocateBuffer(buf.size()));
    if (buf.size() > 0) {
        std::memcpy(dst->mutable_data(), buf.data(), static_cast<size_t>(buf.size()));
    }
    return dst;
}
/** --------------------------------------------------------------------------------------------- ViewBufferFrom
 * @brief Zero-copy CPU → Vulkan view via VK_EXT_external_memory_host.
 *
 * Page-aligned host pointer + page-multiple size → wraps in-place via
 * VulkanBuffer::Wrap with a no-op release_fn (parent_ pins the source
 * Arrow Buffer; the source owns the bytes, so vkFreeMemory has nothing
 * to free). If alignment fails or the extension is absent, returns
 * nullptr to let the caller fall back.
 */
Result<std::shared_ptr<Buffer>> VulkanMemoryManager::ViewBufferFrom(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& from) {
    if (!from->is_cpu() || !buf || buf->size() == 0) {
        return nullptr;
    }
    auto vd = vulkan_device();
    if (!vd->supports_imported_host_memory()) {
        return nullptr;
    }
    const int64_t align = vd->imported_host_pointer_alignment();
    const auto addr = reinterpret_cast<uintptr_t>(buf->data());
    if (align <= 0
        || (addr % static_cast<uintptr_t>(align)) != 0
        || (buf->size() % align) != 0) {
        return nullptr;  ///< caller falls back to copy
    }
    // Wrap with a no-op release: the source Arrow Buffer owns the host memory
    // (kept alive via parent_); vkFreeMemory does NOT free imported pages.
    auto wrapped = VulkanBuffer::Wrap(
        std::static_pointer_cast<VulkanMemoryManager>(shared_from_this()),
        const_cast<uint8_t*>(buf->data()), buf->size(),
        /*release_fn=*/nullptr, /*user_data=*/nullptr,
        /*parent=*/buf);
    if (!wrapped.ok()) {
        return nullptr;
    }
    return std::static_pointer_cast<Buffer>(wrapped.ValueOrDie());
}
/** --------------------------------------------------------------------------------------------- ViewBufferTo
 * @brief Zero-copy Vulkan → CPU view. Always succeeds on host-coherent storage.
 */
Result<std::shared_ptr<Buffer>> VulkanMemoryManager::ViewBufferTo(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& to) {
    if (!to->is_cpu() || !buf) return nullptr;
    auto view = std::make_shared<Buffer>(buf->data(), buf->size(), to, buf,
                                         /*device_type_override=*/std::nullopt);
    return view;
}

bool IsVulkanMemoryManager(const MemoryManager& mm) {
    return mm.device()->type_name() == kVulkanDeviceTypeName;
}
Result<std::shared_ptr<VulkanMemoryManager>> AsVulkanMemoryManager(
    const std::shared_ptr<MemoryManager>& mm) {
    if (!mm || !IsVulkanMemoryManager(*mm)) {
        return Status::Invalid("Expected VulkanMemoryManager");
    }
    return std::static_pointer_cast<VulkanMemoryManager>(mm);
}

}  // namespace vulkan
}  // namespace arrow
