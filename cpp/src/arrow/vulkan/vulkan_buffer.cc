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

#include "arrow/vulkan/vulkan_buffer.h"
#include "arrow/vulkan/vulkan_device.h"
#include "arrow/vulkan/vulkan_internal.hh"
#include "arrow/vulkan/vulkan_memory.h"

#include <utility>

namespace arrow {
namespace vulkan {
namespace {
constexpr const char* kVulkanDeviceTypeName = "arrow::vulkan::VulkanDevice";
}  // namespace

VulkanBuffer::VulkanBuffer(void* vk_buffer, void* vk_memory, void* mapped_ptr,
                           std::shared_ptr<VulkanMemoryManager> mm,
                           std::shared_ptr<Buffer> parent, int64_t logical_size)
    : Buffer(/*data=*/nullptr, /*size=*/0),
      vk_buffer_{vk_buffer},
      vk_memory_{vk_memory},
      offset_{0},
      release_fn_{nullptr},
      user_data_{nullptr} {
    data_ = static_cast<const uint8_t*>(mapped_ptr);
    if (logical_size >= 0) {
        size_ = logical_size;
        capacity_ = logical_size;
    } else {
        // The caller (VulkanMemoryManager::AllocateBuffer) sets logical_size
        // explicitly to the requested user size; this branch is mainly for
        // test code that wants the full mapped region.
        size_ = 0;
        capacity_ = 0;
    }
    is_mutable_ = true;
    parent_ = std::move(parent);
    SetMemoryManager(std::move(mm));
}
VulkanBuffer::VulkanBuffer(const std::shared_ptr<VulkanBuffer>& parent, int64_t offset,
                           int64_t size)
    : Buffer(/*data=*/nullptr, /*size=*/0),
      vk_buffer_{nullptr},
      vk_memory_{nullptr},
      offset_{parent->offset_ + offset},
      release_fn_{nullptr},
      user_data_{nullptr} {
    parent_ = parent;
    data_ = parent->data_ + offset;
    size_ = size;
    capacity_ = size;
    is_mutable_ = parent->is_mutable_;
    SetMemoryManager(parent->memory_manager());
}
VulkanBuffer::~VulkanBuffer() {
    // Slice views: nothing to do (parent owns resources).
    if (vk_buffer_ == nullptr && vk_memory_ == nullptr) {
        return;
    }
    // Root: walk the memory_manager → device chain to recover the VkDevice
    // handle. If the memory manager has been freed (impossible while we
    // hold a strong ref via Buffer's shared_ptr, but defensive), fall through.
    auto mm = std::static_pointer_cast<VulkanMemoryManager>(memory_manager());
    if (!mm) return;
    auto vd = mm->vulkan_device();
    if (!vd) return;
    VkDevice dev = internal::FromOpaque<VkDevice>(vd->vk_device());
    if (dev == VK_NULL_HANDLE) return;
    // Order: unmap → destroy buffer → free memory → invoke release_fn.
    if (vk_memory_ != nullptr) {
        VkDeviceMemory mem = internal::FromOpaque<VkDeviceMemory>(vk_memory_);
        vkUnmapMemory(dev, mem);
    }
    if (vk_buffer_ != nullptr) {
        VkBuffer buf = internal::FromOpaque<VkBuffer>(vk_buffer_);
        vkDestroyBuffer(dev, buf, nullptr);
    }
    if (vk_memory_ != nullptr) {
        VkDeviceMemory mem = internal::FromOpaque<VkDeviceMemory>(vk_memory_);
        vkFreeMemory(dev, mem, nullptr);
    }
    if (release_fn_ != nullptr) {
        release_fn_(const_cast<uint8_t*>(data_), size_, user_data_);
    }
}

void* VulkanBuffer::root_vk_buffer() const {
    const VulkanBuffer* cur = this;
    while (cur->vk_buffer_ == nullptr && cur->parent_ != nullptr) {
        cur = static_cast<const VulkanBuffer*>(cur->parent_.get());
    }
    return cur->vk_buffer_;
}

Result<std::shared_ptr<VulkanBuffer>> VulkanBuffer::FromBuffer(
    std::shared_ptr<Buffer> buffer) {
    if (!buffer) {
        return Status::Invalid("VulkanBuffer::FromBuffer: buffer is null");
    }
    if (buffer->device_type() != DeviceAllocationType::kVULKAN) {
        return Status::Invalid(
            "VulkanBuffer::FromBuffer: buffer is not Vulkan-backed");
    }
    auto cast = std::dynamic_pointer_cast<VulkanBuffer>(buffer);
    if (cast == nullptr) {
        return Status::Invalid("VulkanBuffer::FromBuffer: dynamic_cast failed");
    }
    return cast;
}

bool IsVulkanBuffer(const Buffer& buffer) {
    return buffer.device_type() == DeviceAllocationType::kVULKAN
        && buffer.device()->type_name() == kVulkanDeviceTypeName;
}
/** --------------------------------------------------------------------------------------------- Wrap
 * @brief Adopt host memory via VK_EXT_external_memory_host.
 *
 * Allocates a VkBuffer of the given size, allocates a matching
 * VkDeviceMemory using `VkImportMemoryHostPointerInfoEXT` so the device
 * memory aliases the existing host pages, binds them, and persistently
 * maps. The caller-supplied release_fn fires once at destruction.
 */
Result<std::shared_ptr<VulkanBuffer>> VulkanBuffer::Wrap(
    std::shared_ptr<VulkanMemoryManager> mm, void* host_ptr, int64_t size,
    ReleaseFn release_fn, void* user_data, std::shared_ptr<Buffer> parent) {
    if (host_ptr == nullptr || size <= 0) {
        return Status::Invalid("VulkanBuffer::Wrap: host_ptr/size must be non-null/positive");
    }
    auto vd = mm->vulkan_device();
    if (!vd->supports_imported_host_memory()) {
        return Status::NotImplemented(
            "VulkanBuffer::Wrap: VK_EXT_external_memory_host not supported on this device");
    }
    const int64_t align = vd->imported_host_pointer_alignment();
    if (align <= 0) {
        return Status::Invalid(
            "VulkanBuffer::Wrap: minImportedHostPointerAlignment is zero (driver bug?)");
    }
    const auto addr = reinterpret_cast<uintptr_t>(host_ptr);
    if ((addr % static_cast<uintptr_t>(align)) != 0
        || (size % align) != 0) {
        return Status::Invalid(
            "VulkanBuffer::Wrap: host_ptr and size must be aligned to ",
            align, " bytes (minImportedHostPointerAlignment)");
    }
    VkDevice dev = internal::FromOpaque<VkDevice>(vd->vk_device());
    VkPhysicalDevice phys =
        internal::FromOpaque<VkPhysicalDevice>(vd->vk_physical_device());

    // Query allowed memoryTypeBits for this host pointer.
    VkMemoryHostPointerPropertiesEXT host_props{};
    host_props.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    auto vkGetMemoryHostPointerPropertiesEXT =
        reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT"));
    if (vkGetMemoryHostPointerPropertiesEXT == nullptr) {
        return Status::NotImplemented(
            "VulkanBuffer::Wrap: vkGetMemoryHostPointerPropertiesEXT not loaded");
    }
    ARROW_VK_RETURN_NOT_OK(
        vkGetMemoryHostPointerPropertiesEXT(
            dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host_ptr,
            &host_props),
        "vkGetMemoryHostPointerPropertiesEXT");

    // Pick a memory type allowed by the host pointer that also has
    // HOST_VISIBLE | HOST_COHERENT (DEVICE_LOCAL is irrelevant for imported host).
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    constexpr VkMemoryPropertyFlags kWant =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int chosen = -1;
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
        if (!(host_props.memoryTypeBits & (1u << t))) continue;
        if ((mp.memoryTypes[t].propertyFlags & kWant) == kWant) {
            chosen = static_cast<int>(t);
            break;
        }
    }
    if (chosen < 0) {
        return Status::Invalid(
            "VulkanBuffer::Wrap: no memory type matches host pointer + HOST_COHERENT");
    }

    // Create the VkBuffer with external-memory usage.
    VkExternalMemoryBufferCreateInfo ext_buf{};
    ext_buf.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext_buf;
    bci.size = static_cast<VkDeviceSize>(size);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf = VK_NULL_HANDLE;
    ARROW_VK_RETURN_NOT_OK(vkCreateBuffer(dev, &bci, nullptr, &buf), "vkCreateBuffer");

    // Import the host pointer as VkDeviceMemory.
    VkImportMemoryHostPointerInfoEXT import_info{};
    import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_info.pHostPointer = host_ptr;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &import_info;
    mai.allocationSize = static_cast<VkDeviceSize>(size);
    mai.memoryTypeIndex = static_cast<uint32_t>(chosen);

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult r = vkAllocateMemory(dev, &mai, nullptr, &mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(dev, buf, nullptr);
        return Status::Invalid("VulkanBuffer::Wrap: vkAllocateMemory failed: ",
                               internal::VkResultToString(r));
    }
    r = vkBindBufferMemory(dev, buf, mem, 0);
    if (r != VK_SUCCESS) {
        vkFreeMemory(dev, mem, nullptr);
        vkDestroyBuffer(dev, buf, nullptr);
        return Status::Invalid("VulkanBuffer::Wrap: vkBindBufferMemory failed: ",
                               internal::VkResultToString(r));
    }
    // No vkMapMemory needed — the host pointer IS the mapped pointer.
    auto wrapped = std::make_shared<VulkanBuffer>(
        internal::ToOpaque(buf), internal::ToOpaque(mem), host_ptr, std::move(mm),
        std::move(parent), /*logical_size=*/size);
    wrapped->release_fn_ = release_fn;
    wrapped->user_data_ = user_data;
    return wrapped;
}

}  // namespace vulkan
}  // namespace arrow
