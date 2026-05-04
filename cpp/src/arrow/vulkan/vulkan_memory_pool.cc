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

#include "arrow/vulkan/vulkan_memory_pool.h"
#include "arrow/vulkan/vulkan_internal.hh"

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace arrow {
namespace vulkan {
/** --------------------------------------------------------------------------------------------- VulkanMemoryPool::Impl
 * @brief State for the Vulkan-backed MemoryPool.
 *
 * Reverse-maps the mapped pointer (returned to the caller of Allocate)
 * back to the (VkBuffer, VkDeviceMemory) pair that owns it, plus the
 * allocation size for accounting. Free walks the map under a mutex.
 */
class VulkanMemoryPool::Impl {
 public:
    explicit Impl(std::shared_ptr<VulkanDevice> device) : device_{std::move(device)} {}
    ~Impl() {
        std::lock_guard<std::mutex> lk{mu_};
        if (!device_) return;
        VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
        if (dev == VK_NULL_HANDLE) return;
        for (auto& [_, e] : alive_) {
            vkUnmapMemory(dev, e.mem);
            vkDestroyBuffer(dev, e.buf, nullptr);
            vkFreeMemory(dev, e.mem, nullptr);
        }
    }
    Status Allocate(int64_t size, int64_t /*alignment*/, uint8_t** out) {
        if (size < 0) {
            return Status::Invalid("VulkanMemoryPool::Allocate: negative size ", size);
        }
        VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
        const VkDeviceSize alloc = size == 0 ? 1 : static_cast<VkDeviceSize>(size);

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = alloc;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf = VK_NULL_HANDLE;
        ARROW_VK_RETURN_NOT_OK(vkCreateBuffer(dev, &bci, nullptr, &buf),
                               "vkCreateBuffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, buf, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = device_->coherent_memory_type_index();
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkResult r = vkAllocateMemory(dev, &mai, nullptr, &mem);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(dev, buf, nullptr);
            return Status::OutOfMemory("vkAllocateMemory: ",
                                       internal::VkResultToString(r));
        }
        r = vkBindBufferMemory(dev, buf, mem, 0);
        if (r != VK_SUCCESS) {
            vkFreeMemory(dev, mem, nullptr);
            vkDestroyBuffer(dev, buf, nullptr);
            return Status::Invalid("vkBindBufferMemory: ",
                                   internal::VkResultToString(r));
        }
        void* mapped = nullptr;
        r = vkMapMemory(dev, mem, 0, alloc, 0, &mapped);
        if (r != VK_SUCCESS) {
            vkFreeMemory(dev, mem, nullptr);
            vkDestroyBuffer(dev, buf, nullptr);
            return Status::Invalid("vkMapMemory: ", internal::VkResultToString(r));
        }
        {
            std::lock_guard<std::mutex> lk{mu_};
            alive_.emplace(mapped, Entry{buf, mem, size});
        }
        *out = static_cast<uint8_t*>(mapped);
        bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
        total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
        num_allocations_.fetch_add(1, std::memory_order_relaxed);
        UpdateMaxMemory(bytes_allocated_.load(std::memory_order_relaxed));
        return Status::OK();
    }
    Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                      uint8_t** ptr) {
        uint8_t* fresh = nullptr;
        ARROW_RETURN_NOT_OK(Allocate(new_size, alignment, &fresh));
        const int64_t copy = old_size < new_size ? old_size : new_size;
        if (copy > 0 && *ptr != nullptr) {
            std::memcpy(fresh, *ptr, static_cast<size_t>(copy));
        }
        Free(*ptr, old_size, alignment);
        *ptr = fresh;
        return Status::OK();
    }
    void Free(uint8_t* buffer, int64_t /*size*/, int64_t /*alignment*/) {
        if (buffer == nullptr) return;
        Entry e{};
        bool found = false;
        {
            std::lock_guard<std::mutex> lk{mu_};
            auto it = alive_.find(buffer);
            if (it != alive_.end()) {
                e = it->second;
                alive_.erase(it);
                found = true;
            }
        }
        if (!found) return;  ///< foreign pointer; tolerant of misuse
        VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
        vkUnmapMemory(dev, e.mem);
        vkDestroyBuffer(dev, e.buf, nullptr);
        vkFreeMemory(dev, e.mem, nullptr);
        bytes_allocated_.fetch_sub(e.size, std::memory_order_relaxed);
    }
    int64_t bytes_allocated() const {
        return bytes_allocated_.load(std::memory_order_relaxed);
    }
    int64_t total_bytes_allocated() const {
        return total_bytes_allocated_.load(std::memory_order_relaxed);
    }
    int64_t num_allocations() const {
        return num_allocations_.load(std::memory_order_relaxed);
    }
    int64_t max_memory() const { return max_memory_.load(std::memory_order_relaxed); }
 private:
    struct Entry {
        VkBuffer buf;
        VkDeviceMemory mem;
        int64_t size;
    };
    void UpdateMaxMemory(int64_t current) {
        int64_t prev = max_memory_.load(std::memory_order_relaxed);
        while (current > prev
               && !max_memory_.compare_exchange_weak(prev, current,
                                                    std::memory_order_relaxed)) {
        }
    }
    std::shared_ptr<VulkanDevice> device_;
    std::mutex mu_;
    std::unordered_map<void*, Entry> alive_;
    std::atomic<int64_t> bytes_allocated_{0};
    std::atomic<int64_t> total_bytes_allocated_{0};
    std::atomic<int64_t> num_allocations_{0};
    std::atomic<int64_t> max_memory_{0};
};

VulkanMemoryPool::VulkanMemoryPool(std::shared_ptr<VulkanDevice> device)
    : impl_{std::make_unique<Impl>(std::move(device))} {}
VulkanMemoryPool::~VulkanMemoryPool() = default;

Status VulkanMemoryPool::Allocate(int64_t size, int64_t alignment, uint8_t** out) {
    return impl_->Allocate(size, alignment, out);
}
Status VulkanMemoryPool::Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                                    uint8_t** ptr) {
    return impl_->Reallocate(old_size, new_size, alignment, ptr);
}
void VulkanMemoryPool::Free(uint8_t* buffer, int64_t size, int64_t alignment) {
    impl_->Free(buffer, size, alignment);
}
int64_t VulkanMemoryPool::bytes_allocated() const { return impl_->bytes_allocated(); }
int64_t VulkanMemoryPool::total_bytes_allocated() const {
    return impl_->total_bytes_allocated();
}
int64_t VulkanMemoryPool::num_allocations() const { return impl_->num_allocations(); }
int64_t VulkanMemoryPool::max_memory() const { return impl_->max_memory(); }
std::string VulkanMemoryPool::backend_name() const { return "vulkan"; }

VulkanMemoryPool* VulkanMemoryPool::Instance() {
    static VulkanMemoryPool* singleton = []() -> VulkanMemoryPool* {
        auto dev_result = VulkanDevice::Default();
        if (!dev_result.ok()) {
            return nullptr;
        }
        return new VulkanMemoryPool(dev_result.ValueOrDie());
    }();
    return singleton;
}

}  // namespace vulkan
}  // namespace arrow
