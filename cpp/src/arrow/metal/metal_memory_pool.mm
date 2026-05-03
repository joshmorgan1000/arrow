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

#include "arrow/metal/metal_memory_pool.h"
#include "arrow/metal/metal_internal.hh"

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace arrow {
namespace metal {
/** --------------------------------------------------------------------------------------------- MetalMemoryPool::Impl
 * @brief State for the Metal-backed MemoryPool: reverse map from
 *        the `.contents` pointer (returned to the caller) back to the
 *        retained id<MTLBuffer> opaque handle so Free can release the
 *        right object.
 */
class MetalMemoryPool::Impl {
 public:
    explicit Impl(std::shared_ptr<MetalDevice> device) : device_{std::move(device)} {}
    ~Impl() {
        std::lock_guard<std::mutex> lk{mu_};
        for (auto& [_, opaque] : alive_) {
            internal::ReleaseOpaque(opaque);
        }
    }
    Status Allocate(int64_t size, int64_t /*alignment*/, uint8_t** out) {
        if (size < 0) {
            return Status::Invalid("MetalMemoryPool::Allocate: negative size ", size);
        }
        @autoreleasepool {
            id<MTLDevice> dev =
                internal::FromOpaqueBorrowed<id<MTLDevice>>(device_->mtl_device());
            // Metal rejects a 0-length buffer; round up so callers can still
            // perform the matching Free without special-casing zero.
            const NSUInteger alloc = size == 0 ? 1 : static_cast<NSUInteger>(size);
            id<MTLBuffer> mtl =
                [dev newBufferWithLength:alloc options:MTLResourceStorageModeShared];
            if (mtl == nil) {
                return Status::OutOfMemory(
                    "MetalMemoryPool::Allocate: newBufferWithLength returned nil");
            }
            void* opaque = internal::ToOpaqueRetained(mtl);
            void* contents = [mtl contents];
            {
                std::lock_guard<std::mutex> lk{mu_};
                alive_.emplace(contents, opaque);
            }
            *out = static_cast<uint8_t*>(contents);
            bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
            total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
            num_allocations_.fetch_add(1, std::memory_order_relaxed);
            UpdateMaxMemory(bytes_allocated_.load(std::memory_order_relaxed));
            return Status::OK();
        }
    }
    Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                      uint8_t** ptr) {
        // Aligned reallocate without the OS allocator's help: alloc fresh,
        // copy, free old. Functionally identical to system pool's path.
        uint8_t* fresh = nullptr;
        ARROW_RETURN_NOT_OK(Allocate(new_size, alignment, &fresh));
        const int64_t copy_size = old_size < new_size ? old_size : new_size;
        if (copy_size > 0 && *ptr != nullptr) {
            std::memcpy(fresh, *ptr, static_cast<size_t>(copy_size));
        }
        Free(*ptr, old_size, alignment);
        *ptr = fresh;
        return Status::OK();
    }
    void Free(uint8_t* buffer, int64_t size, int64_t /*alignment*/) {
        if (buffer == nullptr) {
            return;
        }
        void* opaque = nullptr;
        {
            std::lock_guard<std::mutex> lk{mu_};
            auto it = alive_.find(buffer);
            if (it == alive_.end()) {
                // Foreign pointer; ignore (matches system pool's tolerance).
                return;
            }
            opaque = it->second;
            alive_.erase(it);
        }
        internal::ReleaseOpaque(opaque);
        bytes_allocated_.fetch_sub(size, std::memory_order_relaxed);
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
    void UpdateMaxMemory(int64_t current) {
        int64_t prev = max_memory_.load(std::memory_order_relaxed);
        while (current > prev &&
               !max_memory_.compare_exchange_weak(prev, current,
                                                  std::memory_order_relaxed)) {
        }
    }
    std::shared_ptr<MetalDevice> device_;
    std::mutex mu_;
    std::unordered_map<void*, void*> alive_;  ///< contents → retained id<MTLBuffer>
    std::atomic<int64_t> bytes_allocated_{0};
    std::atomic<int64_t> total_bytes_allocated_{0};
    std::atomic<int64_t> num_allocations_{0};
    std::atomic<int64_t> max_memory_{0};
};

MetalMemoryPool::MetalMemoryPool(std::shared_ptr<MetalDevice> device)
    : impl_{std::make_unique<Impl>(std::move(device))} {}
MetalMemoryPool::~MetalMemoryPool() = default;

Status MetalMemoryPool::Allocate(int64_t size, int64_t alignment, uint8_t** out) {
    return impl_->Allocate(size, alignment, out);
}
Status MetalMemoryPool::Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                                   uint8_t** ptr) {
    return impl_->Reallocate(old_size, new_size, alignment, ptr);
}
void MetalMemoryPool::Free(uint8_t* buffer, int64_t size, int64_t alignment) {
    impl_->Free(buffer, size, alignment);
}
int64_t MetalMemoryPool::bytes_allocated() const { return impl_->bytes_allocated(); }
int64_t MetalMemoryPool::total_bytes_allocated() const {
    return impl_->total_bytes_allocated();
}
int64_t MetalMemoryPool::num_allocations() const { return impl_->num_allocations(); }
int64_t MetalMemoryPool::max_memory() const { return impl_->max_memory(); }
std::string MetalMemoryPool::backend_name() const { return "metal"; }

MetalMemoryPool* MetalMemoryPool::Instance() {
    static MetalMemoryPool* singleton = []() -> MetalMemoryPool* {
        auto dev_result = MetalDevice::Default();
        if (!dev_result.ok()) {
            return nullptr;
        }
        // Leaked on purpose: process-wide singleton. arrow::default_memory_pool()
        // follows the same convention.
        return new MetalMemoryPool(dev_result.ValueOrDie());
    }();
    return singleton;
}

}  // namespace metal
}  // namespace arrow
