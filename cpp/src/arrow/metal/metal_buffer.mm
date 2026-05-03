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

#include "arrow/metal/metal_buffer.h"
#include "arrow/metal/metal_device.h"
#include "arrow/metal/metal_internal.hh"
#include "arrow/metal/metal_memory.h"

#include <unistd.h>
#include <utility>

namespace arrow {
namespace metal {
namespace {
constexpr const char* kMetalDeviceTypeName = "arrow::metal::MetalDevice";
}  // namespace

MetalBuffer::MetalBuffer(void* mtl_buffer, std::shared_ptr<MetalMemoryManager> mm,
                         std::shared_ptr<Buffer> parent)
    : Buffer(/*data=*/nullptr, /*size=*/0),
      mtl_buffer_{mtl_buffer},
      offset_{0} {
    @autoreleasepool {
        id<MTLBuffer> buf = internal::FromOpaqueBorrowed<id<MTLBuffer>>(mtl_buffer);
        // Length cast: MTLBuffer length is NSUInteger (size_t); Arrow Buffer
        // size is int64_t. Capacity == size for shared-memory MTL allocations.
        size_ = static_cast<int64_t>([buf length]);
        capacity_ = size_;
        data_ = static_cast<const uint8_t*>([buf contents]);
        is_mutable_ = true;
    }
    parent_ = std::move(parent);
    SetMemoryManager(std::move(mm));
}
MetalBuffer::MetalBuffer(const std::shared_ptr<MetalBuffer>& parent, int64_t offset,
                         int64_t size)
    : Buffer(/*data=*/nullptr, /*size=*/0),
      mtl_buffer_{nullptr},  ///< storage held by parent_
      offset_{parent->offset_ + offset} {
    parent_ = parent;
    data_ = parent->data_ + offset;
    size_ = size;
    capacity_ = size;
    is_mutable_ = parent->is_mutable_;
    SetMemoryManager(parent->memory_manager());
}
MetalBuffer::~MetalBuffer() {
    if (mtl_buffer_ != nullptr) {
        internal::ReleaseOpaque(mtl_buffer_);
    }
}
Result<std::shared_ptr<MetalBuffer>> MetalBuffer::FromBuffer(
    std::shared_ptr<Buffer> buffer) {
    if (!buffer) {
        return Status::Invalid("MetalBuffer::FromBuffer: buffer is null");
    }
    if (buffer->device_type() != DeviceAllocationType::kMETAL) {
        return Status::Invalid(
            "MetalBuffer::FromBuffer: buffer is not Metal-backed (device_type != kMETAL)");
    }
    auto cast = std::dynamic_pointer_cast<MetalBuffer>(buffer);
    if (cast == nullptr) {
        return Status::Invalid("MetalBuffer::FromBuffer: dynamic_cast failed");
    }
    return cast;
}

bool IsMetalBuffer(const Buffer& buffer) {
    return buffer.device_type() == DeviceAllocationType::kMETAL &&
           buffer.device()->type_name() == kMetalDeviceTypeName;
}

Result<std::shared_ptr<MetalBuffer>> MetalBuffer::Wrap(
    std::shared_ptr<MetalMemoryManager> mm, void* host_ptr, int64_t size,
    std::function<void(void*, int64_t)> release_fn) {
    if (host_ptr == nullptr || size <= 0) {
        return Status::Invalid("MetalBuffer::Wrap: host_ptr/size must be non-null/positive");
    }
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const auto addr = reinterpret_cast<uintptr_t>(host_ptr);
    if ((addr % page) != 0 || (static_cast<size_t>(size) % page) != 0) {
        return Status::Invalid(
            "MetalBuffer::Wrap: host_ptr and size must be page-aligned (",
            page, "-byte page)");
    }
    auto release_copy = std::move(release_fn);  ///< captured by Block
    @autoreleasepool {
        id<MTLDevice> dev =
            internal::FromOpaqueBorrowed<id<MTLDevice>>(mm->metal_device()->mtl_device());
        // Apple invokes the deallocator block exactly once when the
        // MTLBuffer's last retain is released. The block captures the
        // std::function by value so the user-supplied release_fn lives
        // on the heap until called.
        id<MTLBuffer> mtl =
            [dev newBufferWithBytesNoCopy:host_ptr
                                  length:static_cast<NSUInteger>(size)
                                 options:MTLResourceStorageModeShared
                             deallocator:^(void* p, NSUInteger sz) {
                               if (release_copy) {
                                 release_copy(p, static_cast<int64_t>(sz));
                               }
                             }];
        if (mtl == nil) {
            return Status::Invalid(
                "MetalBuffer::Wrap: newBufferWithBytesNoCopy returned nil "
                "(check pointer alignment and storage mode)");
        }
        void* opaque = internal::ToOpaqueRetained(mtl);
        // parent must remain nullptr — the deallocator is the sole
        // owner of host_ptr (see ownership contract in the header).
        return std::make_shared<MetalBuffer>(opaque, std::move(mm), /*parent=*/nullptr);
    }
}

}  // namespace metal
}  // namespace arrow
