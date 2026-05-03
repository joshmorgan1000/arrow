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

#include "arrow/metal/metal_memory.h"
#include "arrow/metal/metal_buffer.h"
#include "arrow/metal/metal_device.h"
#include "arrow/metal/metal_internal.hh"

#include <cstring>
#include <unistd.h>
#include <utility>

#include "arrow/buffer.h"
#include "arrow/device.h"
#include "arrow/io/memory.h"

namespace arrow {
namespace metal {
namespace {
/** --------------------------------------------------------------------------------------------- system_page_size
 * @brief Cached system page size — required alignment for the
 *        `newBufferWithBytesNoCopy:` zero-copy fast path.
 *
 * Apple Silicon defaults to 16 KiB, but `getpagesize()` is the only
 * authoritative source. Cached because the syscall is invoked on every
 * View attempt.
 */
size_t system_page_size() {
    static const size_t kPageSize = []() {
        long ps = ::sysconf(_SC_PAGESIZE);
        return ps > 0 ? static_cast<size_t>(ps) : size_t{16384};
    }();
    return kPageSize;
}
/** --------------------------------------------------------------------------------------------- is_page_aligned
 * @brief Whether a (ptr, size) pair satisfies Metal's
 *        newBufferWithBytesNoCopy alignment constraints.
 */
bool is_page_aligned(const void* ptr, int64_t size) {
    const size_t ps = system_page_size();
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    return (addr % ps == 0) && (static_cast<size_t>(size) % ps == 0);
}
}  // namespace

std::shared_ptr<MetalMemoryManager> MetalMemoryManager::Make(
    const std::shared_ptr<MetalDevice>& device) {
    struct EnableMakeShared : MetalMemoryManager {
        explicit EnableMakeShared(const std::shared_ptr<Device>& d) : MetalMemoryManager(d) {}
    };
    return std::make_shared<EnableMakeShared>(device);
}
std::shared_ptr<MetalDevice> MetalMemoryManager::metal_device() const {
    return std::static_pointer_cast<MetalDevice>(device_);
}

Result<std::shared_ptr<io::RandomAccessFile>> MetalMemoryManager::GetBufferReader(
    std::shared_ptr<Buffer> buf) {
    if (!buf) {
        return Status::Invalid("MetalMemoryManager::GetBufferReader: null buffer");
    }
    return std::make_shared<io::BufferReader>(std::move(buf));
}
Result<std::shared_ptr<io::OutputStream>> MetalMemoryManager::GetBufferWriter(
    std::shared_ptr<Buffer> buf) {
    if (!buf) {
        return Status::Invalid("MetalMemoryManager::GetBufferWriter: null buffer");
    }
    if (!buf->is_mutable()) {
        return Status::Invalid("MetalMemoryManager::GetBufferWriter: buffer is read-only");
    }
    return std::make_shared<io::FixedSizeBufferWriter>(std::move(buf));
}
/** --------------------------------------------------------------------------------------------- AllocateBuffer
 * @brief Allocate a fresh MTLBuffer with shared (CPU+GPU) storage.
 *
 * Uses MTLResourceStorageModeShared so the returned buffer's data
 * pointer is a valid CPU pointer AND a valid input to GPU compute /
 * blit encoders. Allocation cost is dominated by the system page
 * allocator; for sub-page sizes Metal still rounds up internally.
 */
Result<std::unique_ptr<Buffer>> MetalMemoryManager::AllocateBuffer(int64_t size) {
    if (size < 0) {
        return Status::Invalid("MetalMemoryManager::AllocateBuffer: negative size ", size);
    }
    @autoreleasepool {
        id<MTLDevice> dev =
            internal::FromOpaqueBorrowed<id<MTLDevice>>(metal_device()->mtl_device());
        // [device newBufferWithLength:0 ...] returns nil on Apple Silicon, so
        // pad to a 1-byte allocation for empty buffers and pass logical_size=0
        // through to MetalBuffer so Arrow consumers see size()==0 / capacity()==0.
        const NSUInteger alloc_size = size == 0 ? 1 : static_cast<NSUInteger>(size);
        id<MTLBuffer> mtl =
            [dev newBufferWithLength:alloc_size options:MTLResourceStorageModeShared];
        if (mtl == nil) {
            return Status::OutOfMemory(
                "MetalMemoryManager::AllocateBuffer: newBufferWithLength returned nil");
        }
        void* opaque = internal::ToOpaqueRetained(mtl);
        auto mm = std::static_pointer_cast<MetalMemoryManager>(shared_from_this());
        return std::make_unique<MetalBuffer>(opaque, std::move(mm),
                                             /*parent=*/nullptr,
                                             /*logical_size=*/size);
    }
}
Result<std::shared_ptr<Device::SyncEvent>> MetalMemoryManager::MakeDeviceSyncEvent() {
    @autoreleasepool {
        id<MTLDevice> dev =
            internal::FromOpaqueBorrowed<id<MTLDevice>>(metal_device()->mtl_device());
        id<MTLSharedEvent> event = [dev newSharedEvent];
        if (event == nil) {
            return Status::Invalid("[device newSharedEvent] returned nil");
        }
        void* opaque = internal::ToOpaqueRetained(event);
        auto release = [](void* p) { internal::ReleaseOpaque(p); };
        struct EnableMakeShared : MetalDevice::SyncEvent {
            EnableMakeShared(std::shared_ptr<MetalDevice> d, void* o,
                             Device::SyncEvent::release_fn_t r)
                : MetalDevice::SyncEvent(std::move(d), o, std::move(r)) {}
        };
        return std::make_shared<EnableMakeShared>(metal_device(), opaque, release);
    }
}
Result<std::shared_ptr<Device::SyncEvent>> MetalMemoryManager::WrapDeviceSyncEvent(
    void* sync_event, Device::SyncEvent::release_fn_t release_sync_event) {
    if (sync_event == nullptr) {
        return Status::Invalid("MetalMemoryManager::WrapDeviceSyncEvent: null event");
    }
    struct EnableMakeShared : MetalDevice::SyncEvent {
        EnableMakeShared(std::shared_ptr<MetalDevice> d, void* o,
                         Device::SyncEvent::release_fn_t r)
            : MetalDevice::SyncEvent(std::move(d), o, std::move(r)) {}
    };
    return std::make_shared<EnableMakeShared>(metal_device(), sync_event,
                                              std::move(release_sync_event));
}
/** --------------------------------------------------------------------------------------------- CopyBufferFrom
 * @brief CPU → Metal copy. Unified memory ⇒ memcpy into the shared MTLBuffer.
 *
 * Allocates a fresh MTLBuffer of the right size, memcpys the source
 * data into the shared `.contents` pointer. No GPU command buffer is
 * involved because the storage is host-visible.
 *
 * Returns nullptr (not an error) for unsupported `from`s — base class
 * contract — to indicate the caller should try the inverse path or
 * fall back to a generic copy.
 */
Result<std::shared_ptr<Buffer>> MetalMemoryManager::CopyBufferFrom(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& from) {
    if (!from->is_cpu()) {
        return nullptr;  ///< unsupported direction; caller will try CopyBufferTo on `from`
    }
    ARROW_ASSIGN_OR_RAISE(auto dst, AllocateBuffer(buf->size()));
    if (buf->size() > 0) {
        std::memcpy(dst->mutable_data(), buf->data(), static_cast<size_t>(buf->size()));
    }
    return std::shared_ptr<Buffer>(std::move(dst));
}
/** --------------------------------------------------------------------------------------------- CopyBufferTo
 * @brief Metal → CPU copy. Unified memory ⇒ memcpy from the shared MTLBuffer.
 */
Result<std::shared_ptr<Buffer>> MetalMemoryManager::CopyBufferTo(
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
Result<std::unique_ptr<Buffer>> MetalMemoryManager::CopyNonOwnedFrom(
    const Buffer& buf, const std::shared_ptr<MemoryManager>& from) {
    if (!from->is_cpu()) {
        return nullptr;
    }
    ARROW_ASSIGN_OR_RAISE(auto dst, AllocateBuffer(buf.size()));
    if (buf.size() > 0) {
        std::memcpy(dst->mutable_data(), buf.data(), static_cast<size_t>(buf.size()));
    }
    return dst;
}
Result<std::unique_ptr<Buffer>> MetalMemoryManager::CopyNonOwnedTo(
    const Buffer& buf, const std::shared_ptr<MemoryManager>& to) {
    if (!to->is_cpu()) {
        return nullptr;
    }
    ARROW_ASSIGN_OR_RAISE(auto dst, to->AllocateBuffer(buf.size()));
    if (buf.size() > 0) {
        std::memcpy(dst->mutable_data(), buf.data(), static_cast<size_t>(buf.size()));
    }
    return dst;
}
/** --------------------------------------------------------------------------------------------- ViewBufferFrom
 * @brief Zero-copy CPU → Metal view (the heart of "Metal-coherent" Arrow).
 *
 * If the source pointer + size satisfy the system page-alignment
 * requirement, wrap the existing CPU memory in an MTLBuffer via
 * `newBufferWithBytesNoCopy:` — no copy, same physical pages. The
 * source buffer is kept alive via the parent_ pointer of the resulting
 * MetalBuffer, and the MTLBuffer's deallocator is a no-op (storage is
 * owned by the source).
 *
 * If alignment fails we return nullptr, signaling the caller to fall
 * back to a Copy.
 */
Result<std::shared_ptr<Buffer>> MetalMemoryManager::ViewBufferFrom(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& from) {
    if (!from->is_cpu() || !buf || buf->size() == 0) {
        return nullptr;
    }
    if (!is_page_aligned(buf->data(), buf->size())) {
        return nullptr;  ///< caller falls back to copy
    }
    @autoreleasepool {
        id<MTLDevice> dev =
            internal::FromOpaqueBorrowed<id<MTLDevice>>(metal_device()->mtl_device());
        // The deallocator is invoked by Metal when the MTLBuffer is freed.
        // We must NOT free the source memory here — the source Buffer owns
        // it. Hence the no-op block.
        id<MTLBuffer> mtl = [dev newBufferWithBytesNoCopy:const_cast<uint8_t*>(buf->data())
                                                  length:static_cast<NSUInteger>(buf->size())
                                                 options:MTLResourceStorageModeShared
                                             deallocator:^(void*, NSUInteger){}];
        if (mtl == nil) {
            return nullptr;  ///< Metal rejected the alignment; caller falls back
        }
        void* opaque = internal::ToOpaqueRetained(mtl);
        auto mm = std::static_pointer_cast<MetalMemoryManager>(shared_from_this());
        // Pin the source buffer for the lifetime of the view via the
        // parent_ slot; this is what keeps the underlying CPU pages valid.
        auto view = std::make_shared<MetalBuffer>(opaque, std::move(mm), buf);
        return std::static_pointer_cast<Buffer>(view);
    }
}
/** --------------------------------------------------------------------------------------------- ViewBufferTo
 * @brief Zero-copy Metal → CPU view. Always succeeds on Apple Silicon.
 *
 * The MTLBuffer's `.contents` pointer IS a valid CPU pointer because
 * storage is shared. Return a plain Buffer wrapping that pointer with
 * the source MetalBuffer pinned via parent_.
 */
Result<std::shared_ptr<Buffer>> MetalMemoryManager::ViewBufferTo(
    const std::shared_ptr<Buffer>& buf, const std::shared_ptr<MemoryManager>& to) {
    if (!to->is_cpu() || !buf) {
        return nullptr;
    }
    auto view = std::make_shared<Buffer>(buf->data(), buf->size(), to, buf,
                                         /*device_type_override=*/std::nullopt);
    return view;
}

bool IsMetalMemoryManager(const MemoryManager& mm) {
    return mm.device()->type_name() == std::string("arrow::metal::MetalDevice");
}
Result<std::shared_ptr<MetalMemoryManager>> AsMetalMemoryManager(
    const std::shared_ptr<MemoryManager>& mm) {
    if (!mm || !IsMetalMemoryManager(*mm)) {
        return Status::Invalid("Expected MetalMemoryManager");
    }
    return std::static_pointer_cast<MetalMemoryManager>(mm);
}

}  // namespace metal
}  // namespace arrow
