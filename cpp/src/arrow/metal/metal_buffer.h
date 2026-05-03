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
#include "arrow/metal/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace metal {

class MetalDevice;
class MetalMemoryManager;

/// \class MetalBuffer
/// \brief An Arrow buffer whose storage is an MTLBuffer (Metal-coherent)
///
/// On Apple Silicon CPU and GPU share physical RAM. A MetalBuffer wraps
/// an `id<MTLBuffer>` allocated with MTLResourceStorageModeShared, and
/// exposes `data()` as the corresponding `[mtl_buffer contents]` pointer.
/// Reads and writes through that pointer are immediately visible to GPU
/// kernels operating on the same MTLBuffer (subject to standard Metal
/// hazard tracking and CPU/GPU synchronisation; see
/// `MetalDevice::SyncEvent`).
///
/// `is_cpu()` returns true and `device_type()` returns kMETAL — the
/// buffer is simultaneously CPU and Metal addressable. This is the
/// fundamental departure from CudaBuffer, which forces a host/device
/// split that does not exist on Apple Silicon.
class ARROW_METAL_EXPORT MetalBuffer : public Buffer {
 public:
    /** ----------------------------------------------------------------------------- Constructor
     * @brief Take ownership of an MTLBuffer allocated with MTLResourceStorageModeShared.
     *
     * @param mtl_buffer    opaque retained handle to id<MTLBuffer> (from
     *                      MetalMemoryManager::AllocateBuffer or internal::ToOpaqueRetained).
     * @param mm            the MetalMemoryManager that produced the buffer.
     * @param parent        optional source buffer pinned for the lifetime of this MetalBuffer;
     *                      used by ViewBufferFrom to keep the underlying CPU pages alive when
     *                      wrapping host memory via newBufferWithBytesNoCopy.
     * @param logical_size  optional Arrow-visible size override. When negative (default),
     *                      size() returns `[mtl_buffer length]`. When non-negative, size()
     *                      returns this value — useful for zero-length Arrow buffers backed
     *                      by a 1-byte MTLBuffer (Apple rejects 0-length newBufferWithLength).
     *                      Must be <= the underlying MTLBuffer length.
     */
    MetalBuffer(void* mtl_buffer, std::shared_ptr<MetalMemoryManager> mm,
                std::shared_ptr<Buffer> parent = nullptr,
                int64_t logical_size = -1);
    /** ----------------------------------------------------------------------------- Slice
     * @brief Construct a sub-range view into a parent MetalBuffer.
     *
     * The slice carries a shared_ptr to the parent so the underlying MTLBuffer outlives
     * every view. data() points at parent->data()+offset, still GPU-addressable because
     * the shared MTLBuffer is unchanged.
     *
     * @param parent  the buffer whose storage is shared.
     * @param offset  byte offset into the parent.
     * @param size    length of the slice.
     */
    MetalBuffer(const std::shared_ptr<MetalBuffer>& parent, int64_t offset, int64_t size);
    ~MetalBuffer() override;
    /// \brief Release callback signature for `MetalBuffer::Wrap`.
    ///
    /// Invoked exactly once when the underlying MTLBuffer's last retain drops. `host_ptr`
    /// and `size` are the values originally passed to `Wrap`; `user_data` is the pointer
    /// the caller supplied alongside `release_fn` (use to thread state without capturing).
    using ReleaseFn = void (*)(void* host_ptr, int64_t size, void* user_data);
    /** ----------------------------------------------------------------------------- FromBuffer
     * @brief Downcast a generic Buffer to MetalBuffer.
     * @param buffer  the buffer to downcast.
     * @return        MetalBuffer or Status::Invalid if the buffer is not Metal-backed.
     */
    static Result<std::shared_ptr<MetalBuffer>> FromBuffer(std::shared_ptr<Buffer> buffer);
    /** ----------------------------------------------------------------------------- Wrap
     * @brief Wrap externally-owned host memory as a Metal-coherent buffer.
     *
     * Calls `[device newBufferWithBytesNoCopy:host_ptr length:size
     * options:MTLResourceStorageModeShared deallocator:cb]` where `cb` invokes the supplied
     * `release_fn(host_ptr, size, user_data)` exactly once when the underlying MTLBuffer is
     * finally released.
     *
     * **Ownership contract — read carefully**:
     * - The returned MetalBuffer owns the MTLBuffer; the MTLBuffer's `deallocator` block
     *   owns the host memory.
     * - When the MetalBuffer is destroyed, the MTLBuffer is released; when the last retain
     *   drops, Apple invokes `release_fn`.
     * - Do NOT pin `host_ptr`'s parent buffer via Arrow's `parent_` slot — that would
     *   create two paths to free the same memory. Either Apple's deallocator owns it
     *   (use Wrap) or Arrow's parent_ owns it (use ViewBuffer with a parent Arrow
     *   Buffer), never both.
     *
     * `host_ptr` and `size` MUST be page-aligned (`getpagesize()`). Misaligned inputs
     * return Status::Invalid; the caller can fall back to AllocateBuffer + memcpy.
     *
     * @param mm          Metal MemoryManager that owns the device.
     * @param host_ptr    page-aligned base of the host allocation.
     * @param size        page-multiple length in bytes.
     * @param release_fn  free function called once at MTLBuffer release. Pass nullptr to
     *                    indicate the host memory is externally managed and needs no
     *                    callback.
     * @param user_data   opaque pointer forwarded to release_fn.
     */
    static Result<std::shared_ptr<MetalBuffer>> Wrap(
        std::shared_ptr<MetalMemoryManager> mm, void* host_ptr, int64_t size,
        ReleaseFn release_fn, void* user_data = nullptr);
    /** ----------------------------------------------------------------------------- mtl_buffer
     * @brief Return the opaque (borrowed) `id<MTLBuffer>` handle, or nullptr for slices.
     *
     * Cast back via `internal::FromOpaqueBorrowed<id<MTLBuffer>>` from .mm code. The
     * returned pointer is owned by this MetalBuffer; do NOT release it.
     *
     * **Slice views**: returns `nullptr` because the MTLBuffer is held by `parent()`.
     * Use `root_mtl_buffer()` paired with `offset()` when encoding GPU work against a
     * slice — that helper walks the parent_ chain so callers don't have to.
     *
     * @return opaque handle suitable for `__bridge` cast to id<MTLBuffer>, or nullptr.
     */
    void* mtl_buffer() const { return mtl_buffer_; }
    /** ----------------------------------------------------------------------------- root_mtl_buffer
     * @brief Walk parent_ chain and return the root MTLBuffer for any view.
     *
     * For root buffers this is identical to `mtl_buffer()`. For slices, recurses
     * through `parent()` until it finds the buffer that owns the MTLBuffer.
     *
     * Pair with `offset()` when encoding GPU work:
     *
     * ```cpp
     * id<MTLBuffer> root = (__bridge id<MTLBuffer>)buf->root_mtl_buffer();
     * [encoder setBuffer:root offset:buf->offset() atIndex:N];
     * ```
     */
    void* root_mtl_buffer() const;
    /** ----------------------------------------------------------------------------- offset
     * @brief Byte offset of this view within the root MTLBuffer.
     *
     * Slice views report a non-zero offset (slice offsets compose, so a slice-of-a-slice
     * reports the absolute offset within the root). Root buffers report 0. GPU encoders
     * consuming this buffer should pass `offset()` directly as the encoder offset.
     */
    int64_t offset() const { return offset_; }
 protected:
    /// \brief Opaque retained handle to the underlying MTLBuffer (root only).
    /// For slice views, mtl_buffer_ is nullptr and the parent owns the storage.
    void* mtl_buffer_;
    /// \brief Byte offset into the root MTLBuffer (already composed for nested slices).
    int64_t offset_;
};
/** --------------------------------------------------------------------------------------------- IsMetalBuffer
 * @brief Whether a Buffer's storage is backed by Metal
 * @param buffer the buffer to test
 * @return true iff buffer is a MetalBuffer
 */
ARROW_METAL_EXPORT bool IsMetalBuffer(const Buffer& buffer);

}  // namespace metal
}  // namespace arrow
