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
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/vulkan/visibility.h"

namespace arrow {
namespace vulkan {

class VulkanDevice;
class VulkanMemoryManager;
/** --------------------------------------------------------------------------------------------- VulkanBuffer
 * @class VulkanBuffer
 * @brief An Arrow buffer whose storage is a host-coherent Vulkan buffer.
 *
 * On integrated GPUs (AMD APU, Intel iGPU, Mali, Adreno, MoltenVK on
 * Apple Silicon, llvmpipe), `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL`
 * memory exposes a single physical region accessible to both CPU and
 * GPU. A VulkanBuffer wraps a (VkBuffer, VkDeviceMemory) pair allocated
 * from such a memory type and exposes the persistent `vkMapMemory`
 * pointer as Arrow's `data()`. Reads and writes through that pointer
 * are immediately visible to GPU compute work targeting the same
 * VkBuffer (subject to standard Vulkan synchronization barriers).
 *
 * `is_cpu()` returns true and `device_type()` returns kVULKAN — the
 * buffer is simultaneously CPU and GPU addressable. Discrete GPUs
 * without ReBAR are out of scope; VulkanDeviceManager refuses them.
 */
class ARROW_VULKAN_EXPORT VulkanBuffer : public Buffer {
 public:
    /** ----------------------------------------------------------------------------- Constructor
     * @brief Take ownership of a (VkBuffer, VkDeviceMemory) pair allocated by
     *        VulkanMemoryManager::AllocateBuffer.
     *
     * The constructor calls vkMapMemory once and stores the resulting host
     * pointer; the destructor unmaps and frees both objects in the correct
     * order. Slice views (the second constructor) share storage with their
     * parent and do not own resources.
     *
     * @param vk_buffer     opaque VkBuffer handle (cast via internal::FromOpaque<VkBuffer>).
     * @param vk_memory     opaque VkDeviceMemory handle bound to vk_buffer.
     * @param mapped_ptr    persistent vkMapMemory result; data() returns this.
     * @param mm            the VulkanMemoryManager that produced the pair.
     * @param parent        optional source buffer pinned for the lifetime of
     *                      this VulkanBuffer (used by ViewBufferFrom to keep
     *                      the underlying CPU pages alive when wrapping host
     *                      memory via VK_EXT_external_memory_host).
     * @param logical_size  optional Arrow-visible size override. When negative
     *                      (default), size() returns the full mapped region.
     *                      When non-negative, size() returns this value —
     *                      used for zero-length Arrow buffers and for sub-page
     *                      requests that get rounded up by Vulkan alignment.
     */
    VulkanBuffer(void* vk_buffer, void* vk_memory, void* mapped_ptr,
                 std::shared_ptr<VulkanMemoryManager> mm,
                 std::shared_ptr<Buffer> parent = nullptr,
                 int64_t logical_size = -1);
    /** ----------------------------------------------------------------------------- Slice
     * @brief Construct a sub-range view into a parent VulkanBuffer.
     *
     * The slice carries a shared_ptr to the parent so the underlying VkBuffer
     * outlives every view. data() points at parent->data()+offset, still
     * GPU-addressable because the shared VkBuffer is unchanged. Slice
     * destruction does NOT touch Vulkan resources — the parent owns them.
     *
     * @param parent  the buffer whose storage is shared.
     * @param offset  byte offset into the parent.
     * @param size    length of the slice.
     */
    VulkanBuffer(const std::shared_ptr<VulkanBuffer>& parent, int64_t offset,
                 int64_t size);
    ~VulkanBuffer() override;
    /// \brief Release callback signature for `VulkanBuffer::Wrap`.
    ///
    /// Invoked exactly once when the underlying VkDeviceMemory is freed.
    /// `host_ptr` and `size` are the values originally passed to Wrap;
    /// `user_data` is the pointer the caller supplied alongside `release_fn`.
    using ReleaseFn = void (*)(void* host_ptr, int64_t size, void* user_data);
    /** ----------------------------------------------------------------------------- FromBuffer
     * @brief Downcast a generic Buffer to VulkanBuffer.
     * @param buffer  the buffer to downcast.
     * @return        VulkanBuffer or Status::Invalid if the buffer is not Vulkan-backed.
     */
    static Result<std::shared_ptr<VulkanBuffer>> FromBuffer(std::shared_ptr<Buffer> buffer);
    /** ----------------------------------------------------------------------------- Wrap
     * @brief Wrap externally-owned host memory as a Vulkan-coherent buffer.
     *
     * Calls `vkAllocateMemory` with `VkImportMemoryHostPointerInfoEXT` chain
     * (handleType = HOST_ALLOCATION_BIT_EXT), creates a VkBuffer of the right
     * size, and binds them. The caller-supplied `release_fn` is invoked
     * exactly once when this VulkanBuffer is destroyed, after vkFreeMemory
     * (which does NOT touch host_ptr).
     *
     * **Ownership contract — read carefully**:
     * - The returned VulkanBuffer owns the (VkBuffer, VkDeviceMemory) pair;
     *   the caller-supplied `release_fn` owns the host memory.
     * - When the VulkanBuffer is destroyed: vkUnmapMemory, vkDestroyBuffer,
     *   vkFreeMemory, then release_fn(host_ptr, size, user_data) — exactly
     *   once.
     * - Do NOT pin `host_ptr`'s parent buffer via Arrow's `parent_` slot —
     *   that would create two paths to free the same memory. Either
     *   release_fn owns it (use Wrap) or Arrow's parent_ owns it (use
     *   ViewBuffer with a parent Arrow Buffer), never both.
     *
     * `host_ptr` and `size` MUST be aligned to the device's
     * `minImportedHostPointerAlignment` (page size on AMD/Intel iGPUs).
     * Misaligned inputs return Status::Invalid; the caller can fall back
     * to AllocateBuffer + memcpy.
     *
     * Requires the device to expose `VK_EXT_external_memory_host`. If the
     * extension is absent, returns Status::NotImplemented.
     *
     * @param mm          Vulkan MemoryManager that owns the device.
     * @param host_ptr    aligned base of the host allocation.
     * @param size        aligned length in bytes.
     * @param release_fn  free function called once after vkFreeMemory.
     *                    Pass nullptr if host memory is externally managed.
     * @param user_data   opaque pointer forwarded to release_fn.
     */
    static Result<std::shared_ptr<VulkanBuffer>> Wrap(
        std::shared_ptr<VulkanMemoryManager> mm, void* host_ptr, int64_t size,
        ReleaseFn release_fn, void* user_data = nullptr,
        std::shared_ptr<Buffer> parent = nullptr);
    /** ----------------------------------------------------------------------------- vk_buffer
     * @brief Return the opaque (borrowed) VkBuffer handle, or nullptr for slices.
     *
     * Cast via `internal::FromOpaque<VkBuffer>` from .cc code. Owned by this
     * VulkanBuffer; do NOT destroy.
     *
     * **Slice views**: returns nullptr because the VkBuffer is held by parent().
     * Use `root_vk_buffer()` paired with `offset()` when encoding GPU work.
     */
    void* vk_buffer() const { return vk_buffer_; }
    /** ----------------------------------------------------------------------------- vk_memory
     * @brief Return the opaque (borrowed) VkDeviceMemory handle, or nullptr for slices.
     */
    void* vk_memory() const { return vk_memory_; }
    /** ----------------------------------------------------------------------------- root_vk_buffer
     * @brief Walk parent_ chain and return the root VkBuffer for any view.
     *
     * For root buffers this is identical to vk_buffer(). For slices, recurses
     * through parent() until it finds the buffer that owns the VkBuffer.
     *
     * Pair with offset() when encoding GPU work:
     *
     * ```cpp
     * VkBuffer root = (VkBuffer) buf->root_vk_buffer();
     * vkCmdBindDescriptorBuffersEXT(...);  // or wherever buf->offset() goes
     * ```
     */
    void* root_vk_buffer() const;
    /** ----------------------------------------------------------------------------- offset
     * @brief Byte offset of this view within the root VkBuffer.
     *
     * Slice views report a non-zero offset (slice offsets compose, so a
     * slice-of-a-slice reports the absolute offset within the root). Root
     * buffers report 0. GPU encoders consuming this buffer should pass
     * offset() directly as the VkDescriptorBufferInfo offset.
     */
    int64_t offset() const { return offset_; }
 protected:
    /// Opaque VkBuffer (root only). Slice views are nullptr; parent_ owns it.
    void* vk_buffer_;
    /// Opaque VkDeviceMemory (root only). Slice views are nullptr.
    void* vk_memory_;
    /// Byte offset into the root VkBuffer (already composed for nested slices).
    int64_t offset_;
    /// Optional release callback invoked at destruction (Wrap path only).
    ReleaseFn release_fn_;
    /// Opaque user_data forwarded to release_fn_.
    void* user_data_;
};
/** --------------------------------------------------------------------------------------------- IsVulkanBuffer
 * @brief Whether a Buffer's storage is backed by Vulkan.
 */
ARROW_VULKAN_EXPORT bool IsVulkanBuffer(const Buffer& buffer);

}  // namespace vulkan
}  // namespace arrow
