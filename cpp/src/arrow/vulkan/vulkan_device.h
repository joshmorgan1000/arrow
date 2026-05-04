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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "arrow/device.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/vulkan/visibility.h"

namespace arrow {
namespace vulkan {

class VulkanDevice;
class VulkanDeviceManager;
class VulkanMemoryManager;
/** --------------------------------------------------------------------------------------------- VulkanDeviceManager
 * @brief Process-wide singleton enumerating and caching VulkanDevice instances.
 *
 * Wraps `vkEnumeratePhysicalDevices` and applies the picker policy:
 * prefer integrated GPUs (`VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`) with
 * a HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL memory type, fall back
 * to CPU (`VK_PHYSICAL_DEVICE_TYPE_CPU`) for llvmpipe, refuse discrete
 * GPUs (project's "zero-copy or refuse" promise).
 *
 * Lifetime: lazy-constructed via std::call_once on first Instance() call.
 * Underlying VkInstance is shared with VulkanDevice via
 * `vulkan::internal::VulkanInstance::Get()`.
 */
class ARROW_VULKAN_EXPORT VulkanDeviceManager {
 public:
    ~VulkanDeviceManager();
    VulkanDeviceManager(const VulkanDeviceManager&) = delete;
    VulkanDeviceManager& operator=(const VulkanDeviceManager&) = delete;
    /// \brief Return the process-wide VulkanDeviceManager.
    static Result<VulkanDeviceManager*> Instance();
    /// \brief VulkanDevice at the given enumeration index in [0, num_devices()).
    Result<std::shared_ptr<VulkanDevice>> GetDevice(int device_number);
    /// \brief VulkanDevice keyed by 16-byte VkPhysicalDeviceProperties::pipelineCacheUUID.
    Result<std::shared_ptr<VulkanDevice>> GetDeviceByUuid(const uint8_t* uuid);
    /// \brief Number of acceptable Vulkan devices visible to the process.
    int num_devices() const;
 private:
    VulkanDeviceManager();
    class Impl;
    std::shared_ptr<Impl> impl_;
};
/** --------------------------------------------------------------------------------------------- VulkanDevice
 * @brief Device implementation for a single Vulkan physical device.
 *
 * Each VulkanDevice instance pins one VkPhysicalDevice + creates one
 * VkDevice (logical device) + one VkQueue from a queue family that
 * supports compute. Because the picker only accepts integrated/CPU
 * physical devices, `is_cpu()` returns true (the unified-memory trick
 * that worked for Metal); `device_type()` returns kVULKAN.
 */
class ARROW_VULKAN_EXPORT VulkanDevice : public Device {
 public:
    const char* type_name() const override;
    std::string ToString() const override;
    bool Equals(const Device& other) const override;
    DeviceAllocationType device_type() const override {
        return DeviceAllocationType::kVULKAN;
    }
    int64_t device_id() const override;
    /// \brief Default MemoryManager bound to this device.
    std::shared_ptr<MemoryManager> default_memory_manager() override;
    /// \brief Stable hash of `VkPhysicalDeviceProperties::pipelineCacheUUID`.
    int64_t uuid_hash() const;
    /// \brief Raw 16-byte UUID (for C-Data import lookups).
    const uint8_t* uuid() const;
    /// \brief Human-readable device name (e.g. "AMD Radeon 890M Graphics").
    std::string device_name() const;
    /// \brief Total RAM accessible via DEVICE_LOCAL heaps (bytes).
    int64_t total_memory() const;
    /// \brief Whether the device exposes VK_EXT_external_memory_host (Wrap path).
    bool supports_imported_host_memory() const;
    /// \brief Required alignment for vkImportMemoryHostPointerInfoEXT inputs.
    /// Returns 0 if VK_EXT_external_memory_host is not supported.
    int64_t imported_host_pointer_alignment() const;
    /// \brief Opaque VkPhysicalDevice handle. Cast via internal::FromOpaque<VkPhysicalDevice>.
    void* vk_physical_device() const;
    /// \brief Opaque VkDevice handle (logical device).
    void* vk_device() const;
    /// \brief Opaque VkQueue handle. Compute-capable queue from queue_family_index().
    void* vk_queue() const;
    /// \brief Queue family index used by vk_queue().
    uint32_t queue_family_index() const;
    /// \brief Memory type index for HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL allocations.
    uint32_t coherent_memory_type_index() const;
    /// \brief Return the system default VulkanDevice (first acceptable in the picker).
    static Result<std::shared_ptr<VulkanDevice>> Default();
    /// \brief Construct a VulkanDevice for a particular pipelineCacheUUID.
    static Result<std::shared_ptr<VulkanDevice>> Make(const uint8_t* uuid);
    /** ----------------------------------------------------------------------- Stream
     * @brief A wrapper around (VkQueue, VkCommandPool).
     *
     * MakeStream creates a fresh transient VkCommandPool bound to the
     * device's compute queue family. Synchronize() calls vkQueueWaitIdle
     * to drain all submitted work.
     */
    class ARROW_VULKAN_EXPORT Stream : public Device::Stream {
     public:
        ~Stream() override;
        const void* get_raw() const noexcept override { return stream_.get(); }
        Status WaitEvent(const Device::SyncEvent& event) override;
        Status Synchronize() const override;
        /// \brief Opaque VkCommandPool — cast via internal::FromOpaque<VkCommandPool>.
        void* vk_command_pool() const;
     protected:
        friend class VulkanDevice;
        Stream(std::shared_ptr<VulkanDevice> device, void* queue, void* command_pool,
               Device::Stream::release_fn_t release_fn);
     private:
        std::shared_ptr<VulkanDevice> device_;
        void* command_pool_;  ///< owned VkCommandPool, destroyed by ~Stream
    };
    Result<std::shared_ptr<Device::Stream>> MakeStream() override {
        return MakeStream(0);
    }
    Result<std::shared_ptr<Device::Stream>> MakeStream(unsigned int flags) override;
    Result<std::shared_ptr<Device::Stream>> WrapStream(
        void* device_stream, Stream::release_fn_t release_fn) override;
    /** ----------------------------------------------------------------------- SyncEvent
     * @brief A wrapper around a VkTimelineSemaphore (Vulkan 1.2 core).
     *
     * VkTimelineSemaphore is the closest analog to MTLSharedEvent: a
     * monotonically increasing 64-bit counter that the GPU signals via
     * vkQueueSubmit2 with VkSemaphoreSubmitInfo, and the CPU waits for
     * via vkWaitSemaphores. signal_value_ is std::atomic so concurrent
     * Record() and Wait() callers don't race on the read-modify-write.
     */
    class ARROW_VULKAN_EXPORT SyncEvent : public Device::SyncEvent {
     public:
        Status Wait() override;
        Status Record(const Device::Stream& stream) override;
        /// \brief Const-correct accessor for the underlying opaque
        /// VkSemaphore handle. Shadows the non-const upstream get_raw()
        /// so const-bound callers don't need to const_cast.
        const void* get_raw() const { return sync_event_.get(); }
        using Device::SyncEvent::get_raw;  ///< keep non-const overload
        /// \brief Counter value the underlying VkTimelineSemaphore must reach.
        uint64_t signal_value() const {
            return signal_value_.load(std::memory_order_acquire);
        }
     protected:
        friend class VulkanMemoryManager;
        SyncEvent(std::shared_ptr<VulkanDevice> device, void* semaphore,
                  Device::SyncEvent::release_fn_t release_event)
            : Device::SyncEvent(semaphore, release_event),
              device_{std::move(device)},
              signal_value_{0} {}
     private:
        std::shared_ptr<VulkanDevice> device_;
        /// Monotonic counter; atomic so concurrent Record/Wait is safe.
        std::atomic<uint64_t> signal_value_;
    };
 public:
    struct Impl;  ///< opaque, defined in vulkan_device.cc
 protected:
    explicit VulkanDevice(std::shared_ptr<Impl> impl);
    friend class VulkanDeviceManager;
    std::shared_ptr<Impl> impl_;
};
/** --------------------------------------------------------------------------------------------- IsVulkanDevice
 * @brief Whether a device instance is a VulkanDevice.
 */
ARROW_VULKAN_EXPORT bool IsVulkanDevice(const Device& device);
/** --------------------------------------------------------------------------------------------- AsVulkanDevice
 * @brief Cast a Device to VulkanDevice or fail with Status::Invalid.
 */
ARROW_VULKAN_EXPORT Result<std::shared_ptr<VulkanDevice>> AsVulkanDevice(
    const std::shared_ptr<Device>& device);

}  // namespace vulkan
}  // namespace arrow
