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
#include "arrow/metal/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace metal {

class MetalDevice;
class MetalMemoryManager;
class MetalDeviceManager;
/** --------------------------------------------------------------------------------------------- MetalDeviceManager
 * @brief Process-wide singleton managing Metal devices (Apple Silicon)
 *
 * Wraps `MTLCopyAllDevices` (macOS) / `MTLCreateSystemDefaultDevice`
 * (iOS) and caches MetalDevice instances by `registryID`. There is
 * usually one MTLDevice per Apple Silicon system; Mac Pro hosts can
 * present multiple.
 */
class ARROW_METAL_EXPORT MetalDeviceManager {
 public:
    /// \brief Return the process-wide MetalDeviceManager
    static Result<MetalDeviceManager*> Instance();
    /// \brief Return the MetalDevice at the given index in [0, num_devices())
    Result<std::shared_ptr<MetalDevice>> GetDevice(int device_number);
    /// \brief Return the MetalDevice keyed by Apple registryID
    Result<std::shared_ptr<MetalDevice>> GetDeviceByRegistryId(int64_t registry_id);
    /// \brief Number of Metal devices visible to the process
    int num_devices() const;
    ~MetalDeviceManager();
    MetalDeviceManager(const MetalDeviceManager&) = delete;
    MetalDeviceManager& operator=(const MetalDeviceManager&) = delete;
 private:
    MetalDeviceManager();
    class Impl;
    std::shared_ptr<Impl> impl_;
};
/** --------------------------------------------------------------------------------------------- MetalDevice
 * @brief Device implementation for a single Apple Metal device
 *
 * Each MetalDevice instance pins one `id<MTLDevice>` for its lifetime.
 * Because Apple Silicon has unified memory, `is_cpu()` returns true
 * even though `device_type()` returns kMETAL — buffers allocated
 * through this device's MemoryManager are simultaneously CPU- and
 * GPU-addressable.
 */
class ARROW_METAL_EXPORT MetalDevice : public Device {
 public:
    const char* type_name() const override;
    std::string ToString() const override;
    bool Equals(const Device& other) const override;
    DeviceAllocationType device_type() const override {
        return DeviceAllocationType::kMETAL;
    }
    int64_t device_id() const override;
    /// \brief Default MemoryManager bound to this device
    std::shared_ptr<MemoryManager> default_memory_manager() override;
    /// \brief Apple registry ID for the underlying MTLDevice
    int64_t registry_id() const;
    /// \brief Human-readable device name (e.g. "Apple M4 Pro")
    std::string device_name() const;
    /// \brief Total RAM accessible to this device (bytes)
    int64_t total_memory() const;
    /** ----------------------------------------------------------------------- mtl_device
     * @brief Return the opaque (borrowed) `id<MTLDevice>` handle
     *
     * Cast via `internal::FromOpaqueBorrowed<id<MTLDevice>>`. The handle
     * is owned by this MetalDevice; do NOT release.
     */
    void* mtl_device() const;
    /** ----------------------------------------------------------------------- Default
     * @brief Return the system default MetalDevice (MTLCreateSystemDefaultDevice)
     */
    static Result<std::shared_ptr<MetalDevice>> Default();
    /// \brief Construct a MetalDevice for a particular registry ID
    static Result<std::shared_ptr<MetalDevice>> Make(int64_t registry_id);
    /** ----------------------------------------------------------------------- Stream
     * @brief A wrapper around `id<MTLCommandQueue>`
     *
     * MakeStream creates a fresh queue via `[device newCommandQueue]`.
     * WrapStream adopts an externally constructed queue with a caller
     * supplied release function (pass nullptr to indicate the caller
     * keeps ownership of the queue).
     */
    class ARROW_METAL_EXPORT Stream : public Device::Stream {
     public:
        ~Stream() = default;
        const void* get_raw() const noexcept override { return stream_.get(); }
        Status WaitEvent(const Device::SyncEvent& event) override;
        Status Synchronize() const override;
     protected:
        friend class MetalDevice;
        Stream(std::shared_ptr<MetalDevice> device, void* queue,
               Device::Stream::release_fn_t release_fn)
            : Device::Stream(queue, release_fn), device_{std::move(device)} {}
     private:
        std::shared_ptr<MetalDevice> device_;
    };
    Result<std::shared_ptr<Device::Stream>> MakeStream() override {
        return MakeStream(0);
    }
    Result<std::shared_ptr<Device::Stream>> MakeStream(unsigned int flags) override;
    Result<std::shared_ptr<Device::Stream>> WrapStream(
        void* device_stream, Stream::release_fn_t release_fn) override;
    /** ----------------------------------------------------------------------- SyncEvent
     * @brief A wrapper around `id<MTLSharedEvent>` plus a counter value
     *
     * Apple's MTLSharedEvent is the closest analogue to CUevent: a
     * monotonically increasing 64-bit counter that the GPU signals via
     * `[encoder encodeSignalEvent:value:]` and the CPU waits for via
     * `[event waitUntilSignaledValue:]`.
     */
    class ARROW_METAL_EXPORT SyncEvent : public Device::SyncEvent {
     public:
        Status Wait() override;
        Status Record(const Device::Stream& stream) override;
        /// \brief Const-correct accessor for the underlying opaque
        /// `id<MTLSharedEvent>` handle. Shadows the non-const
        /// `Device::SyncEvent::get_raw()` (upstream omits the const
        /// overload) so `WaitEvent(const Device::SyncEvent&)` callers
        /// don't need to const_cast their argument.
        const void* get_raw() const { return sync_event_.get(); }
        using Device::SyncEvent::get_raw;  ///< keep non-const overload
        /// \brief Counter value the underlying MTLSharedEvent must reach
        uint64_t signal_value() const {
            return signal_value_.load(std::memory_order_acquire);
        }
     protected:
        friend class MetalMemoryManager;
        SyncEvent(std::shared_ptr<MetalDevice> device, void* event,
                  Device::SyncEvent::release_fn_t release_event)
            : Device::SyncEvent(event, release_event),
              device_{std::move(device)},
              signal_value_{0} {}
     private:
        std::shared_ptr<MetalDevice> device_;
        /// Monotonically increasing counter. Atomic so concurrent
        /// `Record()` and `Wait()` callers don't race on the read /
        /// modify / write sequence in `Record`.
        std::atomic<uint64_t> signal_value_;
    };
 public:
    struct Impl;  ///< opaque, defined in metal_device.mm
 protected:
    explicit MetalDevice(std::shared_ptr<Impl> impl);
    friend class MetalDeviceManager;
    std::shared_ptr<Impl> impl_;
};
/** --------------------------------------------------------------------------------------------- IsMetalDevice
 * @brief Whether a device instance is a MetalDevice
 */
ARROW_METAL_EXPORT bool IsMetalDevice(const Device& device);
/** --------------------------------------------------------------------------------------------- AsMetalDevice
 * @brief Cast a Device to MetalDevice or fail with Status::Invalid
 */
ARROW_METAL_EXPORT Result<std::shared_ptr<MetalDevice>> AsMetalDevice(
    const std::shared_ptr<Device>& device);

}  // namespace metal
}  // namespace arrow
