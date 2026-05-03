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

#include "arrow/metal/metal_device.h"
#include "arrow/metal/metal_memory.h"
#include "arrow/metal/metal_internal.hh"

#include <atomic>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "arrow/util/logging_internal.h"

namespace arrow {
namespace metal {
namespace {
constexpr const char* kMetalDeviceTypeName = "arrow::metal::MetalDevice";
}  // namespace
/** --------------------------------------------------------------------------------------------- MetalDevice::Impl
 * @brief pImpl for MetalDevice — owns the retained id<MTLDevice> handle.
 *
 * Storage is opaque (`void* mtl_device_opaque`) so the public header
 * stays clean of `<Metal/Metal.h>`. The bridging happens in this .mm
 * unit only.
 */
struct MetalDevice::Impl {
    void* mtl_device_opaque;       ///< retained __bridge_retained id<MTLDevice>
    int64_t registry_id;
    std::string name;
    int64_t total_memory;
    std::shared_ptr<MetalMemoryManager> default_mm;
    std::once_flag default_mm_init;
    Impl(void* dev, int64_t rid, std::string n, int64_t tm)
        : mtl_device_opaque{dev}, registry_id{rid}, name{std::move(n)}, total_memory{tm} {}
    ~Impl() { internal::ReleaseOpaque(mtl_device_opaque); }
};

MetalDevice::MetalDevice(std::shared_ptr<Impl> impl)
    : Device(/*is_cpu=*/true), impl_{std::move(impl)} {}
const char* MetalDevice::type_name() const { return kMetalDeviceTypeName; }
std::string MetalDevice::ToString() const {
    std::ostringstream ss;
    ss << "MetalDevice(name=\"" << impl_->name << "\", registry_id=" << impl_->registry_id
       << ", total_memory=" << impl_->total_memory << ")";
    return ss.str();
}
bool MetalDevice::Equals(const Device& other) const {
    if (other.type_name() != type_name()) {
        return false;
    }
    const auto& md = static_cast<const MetalDevice&>(other);
    return md.impl_->registry_id == impl_->registry_id;
}
int64_t MetalDevice::device_id() const { return impl_->registry_id; }
int64_t MetalDevice::registry_id() const { return impl_->registry_id; }
std::string MetalDevice::device_name() const { return impl_->name; }
int64_t MetalDevice::total_memory() const { return impl_->total_memory; }
void* MetalDevice::mtl_device() const { return impl_->mtl_device_opaque; }
std::shared_ptr<MemoryManager> MetalDevice::default_memory_manager() {
    std::call_once(impl_->default_mm_init, [this]() {
        impl_->default_mm =
            MetalMemoryManager::Make(std::static_pointer_cast<MetalDevice>(shared_from_this()));
    });
    return impl_->default_mm;
}
/** --------------------------------------------------------------------------------------------- MakeImpl
 * @brief Build a MetalDevice::Impl from an unretained id<MTLDevice>
 *
 * Helper used by `MetalDevice::Default` and `MetalDeviceManager`. Takes
 * a borrowed id and retains it into the opaque slot so the Impl owns
 * the only strong reference; ARC inside this function balances books.
 */
static std::shared_ptr<MetalDevice::Impl> MakeImpl(id<MTLDevice> dev) {
    auto opaque = internal::ToOpaqueRetained(dev);
    int64_t rid = static_cast<int64_t>([dev registryID]);
    NSString* name_ns = [dev name];
    std::string name = name_ns ? std::string([name_ns UTF8String]) : std::string{};
    int64_t total = static_cast<int64_t>([dev recommendedMaxWorkingSetSize]);
    return std::make_shared<MetalDevice::Impl>(opaque, rid, std::move(name), total);
}

Result<std::shared_ptr<MetalDevice>> MetalDevice::Default() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            return Status::Invalid("MTLCreateSystemDefaultDevice() returned nil");
        }
        return std::shared_ptr<MetalDevice>(new MetalDevice(MakeImpl(dev)));
    }
}

Result<std::shared_ptr<MetalDevice>> MetalDevice::Make(int64_t registry_id) {
    ARROW_ASSIGN_OR_RAISE(auto* mgr, MetalDeviceManager::Instance());
    return mgr->GetDeviceByRegistryId(registry_id);
}

bool IsMetalDevice(const Device& device) {
    return device.type_name() == kMetalDeviceTypeName;
}
Result<std::shared_ptr<MetalDevice>> AsMetalDevice(const std::shared_ptr<Device>& device) {
    if (!device || !IsMetalDevice(*device)) {
        return Status::Invalid("Expected MetalDevice");
    }
    return std::static_pointer_cast<MetalDevice>(device);
}
/** --------------------------------------------------------------------------------------------- MetalDeviceManager::Impl
 * @brief Singleton state — list of MTLDevice handles + cached MetalDevice wrappers.
 */
class MetalDeviceManager::Impl {
 public:
    Impl() {
        @autoreleasepool {
#if TARGET_OS_OSX
            NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
            for (id<MTLDevice> d in devs) {
                devices_.push_back(internal::ToOpaqueRetained(d));
                registry_ids_.push_back(static_cast<int64_t>([d registryID]));
            }
#else
            id<MTLDevice> d = MTLCreateSystemDefaultDevice();
            if (d != nil) {
                devices_.push_back(internal::ToOpaqueRetained(d));
                registry_ids_.push_back(static_cast<int64_t>([d registryID]));
            }
#endif
        }
    }
    ~Impl() {
        for (void* o : devices_) {
            internal::ReleaseOpaque(o);
        }
    }
    int num_devices() const { return static_cast<int>(devices_.size()); }
    Result<std::shared_ptr<MetalDevice>> GetDevice(int idx) {
        if (idx < 0 || idx >= num_devices()) {
            return Status::Invalid("MetalDevice index ", idx, " out of range [0,",
                                   num_devices(), ")");
        }
        return ResolveCached(idx);
    }
    Result<std::shared_ptr<MetalDevice>> GetDeviceByRegistryId(int64_t rid) {
        for (int i = 0; i < num_devices(); ++i) {
            if (registry_ids_[i] == rid) {
                return ResolveCached(i);
            }
        }
        return Status::Invalid("No Metal device with registry_id ", rid);
    }
 private:
    Result<std::shared_ptr<MetalDevice>> ResolveCached(int idx) {
        std::lock_guard<std::mutex> lk{cache_mu_};
        auto it = cache_.find(idx);
        if (it != cache_.end()) {
            return it->second;
        }
        @autoreleasepool {
            id<MTLDevice> dev = internal::FromOpaqueBorrowed<id<MTLDevice>>(devices_[idx]);
            auto impl = MakeImpl(dev);
            std::shared_ptr<MetalDevice> md{new MetalDevice(std::move(impl))};
            cache_.emplace(idx, md);
            return md;
        }
    }
    std::vector<void*> devices_;       ///< retained id<MTLDevice>
    std::vector<int64_t> registry_ids_;
    std::mutex cache_mu_;
    std::unordered_map<int, std::shared_ptr<MetalDevice>> cache_;
};

MetalDeviceManager::MetalDeviceManager() : impl_{std::make_shared<Impl>()} {}
MetalDeviceManager::~MetalDeviceManager() = default;

Result<MetalDeviceManager*> MetalDeviceManager::Instance() {
    static std::unique_ptr<MetalDeviceManager> instance;
    static std::once_flag init;
    std::call_once(init, []() {
        instance.reset(new MetalDeviceManager());
    });
    if (instance->num_devices() == 0) {
        return Status::Invalid("No Metal devices available on this system");
    }
    return instance.get();
}
int MetalDeviceManager::num_devices() const { return impl_->num_devices(); }
Result<std::shared_ptr<MetalDevice>> MetalDeviceManager::GetDevice(int idx) {
    return impl_->GetDevice(idx);
}
Result<std::shared_ptr<MetalDevice>> MetalDeviceManager::GetDeviceByRegistryId(
    int64_t rid) {
    return impl_->GetDeviceByRegistryId(rid);
}
/** --------------------------------------------------------------------------------------------- MetalDevice::Stream
 * @brief Wraps id<MTLCommandQueue>. Synchronize commits an empty command
 *        buffer with `waitUntilCompleted`, which drains all in-flight
 *        encoded work targeting this queue.
 */
Status MetalDevice::Stream::Synchronize() const {
    @autoreleasepool {
        id<MTLCommandQueue> queue =
            internal::FromOpaqueBorrowed<id<MTLCommandQueue>>(stream_.get());
        if (queue == nil) {
            return Status::Invalid("MetalDevice::Stream is null");
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
        return Status::OK();
    }
}
Status MetalDevice::Stream::WaitEvent(const Device::SyncEvent& event) {
    @autoreleasepool {
        id<MTLCommandQueue> queue =
            internal::FromOpaqueBorrowed<id<MTLCommandQueue>>(stream_.get());
        auto& mev = const_cast<MetalDevice::SyncEvent&>(
            static_cast<const MetalDevice::SyncEvent&>(event));
        id<MTLSharedEvent> shared =
            internal::FromOpaqueBorrowed<id<MTLSharedEvent>>(mev.get_raw());
        if (queue == nil || shared == nil) {
            return Status::Invalid("MetalDevice::Stream::WaitEvent received nil handles");
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [cb encodeWaitForEvent:shared value:mev.signal_value()];
        [cb commit];
        return Status::OK();
    }
}

Result<std::shared_ptr<Device::Stream>> MetalDevice::MakeStream(unsigned int /*flags*/) {
    @autoreleasepool {
        id<MTLDevice> dev = internal::FromOpaqueBorrowed<id<MTLDevice>>(impl_->mtl_device_opaque);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (queue == nil) {
            return Status::Invalid("[device newCommandQueue] returned nil");
        }
        void* opaque = internal::ToOpaqueRetained(queue);
        auto release = [](void* p) { internal::ReleaseOpaque(p); };
        return std::shared_ptr<Device::Stream>(new Stream(
            std::static_pointer_cast<MetalDevice>(shared_from_this()), opaque, release));
    }
}
Result<std::shared_ptr<Device::Stream>> MetalDevice::WrapStream(
    void* device_stream, Stream::release_fn_t release_fn) {
    if (device_stream == nullptr) {
        return Status::Invalid("MetalDevice::WrapStream received nullptr stream");
    }
    return std::shared_ptr<Device::Stream>(new Stream(
        std::static_pointer_cast<MetalDevice>(shared_from_this()), device_stream, release_fn));
}
/** --------------------------------------------------------------------------------------------- MetalDevice::SyncEvent
 * @brief Wraps id<MTLSharedEvent> + monotonic counter
 *
 * Record() bumps the counter, encodes a signal of that value on a fresh
 * command buffer and commits it. Wait() blocks until the underlying
 * MTLSharedEvent reaches the recorded value.
 */
Status MetalDevice::SyncEvent::Record(const Device::Stream& stream) {
    @autoreleasepool {
        id<MTLSharedEvent> shared =
            internal::FromOpaqueBorrowed<id<MTLSharedEvent>>(sync_event_.get());
        id<MTLCommandQueue> queue =
            internal::FromOpaqueBorrowed<id<MTLCommandQueue>>(
                const_cast<void*>(stream.get_raw()));
        if (shared == nil || queue == nil) {
            return Status::Invalid("MetalDevice::SyncEvent::Record received nil handles");
        }
        signal_value_ += 1;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [cb encodeSignalEvent:shared value:signal_value_];
        [cb commit];
        return Status::OK();
    }
}
Status MetalDevice::SyncEvent::Wait() {
    @autoreleasepool {
        id<MTLSharedEvent> shared =
            internal::FromOpaqueBorrowed<id<MTLSharedEvent>>(sync_event_.get());
        if (shared == nil) {
            return Status::Invalid("MetalDevice::SyncEvent::Wait received nil event");
        }
        // Spin-wait via [event signaledValue] is wasteful; use the listener API.
        // Returns YES when value reached, NO on timeout. timeoutMS=0 means
        // "wait forever" per Apple docs.
        BOOL ok = [shared waitUntilSignaledValue:signal_value_ timeoutMS:UINT64_MAX];
        if (!ok) {
            return Status::IOError("MetalDevice::SyncEvent::Wait timed out");
        }
        return Status::OK();
    }
}

}  // namespace metal
}  // namespace arrow
