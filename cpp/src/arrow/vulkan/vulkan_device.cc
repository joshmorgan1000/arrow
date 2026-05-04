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

#include "arrow/vulkan/vulkan_device.h"
#include "arrow/vulkan/vulkan_internal.hh"
#include "arrow/vulkan/vulkan_memory.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "arrow/util/logging_internal.h"

namespace arrow {
namespace vulkan {
namespace {
constexpr const char* kVulkanDeviceTypeName = "arrow::vulkan::VulkanDevice";
constexpr uint32_t kRequiredApiVersion = VK_API_VERSION_1_2;
}  // namespace
namespace internal {
/** --------------------------------------------------------------------------------------------- VulkanInstance
 * @brief Process-wide VkInstance singleton.
 *
 * One VkInstance per process — created lazily, destroyed at exit.
 * Validation layers are opt-in via `ARROW_VULKAN_VALIDATION=1` env var
 * (off by default; enabling on a system without the validation layer
 * package logs a warning and proceeds without).
 */
class VulkanInstance {
 public:
    static Result<VkInstance> Get() {
        static VulkanInstance instance;
        if (instance.handle_ == VK_NULL_HANDLE) {
            return Status::Invalid(
                "VulkanInstance: vkCreateInstance failed during static init "
                "(check loader and ICD installation)");
        }
        return instance.handle_;
    }
 private:
    VulkanInstance() {
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.pApplicationName = "Apache Arrow (arrow_vulkan)";
        ai.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        ai.pEngineName = "Apache Arrow";
        ai.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        ai.apiVersion = kRequiredApiVersion;

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;

        const char* val_layer = "VK_LAYER_KHRONOS_validation";
        std::array<const char*, 1> layers = {val_layer};
        const char* env = std::getenv("ARROW_VULKAN_VALIDATION");
        if (env != nullptr && env[0] == '1') {
            ici.enabledLayerCount = 1;
            ici.ppEnabledLayerNames = layers.data();
        }

        // MoltenVK on macOS requires the portability enumeration flag plus
        // the corresponding extension. The flag is harmless on Linux.
#ifdef __APPLE__
        const char* portability_ext = "VK_KHR_portability_enumeration";
        ici.flags = 0x00000001;  // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        std::array<const char*, 1> exts = {portability_ext};
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = exts.data();
#endif

        VkResult r = vkCreateInstance(&ici, nullptr, &handle_);
        if (r != VK_SUCCESS) {
            // Validation layer requested but not installed → retry without.
            if (r == VK_ERROR_LAYER_NOT_PRESENT) {
                ici.enabledLayerCount = 0;
                ici.ppEnabledLayerNames = nullptr;
                r = vkCreateInstance(&ici, nullptr, &handle_);
            }
        }
        if (r != VK_SUCCESS) {
            ARROW_LOG(WARNING) << "vkCreateInstance failed: "
                               << VkResultToString(r);
            handle_ = VK_NULL_HANDLE;
        }
    }
    ~VulkanInstance() {
        if (handle_ != VK_NULL_HANDLE) {
            vkDestroyInstance(handle_, nullptr);
        }
    }
    VulkanInstance(const VulkanInstance&) = delete;
    VulkanInstance& operator=(const VulkanInstance&) = delete;
    VkInstance handle_{VK_NULL_HANDLE};
};
/** --------------------------------------------------------------------------------------------- PickDeviceIndex
 * @brief Apply the Vulkan picker policy to a list of physical devices.
 *
 * Policy (see task #37 description):
 *   1. INTEGRATED_GPU with HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL.
 *   2. CPU type (llvmpipe).
 *   3. Refuse DISCRETE_GPU (zero-copy or refuse).
 *
 * Returns -1 if nothing matches.
 */
static int PickDeviceIndex(const std::vector<VkPhysicalDevice>& devs) {
    constexpr VkMemoryPropertyFlags kWant = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                          | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto has_coherent = [&](VkPhysicalDevice d) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(d, &mp);
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
            if ((mp.memoryTypes[t].propertyFlags & kWant) == kWant) {
                return true;
            }
        }
        return false;
    };
    int integrated_idx = -1, cpu_idx = -1;
    for (size_t i = 0; i < devs.size(); ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (!has_coherent(devs[i])) {
            continue;
        }
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
            && integrated_idx < 0) {
            integrated_idx = static_cast<int>(i);
        } else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && cpu_idx < 0) {
            cpu_idx = static_cast<int>(i);
        }
    }
    if (integrated_idx >= 0) {
        return integrated_idx;
    }
    return cpu_idx;
}
/** --------------------------------------------------------------------------------------------- FindCoherentMemoryType
 * @brief Lowest-index memory type with HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL.
 *
 * Vulkan convention: lower-index types are preferred.
 */
static int FindCoherentMemoryType(VkPhysicalDevice dev) {
    constexpr VkMemoryPropertyFlags kWant = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                          | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(dev, &mp);
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
        if ((mp.memoryTypes[t].propertyFlags & kWant) == kWant) {
            return static_cast<int>(t);
        }
    }
    return -1;
}
/** --------------------------------------------------------------------------------------------- FindComputeQueueFamily
 * @brief First queue family supporting VK_QUEUE_COMPUTE_BIT.
 *
 * Compute capability is mandatory in Vulkan 1.0+ on any non-trivial
 * physical device. Graphics-capable families are also compute-capable.
 */
static int FindComputeQueueFamily(VkPhysicalDevice dev) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, nullptr);
    std::vector<VkQueueFamilyProperties> props(n);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, props.data());
    for (uint32_t i = 0; i < n; ++i) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return static_cast<int>(i);
        }
    }
    return -1;
}
/** --------------------------------------------------------------------------------------------- DeviceHasExtension
 */
static bool DeviceHasExtension(VkPhysicalDevice dev, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, exts.data());
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace internal
/** --------------------------------------------------------------------------------------------- VulkanDevice::Impl
 * @brief pImpl for VulkanDevice — owns the logical VkDevice handle.
 *
 * Cached fields are queried at construction (one Vulkan call each) so
 * accessors are O(1). The destructor calls vkDestroyDevice; the parent
 * VkInstance lives in the singleton and outlives every Impl.
 */
struct VulkanDevice::Impl {
    VkPhysicalDevice physical{VK_NULL_HANDLE};   ///< borrowed from instance
    VkDevice logical{VK_NULL_HANDLE};            ///< owned, destroyed in dtor
    VkQueue queue{VK_NULL_HANDLE};               ///< borrowed from logical
    uint32_t queue_family_index{0};
    uint32_t coherent_memory_type{0};
    bool supports_imported_host_memory{false};
    int64_t imported_host_pointer_alignment{0};
    std::array<uint8_t, VK_UUID_SIZE> uuid{};
    int64_t uuid_hash{0};
    std::string device_name;
    int64_t total_memory{0};
    std::shared_ptr<VulkanMemoryManager> default_mm;
    std::once_flag default_mm_init;
    Impl() = default;
    ~Impl() {
        if (logical != VK_NULL_HANDLE) {
            vkDestroyDevice(logical, nullptr);
        }
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};
/** --------------------------------------------------------------------------------------------- BuildImpl
 * @brief Build a fully-initialized VulkanDevice::Impl from a VkPhysicalDevice.
 */
static Result<std::shared_ptr<VulkanDevice::Impl>> BuildImpl(VkPhysicalDevice physical) {
    auto impl = std::make_shared<VulkanDevice::Impl>();
    impl->physical = physical;
    int qf = internal::FindComputeQueueFamily(physical);
    if (qf < 0) {
        return Status::Invalid("VulkanDevice: no compute queue family");
    }
    impl->queue_family_index = static_cast<uint32_t>(qf);
    int mt = internal::FindCoherentMemoryType(physical);
    if (mt < 0) {
        return Status::Invalid(
            "VulkanDevice: no HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL memory type");
    }
    impl->coherent_memory_type = static_cast<uint32_t>(mt);
    impl->supports_imported_host_memory =
        internal::DeviceHasExtension(physical, "VK_EXT_external_memory_host");
    // Probe minImportedHostPointerAlignment if extension is present. The
    // actual Wrap-time validation happens after vkCreateDevice — see the
    // post-create probe below — because some drivers (notably MoltenVK)
    // advertise the extension but refuse vkCreateBuffer with the external-
    // memory chain.
    if (impl->supports_imported_host_memory) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props{};
        host_props.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &host_props;
        vkGetPhysicalDeviceProperties2(physical, &props2);
        impl->imported_host_pointer_alignment =
            static_cast<int64_t>(host_props.minImportedHostPointerAlignment);
    }
    // Cached identification fields.
    VkPhysicalDeviceProperties dp;
    vkGetPhysicalDeviceProperties(physical, &dp);
    std::memcpy(impl->uuid.data(), dp.pipelineCacheUUID, VK_UUID_SIZE);
    impl->device_name = dp.deviceName;
    // Hash the UUID into a stable int64_t for arrow::Device::device_id().
    // Simple FNV-1a is deterministic, stable across runs, and conflicts
    // are fine in practice (we only need uniqueness within one host).
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint8_t b : impl->uuid) {
        h ^= b;
        h *= 0x100000001b3ULL;
    }
    impl->uuid_hash = static_cast<int64_t>(h);
    // total_memory: sum of DEVICE_LOCAL heap sizes.
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physical, &mp);
    int64_t total = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += static_cast<int64_t>(mp.memoryHeaps[i].size);
        }
    }
    impl->total_memory = total;
    // Create the logical device with a single compute queue + the
    // imported-host extension if available + timeline semaphore feature.
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = impl->queue_family_index;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_priority;

    std::vector<const char*> dev_exts;
    if (impl->supports_imported_host_memory) {
        dev_exts.push_back("VK_EXT_external_memory_host");
    }
#ifdef __APPLE__
    if (internal::DeviceHasExtension(physical, "VK_KHR_portability_subset")) {
        dev_exts.push_back("VK_KHR_portability_subset");
    }
#endif

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.timelineSemaphore = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();

    ARROW_VK_RETURN_NOT_OK(vkCreateDevice(physical, &dci, nullptr, &impl->logical),
                           "vkCreateDevice");
    vkGetDeviceQueue(impl->logical, impl->queue_family_index, 0, &impl->queue);
    // Honest probe for VK_EXT_external_memory_host. MoltenVK advertises the
    // extension at the physical-device level but its vkCreateBuffer rejects
    // VkExternalMemoryBufferCreateInfo with VK_ERROR_FEATURE_NOT_PRESENT.
    // Try a 1-page dummy Wrap; if it fails, downgrade the capability flag
    // so VulkanBuffer::Wrap and ViewBufferFrom return a clean Status::Invalid
    // / nullptr instead of leaking the runtime mismatch to callers.
    if (impl->supports_imported_host_memory && impl->imported_host_pointer_alignment > 0) {
        const int64_t a = impl->imported_host_pointer_alignment;
        void* probe = nullptr;
        if (::posix_memalign(&probe, static_cast<size_t>(a), static_cast<size_t>(a)) == 0) {
            VkExternalMemoryBufferCreateInfo ext_buf{};
            ext_buf.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
            ext_buf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.pNext = &ext_buf;
            bci.size = static_cast<VkDeviceSize>(a);
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer probe_buf = VK_NULL_HANDLE;
            VkResult r = vkCreateBuffer(impl->logical, &bci, nullptr, &probe_buf);
            if (r != VK_SUCCESS) {
                impl->supports_imported_host_memory = false;
                impl->imported_host_pointer_alignment = 0;
            } else {
                vkDestroyBuffer(impl->logical, probe_buf, nullptr);
            }
            std::free(probe);
        }
    }
    return impl;
}
/** --------------------------------------------------------------------------------------------- VulkanDeviceManager::Impl
 * @brief Singleton state — list of physical devices + cached VulkanDevice wrappers.
 */
class VulkanDeviceManager::Impl {
 public:
    explicit Impl(VkInstance inst) : instance_{inst} {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        std::vector<VkPhysicalDevice> all(n);
        vkEnumeratePhysicalDevices(instance_, &n, all.data());
        // Filter via the picker policy. Build acceptable_ in original order
        // so GetDevice(0) matches the picker's preferred device.
        int picked = internal::PickDeviceIndex(all);
        if (picked >= 0) {
            // Add the picker's choice first, then the rest of the
            // acceptable devices in enumeration order (so users who
            // really want device N can get it).
            acceptable_.push_back(all[picked]);
            for (size_t i = 0; i < all.size(); ++i) {
                if (static_cast<int>(i) == picked) continue;
                VkPhysicalDeviceProperties p;
                vkGetPhysicalDeviceProperties(all[i], &p);
                if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                    || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
                    if (internal::FindCoherentMemoryType(all[i]) >= 0) {
                        acceptable_.push_back(all[i]);
                    }
                }
            }
        }
    }
    int num_devices() const { return static_cast<int>(acceptable_.size()); }
    Result<std::shared_ptr<VulkanDevice>> GetDevice(int idx) {
        if (idx < 0 || idx >= num_devices()) {
            return Status::Invalid("VulkanDevice index ", idx, " out of range [0,",
                                   num_devices(), ")");
        }
        return ResolveCached(idx);
    }
    Result<std::shared_ptr<VulkanDevice>> GetDeviceByUuid(const uint8_t* uuid) {
        for (int i = 0; i < num_devices(); ++i) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(acceptable_[i], &p);
            if (std::memcmp(p.pipelineCacheUUID, uuid, VK_UUID_SIZE) == 0) {
                return ResolveCached(i);
            }
        }
        return Status::Invalid("No Vulkan device with the given pipelineCacheUUID");
    }
 private:
    Result<std::shared_ptr<VulkanDevice>> ResolveCached(int idx) {
        std::lock_guard<std::mutex> lk{cache_mu_};
        auto it = cache_.find(idx);
        if (it != cache_.end()) {
            return it->second;
        }
        ARROW_ASSIGN_OR_RAISE(auto impl, BuildImpl(acceptable_[idx]));
        std::shared_ptr<VulkanDevice> d{new VulkanDevice(std::move(impl))};
        cache_.emplace(idx, d);
        return d;
    }
    VkInstance instance_;
    std::vector<VkPhysicalDevice> acceptable_;
    std::mutex cache_mu_;
    std::unordered_map<int, std::shared_ptr<VulkanDevice>> cache_;
};

VulkanDeviceManager::VulkanDeviceManager() = default;
VulkanDeviceManager::~VulkanDeviceManager() = default;

Result<VulkanDeviceManager*> VulkanDeviceManager::Instance() {
    static std::unique_ptr<VulkanDeviceManager> instance;
    static std::once_flag init;
    static Status init_status;
    std::call_once(init, []() {
        auto inst_result = internal::VulkanInstance::Get();
        if (!inst_result.ok()) {
            init_status = inst_result.status();
            return;
        }
        instance.reset(new VulkanDeviceManager());
        instance->impl_ = std::make_shared<Impl>(inst_result.ValueOrDie());
    });
    if (!init_status.ok()) {
        return init_status;
    }
    if (instance->impl_->num_devices() == 0) {
        return Status::NotImplemented(
            "no Vulkan device with HOST_COHERENT | DEVICE_LOCAL memory found");
    }
    return instance.get();
}
int VulkanDeviceManager::num_devices() const { return impl_->num_devices(); }
Result<std::shared_ptr<VulkanDevice>> VulkanDeviceManager::GetDevice(int idx) {
    return impl_->GetDevice(idx);
}
Result<std::shared_ptr<VulkanDevice>> VulkanDeviceManager::GetDeviceByUuid(
    const uint8_t* uuid) {
    return impl_->GetDeviceByUuid(uuid);
}

VulkanDevice::VulkanDevice(std::shared_ptr<Impl> impl)
    : Device(/*is_cpu=*/true), impl_{std::move(impl)} {}
const char* VulkanDevice::type_name() const { return kVulkanDeviceTypeName; }
std::string VulkanDevice::ToString() const {
    std::ostringstream ss;
    ss << "VulkanDevice(name=\"" << impl_->device_name << "\", uuid_hash=0x"
       << std::hex << impl_->uuid_hash << std::dec
       << ", total_memory=" << impl_->total_memory << ")";
    return ss.str();
}
bool VulkanDevice::Equals(const Device& other) const {
    if (other.type_name() != type_name()) return false;
    const auto& vd = static_cast<const VulkanDevice&>(other);
    return std::memcmp(vd.impl_->uuid.data(), impl_->uuid.data(), VK_UUID_SIZE) == 0;
}
int64_t VulkanDevice::device_id() const { return impl_->uuid_hash; }
int64_t VulkanDevice::uuid_hash() const { return impl_->uuid_hash; }
const uint8_t* VulkanDevice::uuid() const { return impl_->uuid.data(); }
std::string VulkanDevice::device_name() const { return impl_->device_name; }
int64_t VulkanDevice::total_memory() const { return impl_->total_memory; }
bool VulkanDevice::supports_imported_host_memory() const {
    return impl_->supports_imported_host_memory;
}
int64_t VulkanDevice::imported_host_pointer_alignment() const {
    return impl_->imported_host_pointer_alignment;
}
void* VulkanDevice::vk_physical_device() const {
    return internal::ToOpaque(impl_->physical);
}
void* VulkanDevice::vk_device() const { return internal::ToOpaque(impl_->logical); }
void* VulkanDevice::vk_queue() const { return internal::ToOpaque(impl_->queue); }
uint32_t VulkanDevice::queue_family_index() const { return impl_->queue_family_index; }
uint32_t VulkanDevice::coherent_memory_type_index() const {
    return impl_->coherent_memory_type;
}
std::shared_ptr<MemoryManager> VulkanDevice::default_memory_manager() {
    std::call_once(impl_->default_mm_init, [this]() {
        impl_->default_mm = VulkanMemoryManager::Make(
            std::static_pointer_cast<VulkanDevice>(shared_from_this()));
    });
    return impl_->default_mm;
}

Result<std::shared_ptr<VulkanDevice>> VulkanDevice::Default() {
    ARROW_ASSIGN_OR_RAISE(auto* mgr, VulkanDeviceManager::Instance());
    return mgr->GetDevice(0);
}
Result<std::shared_ptr<VulkanDevice>> VulkanDevice::Make(const uint8_t* uuid) {
    ARROW_ASSIGN_OR_RAISE(auto* mgr, VulkanDeviceManager::Instance());
    return mgr->GetDeviceByUuid(uuid);
}

bool IsVulkanDevice(const Device& device) {
    return device.type_name() == kVulkanDeviceTypeName;
}
Result<std::shared_ptr<VulkanDevice>> AsVulkanDevice(
    const std::shared_ptr<Device>& device) {
    if (!device || !IsVulkanDevice(*device)) {
        return Status::Invalid("Expected VulkanDevice");
    }
    return std::static_pointer_cast<VulkanDevice>(device);
}
/** --------------------------------------------------------------------------------------------- VulkanDevice::Stream
 * @brief VkQueue + transient VkCommandPool wrapper.
 *
 * Vulkan command pools are not thread-safe to allocate-from concurrently,
 * so each Stream owns its own pool. Synchronize() drains the queue (not
 * just the pool) — this matches MetalDevice::Stream::Synchronize.
 */
VulkanDevice::Stream::Stream(std::shared_ptr<VulkanDevice> device, void* queue,
                             void* command_pool,
                             Device::Stream::release_fn_t release_fn)
    : Device::Stream(queue, release_fn),
      device_{std::move(device)},
      command_pool_{command_pool} {}
VulkanDevice::Stream::~Stream() {
    if (command_pool_ != nullptr && device_ != nullptr) {
        VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
        VkCommandPool pool = internal::FromOpaque<VkCommandPool>(command_pool_);
        vkDestroyCommandPool(dev, pool, nullptr);
    }
}
void* VulkanDevice::Stream::vk_command_pool() const { return command_pool_; }
Status VulkanDevice::Stream::Synchronize() const {
    VkQueue q = internal::FromOpaque<VkQueue>(const_cast<void*>(stream_.get()));
    if (q == VK_NULL_HANDLE) {
        return Status::Invalid("VulkanDevice::Stream is null");
    }
    ARROW_VK_RETURN_NOT_OK(vkQueueWaitIdle(q), "vkQueueWaitIdle");
    return Status::OK();
}
Status VulkanDevice::Stream::WaitEvent(const Device::SyncEvent& event) {
    const auto& ev = static_cast<const VulkanDevice::SyncEvent&>(event);
    VkQueue q = internal::FromOpaque<VkQueue>(const_cast<void*>(stream_.get()));
    VkSemaphore sem = internal::FromOpaque<VkSemaphore>(const_cast<void*>(ev.get_raw()));
    if (q == VK_NULL_HANDLE || sem == VK_NULL_HANDLE) {
        return Status::Invalid("VulkanDevice::Stream::WaitEvent received nil handles");
    }
    if (command_pool_ == nullptr) {
        return Status::Invalid("VulkanDevice::Stream::WaitEvent requires an "
                               "owned command pool (use MakeStream, not WrapStream)");
    }
    VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
    VkCommandPool pool = internal::FromOpaque<VkCommandPool>(command_pool_);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ARROW_VK_RETURN_NOT_OK(vkAllocateCommandBuffers(dev, &cbai, &cmd),
                           "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ARROW_VK_RETURN_NOT_OK(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    ARROW_VK_RETURN_NOT_OK(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    // Legacy vkQueueSubmit with VkTimelineSemaphoreSubmitInfo chain.
    // Works on Vulkan 1.2 unconditionally; vkQueueSubmit2 would require
    // the synchronization2 feature which is 1.3 / KHR extension.
    const uint64_t target = ev.signal_value();
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tssi.waitSemaphoreValueCount = 1;
    tssi.pWaitSemaphoreValues = &target;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tssi;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &sem;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ARROW_VK_RETURN_NOT_OK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    return Status::OK();
}

Result<std::shared_ptr<Device::Stream>> VulkanDevice::MakeStream(unsigned int /*flags*/) {
    VkDevice dev = internal::FromOpaque<VkDevice>(impl_->logical);
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
              | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = impl_->queue_family_index;
    VkCommandPool pool = VK_NULL_HANDLE;
    ARROW_VK_RETURN_NOT_OK(vkCreateCommandPool(dev, &pci, nullptr, &pool),
                           "vkCreateCommandPool");
    void* opaque_queue = internal::ToOpaque(impl_->queue);
    void* opaque_pool = internal::ToOpaque(pool);
    auto release = [](void*) { /* command pool freed in ~Stream */ };
    return std::shared_ptr<Device::Stream>(new Stream(
        std::static_pointer_cast<VulkanDevice>(shared_from_this()),
        opaque_queue, opaque_pool, release));
}
Result<std::shared_ptr<Device::Stream>> VulkanDevice::WrapStream(
    void* device_stream, Stream::release_fn_t release_fn) {
    if (device_stream == nullptr) {
        return Status::Invalid("VulkanDevice::WrapStream received nullptr stream");
    }
    return std::shared_ptr<Device::Stream>(new Stream(
        std::static_pointer_cast<VulkanDevice>(shared_from_this()),
        device_stream, /*command_pool=*/nullptr, release_fn));
}
/** --------------------------------------------------------------------------------------------- VulkanDevice::SyncEvent
 * @brief VkTimelineSemaphore wrapper.
 *
 * Record() atomically reserves the next signal value and submits an
 * empty command buffer that signals the timeline semaphore at that
 * value. Wait() blocks via vkWaitSemaphores until the value is reached.
 */
Status VulkanDevice::SyncEvent::Record(const Device::Stream& stream) {
    VkSemaphore sem = internal::FromOpaque<VkSemaphore>(sync_event_.get());
    VkQueue q = internal::FromOpaque<VkQueue>(const_cast<void*>(stream.get_raw()));
    if (sem == VK_NULL_HANDLE || q == VK_NULL_HANDLE) {
        return Status::Invalid("VulkanDevice::SyncEvent::Record received nil handles");
    }
    // Need a command pool to allocate a (no-op) command buffer. MoltenVK
    // (and some other drivers) refuse signal-only submits with
    // commandBufferInfoCount == 0; an empty cb is the portable workaround.
    const auto* vk_stream = dynamic_cast<const VulkanDevice::Stream*>(&stream);
    if (vk_stream == nullptr || vk_stream->vk_command_pool() == nullptr) {
        return Status::Invalid("VulkanDevice::SyncEvent::Record requires a "
                               "VulkanDevice::Stream with an owned command pool");
    }
    VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
    VkCommandPool pool = internal::FromOpaque<VkCommandPool>(vk_stream->vk_command_pool());

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ARROW_VK_RETURN_NOT_OK(vkAllocateCommandBuffers(dev, &cbai, &cmd),
                           "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ARROW_VK_RETURN_NOT_OK(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    ARROW_VK_RETURN_NOT_OK(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    const uint64_t v = signal_value_.fetch_add(1, std::memory_order_acq_rel) + 1;
    // Legacy vkQueueSubmit with VkTimelineSemaphoreSubmitInfo chain (1.2 core).
    VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tssi.signalSemaphoreValueCount = 1;
    tssi.pSignalSemaphoreValues = &v;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tssi;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sem;
    ARROW_VK_RETURN_NOT_OK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    // The command buffer can be freed after the submit — Vulkan keeps a
    // reference internally until the submission completes. We don't free
    // here; the pool is destroyed when the Stream is destroyed.
    return Status::OK();
}
Status VulkanDevice::SyncEvent::Wait() {
    VkSemaphore sem = internal::FromOpaque<VkSemaphore>(sync_event_.get());
    if (sem == VK_NULL_HANDLE) {
        return Status::Invalid("VulkanDevice::SyncEvent::Wait received nil event");
    }
    const uint64_t target = signal_value_.load(std::memory_order_acquire);
    VkSemaphoreWaitInfo wi{};
    wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores = &sem;
    wi.pValues = &target;

    VkDevice dev = internal::FromOpaque<VkDevice>(device_->vk_device());
    // UINT64_MAX = "wait forever". Do NOT change to 0: per Vulkan spec
    // timeout=0 returns immediately if the value is not yet reached,
    // which would turn this into a non-blocking poll and silently break
    // all CPU/GPU synchronization. (Same lesson as Metal's MTLSharedEvent.)
    VkResult r = vkWaitSemaphores(dev, &wi, UINT64_MAX);
    if (r == VK_TIMEOUT) {
        return Status::IOError("VulkanDevice::SyncEvent::Wait timed out");
    }
    ARROW_VK_RETURN_NOT_OK(r, "vkWaitSemaphores");
    return Status::OK();
}

}  // namespace vulkan
}  // namespace arrow
