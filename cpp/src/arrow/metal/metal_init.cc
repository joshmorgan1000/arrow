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

// Process-wide initialization for arrow_metal.
//
// Registers the Metal DeviceMapper at static-init time so that
// arrow::ImportDeviceArray / ImportDeviceRecordBatch resolve a
// MetalMemoryManager when handed an ArrowDeviceArray with
// device_type == ARROW_DEVICE_METAL (= 8).

#include "arrow/device.h"
#include "arrow/metal/metal_device.h"
#include "arrow/metal/metal_memory.h"
#include "arrow/metal/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/logging_internal.h"

namespace arrow {
namespace metal {
namespace internal {
/** --------------------------------------------------------------------------------------------- MetalDeviceMapper
 * @brief Resolve `device_id` (Apple registry ID) → MetalMemoryManager.
 *
 * Bound at static-init via RegisterDeviceMapper. The C-Data Interface
 * import path consults the registry to convert a foreign-process
 * MTLDevice handle into a local MetalMemoryManager that can wrap the
 * incoming pointer as a MetalBuffer.
 */
static Result<std::shared_ptr<MemoryManager>> MetalDeviceMapper(int64_t device_id) {
    ARROW_ASSIGN_OR_RAISE(auto* mgr, MetalDeviceManager::Instance());
    Result<std::shared_ptr<MetalDevice>> dev_result = mgr->GetDeviceByRegistryId(device_id);
    if (!dev_result.ok()) {
        // Fall back to the system default device when the foreign registry ID
        // does not match anything local. This covers the common single-GPU
        // laptop case where producer and consumer share the same device but
        // the producer encoded a non-portable ID.
        ARROW_ASSIGN_OR_RAISE(auto fallback, mgr->GetDevice(0));
        return fallback->default_memory_manager();
    }
    return dev_result.ValueOrDie()->default_memory_manager();
}
/** --------------------------------------------------------------------------------------------- RegisterMetalDeviceMapper
 * @brief Static-init registration. Idempotent via std::call_once so
 *        repeated dynamic-loads of arrow_metal don't double-register.
 */
ARROW_METAL_EXPORT void EnsureMetalLinked();
void EnsureMetalLinked() {
    static const auto kRegister = []() {
        Status s = RegisterDeviceMapper(DeviceAllocationType::kMETAL, &MetalDeviceMapper);
        if (!s.ok()) {
            ARROW_LOG(WARNING) << "arrow_metal: RegisterDeviceMapper failed: " << s;
        }
        return 0;
    }();
    (void)kRegister;
}
namespace {
/// \brief Force-trigger registration when arrow_metal is loaded.
struct MetalDeviceMapperAutoRegister {
    MetalDeviceMapperAutoRegister() { EnsureMetalLinked(); }
};
static MetalDeviceMapperAutoRegister kMetalDeviceMapperAutoRegister;
}  // namespace

}  // namespace internal
}  // namespace metal
}  // namespace arrow
