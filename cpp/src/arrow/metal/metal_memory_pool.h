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
#include <string>

#include "arrow/memory_pool.h"
#include "arrow/metal/metal_device.h"
#include "arrow/metal/visibility.h"

namespace arrow {
namespace metal {
/** --------------------------------------------------------------------------------------------- MetalMemoryPool
 * @brief Drop-in arrow::MemoryPool whose allocations are Metal-coherent
 *
 * Bytes returned from `Allocate` come from the `.contents` of a
 * MTLBuffer (`MTLResourceStorageModeShared`), so any Arrow data
 * structure built atop this pool is simultaneously CPU- and GPU-
 * addressable on Apple Silicon. Wire it in via `default_memory_pool`
 * override or by passing `MetalMemoryPool::Instance()` directly to the
 * relevant builders.
 *
 * Allocation tracking and stats follow the system pool's contract
 * exactly (see `arrow::MemoryPool`); behavior is identical except for
 * where the bytes live.
 */
class ARROW_METAL_EXPORT MetalMemoryPool : public MemoryPool {
 public:
    /// \brief Construct a pool bound to the given MetalDevice
    explicit MetalMemoryPool(std::shared_ptr<MetalDevice> device);
    ~MetalMemoryPool() override;
    using MemoryPool::Allocate;
    using MemoryPool::Reallocate;
    using MemoryPool::Free;
    Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
    Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                      uint8_t** ptr) override;
    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;
    int64_t bytes_allocated() const override;
    int64_t total_bytes_allocated() const override;
    int64_t num_allocations() const override;
    int64_t max_memory() const override;
    std::string backend_name() const override;
    /// \brief Process-wide singleton bound to the system default MetalDevice
    static MetalMemoryPool* Instance();
 private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace metal
}  // namespace arrow
