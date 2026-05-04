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

#include "benchmark/benchmark.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unistd.h>

#include "arrow/buffer.h"
#include "arrow/memory_pool.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/logging.h"
#include "arrow/vulkan/vulkan_api.h"

namespace arrow {
namespace vulkan {
namespace {
/// Skip benchmark gracefully if no Vulkan device exists.
bool RequireVulkan(benchmark::State& state) {
    if (!VulkanDevice::Default().ok()) {
        state.SkipWithError("No Vulkan device on this host");
        return false;
    }
    return true;
}
/** --------------------------------------------------------------------------------------------- AllocateBuffer
 * @brief vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory + vkMapMemory round-trip.
 */
void BM_AllocateBuffer(benchmark::State& state) {
    if (!RequireVulkan(state)) return;
    auto device = VulkanDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<VulkanMemoryManager>(device->default_memory_manager());
    const int64_t size = state.range(0);
    for (auto _ : state) {
        auto buf = mm->AllocateBuffer(size).ValueOrDie();
        benchmark::DoNotOptimize(buf->data());
    }
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_AllocateBuffer)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28);
/** --------------------------------------------------------------------------------------------- CPUtoVulkanCopy
 * @brief CPU→Vulkan copy via MemoryManager::CopyBuffer. memcpy-bound on integrated.
 */
void BM_CPUtoVulkanCopy(benchmark::State& state) {
    if (!RequireVulkan(state)) return;
    auto device = VulkanDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<VulkanMemoryManager>(device->default_memory_manager());
    const int64_t size = state.range(0);
    auto src = AllocateBuffer(size).ValueOrDie();
    std::memset(src->mutable_data(), 0x42, size);
    auto src_shared = std::shared_ptr<Buffer>(std::move(src));
    for (auto _ : state) {
        auto dst = MemoryManager::CopyBuffer(src_shared, mm).ValueOrDie();
        benchmark::DoNotOptimize(dst->data());
    }
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_CPUtoVulkanCopy)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28);
/** --------------------------------------------------------------------------------------------- CPUtoVulkanView
 * @brief Zero-copy CPU→Vulkan view via VK_EXT_external_memory_host.
 *
 * Skipped if the device does not actually support imported host memory
 * (e.g., MoltenVK advertises but rejects the buffer create).
 */
void BM_CPUtoVulkanView(benchmark::State& state) {
    if (!RequireVulkan(state)) return;
    auto device = VulkanDevice::Default().ValueOrDie();
    if (!device->supports_imported_host_memory()) {
        state.SkipWithError("VK_EXT_external_memory_host not supported");
        return;
    }
    auto mm = std::static_pointer_cast<VulkanMemoryManager>(device->default_memory_manager());
    const int64_t align = device->imported_host_pointer_alignment();
    const int64_t requested = state.range(0);
    const int64_t size = ((requested + align - 1) / align) * align;
    void* aligned = nullptr;
    if (::posix_memalign(&aligned, static_cast<size_t>(align),
                         static_cast<size_t>(size)) != 0) {
        state.SkipWithError("posix_memalign failed");
        return;
    }
    std::memset(aligned, 0x42, static_cast<size_t>(size));
    auto cpu = std::make_shared<Buffer>(static_cast<const uint8_t*>(aligned), size);
    for (auto _ : state) {
        auto view = MemoryManager::ViewBuffer(cpu, mm).ValueOrDie();
        benchmark::DoNotOptimize(view->data());
    }
    state.SetBytesProcessed(state.iterations() * size);
    std::free(aligned);
}
BENCHMARK(BM_CPUtoVulkanView)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28);
/** --------------------------------------------------------------------------------------------- SyncEventRecordWait
 * @brief VkTimelineSemaphore Record+Wait latency.
 */
void BM_SyncEventRecordWait(benchmark::State& state) {
    if (!RequireVulkan(state)) return;
    auto device = VulkanDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<VulkanMemoryManager>(device->default_memory_manager());
    auto event = mm->MakeDeviceSyncEvent().ValueOrDie();
    auto stream = device->MakeStream().ValueOrDie();
    for (auto _ : state) {
        auto s = event->Record(*stream);
        benchmark::DoNotOptimize(s);
        s = event->Wait();
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_SyncEventRecordWait);
/** --------------------------------------------------------------------------------------------- VulkanPoolAllocFree
 * @brief Allocator-only round-trip for VulkanMemoryPool.
 */
void BM_VulkanPoolAllocFree(benchmark::State& state) {
    auto* pool = VulkanMemoryPool::Instance();
    if (pool == nullptr) {
        state.SkipWithError("No Vulkan device on this host");
        return;
    }
    const int64_t size = state.range(0);
    for (auto _ : state) {
        uint8_t* p = nullptr;
        ARROW_CHECK_OK(pool->Allocate(size, &p));
        benchmark::DoNotOptimize(p);
        pool->Free(p, size);
    }
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_VulkanPoolAllocFree)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24);

}  // namespace
}  // namespace vulkan
}  // namespace arrow
