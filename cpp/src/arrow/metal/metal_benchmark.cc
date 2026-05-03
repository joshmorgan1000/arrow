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
#include "arrow/metal/metal_api.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/logging.h"

namespace arrow {
namespace metal {
namespace {
/// Skip the benchmark gracefully if no Metal device exists (headless CI
/// runners on Intel macOS, virtualized hosts). Returns false and marks
/// the benchmark skipped; caller should `return` immediately.
bool RequireMetal(benchmark::State& state) {
    if (!MetalDevice::Default().ok()) {
        state.SkipWithError("No Metal device on this host");
        return false;
    }
    return true;
}
/** --------------------------------------------------------------------------------------------- AllocateBuffer
 * @brief Time MetalMemoryManager::AllocateBuffer at varying sizes.
 *
 * Measures the cost of `[device newBufferWithLength:options:]` against
 * Apple's page allocator. Most of the cost is the system call to back
 * pages; for small allocations Metal's caching dominates.
 */
void BM_AllocateBuffer(benchmark::State& state) {
    if (!RequireMetal(state)) {
        return;
    }
    auto device = MetalDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<MetalMemoryManager>(device->default_memory_manager());
    const int64_t size = state.range(0);
    for (auto _ : state) {
        auto buf = mm->AllocateBuffer(size).ValueOrDie();
        benchmark::DoNotOptimize(buf->data());
    }
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_AllocateBuffer)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28);
/** --------------------------------------------------------------------------------------------- CPUtoMetalCopy
 * @brief Time CPU → Metal copy via MemoryManager::CopyBuffer.
 *
 * Establishes the lower bound for "old-school" data ingest. Should be
 * memcpy-bound on Apple Silicon since storage is unified.
 */
void BM_CPUtoMetalCopy(benchmark::State& state) {
    if (!RequireMetal(state)) {
        return;
    }
    auto device = MetalDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<MetalMemoryManager>(device->default_memory_manager());
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
BENCHMARK(BM_CPUtoMetalCopy)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28);
/** --------------------------------------------------------------------------------------------- CPUtoMetalView
 * @brief Time the zero-copy host→Metal view path. Should be ~constant time.
 *
 * Allocates a page-aligned host buffer, then re-views it as Metal via
 * MemoryManager::ViewBuffer. The cost is `[device
 * newBufferWithBytesNoCopy:...]` plus retain/release; no data is copied.
 */
void BM_CPUtoMetalView(benchmark::State& state) {
    if (!RequireMetal(state)) {
        return;
    }
    auto device = MetalDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<MetalMemoryManager>(device->default_memory_manager());
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const int64_t size = state.range(0);
    void* aligned = nullptr;
    if (::posix_memalign(&aligned, page, size) != 0) {
        state.SkipWithError("posix_memalign failed");
        return;
    }
    std::memset(aligned, 0x42, size);
    auto cpu = std::make_shared<Buffer>(static_cast<const uint8_t*>(aligned), size);
    for (auto _ : state) {
        auto view = MemoryManager::ViewBuffer(cpu, mm).ValueOrDie();
        benchmark::DoNotOptimize(view->data());
    }
    state.SetBytesProcessed(state.iterations() * size);
    std::free(aligned);
}
BENCHMARK(BM_CPUtoMetalView)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28);
/** --------------------------------------------------------------------------------------------- SyncEventRecordWait
 * @brief Time a Record + Wait round-trip through MTLSharedEvent.
 *
 * Measures latency, not throughput. The dispatched command buffer is
 * empty so the kernel is queue+event scheduling overhead.
 */
void BM_SyncEventRecordWait(benchmark::State& state) {
    if (!RequireMetal(state)) {
        return;
    }
    auto device = MetalDevice::Default().ValueOrDie();
    auto mm = std::static_pointer_cast<MetalMemoryManager>(device->default_memory_manager());
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
/** --------------------------------------------------------------------------------------------- MetalPoolAllocFree
 * @brief Allocator-only round-trip for MetalMemoryPool.
 *
 * Compares against the system pool to expose any overhead from the
 * MTLBuffer wrapper layer (typically dominated by Metal page allocator).
 */
void BM_MetalPoolAllocFree(benchmark::State& state) {
    auto* pool = MetalMemoryPool::Instance();
    if (pool == nullptr) {
        state.SkipWithError("No Metal device on this host");
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
BENCHMARK(BM_MetalPoolAllocFree)->Arg(1 << 12)->Arg(1 << 20)->Arg(1 << 24);

}  // namespace
}  // namespace metal
}  // namespace arrow
