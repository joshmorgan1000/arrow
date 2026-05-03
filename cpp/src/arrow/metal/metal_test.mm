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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "arrow/buffer.h"
#include "arrow/device.h"
#include "arrow/memory_pool.h"
#include "arrow/metal/metal_api.h"
#include "arrow/metal/metal_internal.hh"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace metal {
/** --------------------------------------------------------------------------------------------- MetalEnvironment
 * @brief One-time fixture that resolves the system default MetalDevice
 *        and exposes its default MemoryManager to every test.
 */
class MetalEnvironment : public ::testing::Test {
 protected:
    void SetUp() override {
        // GitHub Actions Intel-macOS runners and other headless hosts
        // can lack a Metal device. Skip rather than fail so the suite is
        // useful in CI.
        auto dev_result = MetalDevice::Default();
        if (!dev_result.ok()) {
            GTEST_SKIP() << "No Metal device available: " << dev_result.status();
        }
        device_ = dev_result.ValueOrDie();
        mm_ = std::static_pointer_cast<MetalMemoryManager>(device_->default_memory_manager());
    }
    std::shared_ptr<MetalDevice> device_;
    std::shared_ptr<MetalMemoryManager> mm_;
};
TEST_F(MetalEnvironment, DeviceProperties) {
    ASSERT_NE(device_, nullptr);
    EXPECT_STREQ(device_->type_name(), "arrow::metal::MetalDevice");
    EXPECT_EQ(device_->device_type(), DeviceAllocationType::kMETAL);
    EXPECT_TRUE(device_->is_cpu());  ///< unified memory: true on Apple Silicon
    EXPECT_TRUE(device_->Equals(*device_));
    EXPECT_NE(device_->device_id(), -1);
    EXPECT_FALSE(device_->device_name().empty());
    EXPECT_GT(device_->total_memory(), 0);
}
TEST_F(MetalEnvironment, DeviceManagerSingleton) {
    ASSERT_OK_AND_ASSIGN(auto* mgr, MetalDeviceManager::Instance());
    EXPECT_GE(mgr->num_devices(), 1);
    ASSERT_OK_AND_ASSIGN(auto dev0, mgr->GetDevice(0));
    EXPECT_EQ(dev0->device_type(), DeviceAllocationType::kMETAL);
}
TEST_F(MetalEnvironment, AllocateBuffer) {
    ASSERT_OK_AND_ASSIGN(auto buf, mm_->AllocateBuffer(4096));
    ASSERT_NE(buf->data(), nullptr);
    EXPECT_EQ(buf->size(), 4096);
    EXPECT_TRUE(buf->is_mutable());
    EXPECT_TRUE(buf->is_cpu());  ///< MetalBuffer is simultaneously CPU+GPU addressable
    EXPECT_EQ(buf->device_type(), DeviceAllocationType::kMETAL);
    EXPECT_TRUE(IsMetalBuffer(*buf));
}
TEST_F(MetalEnvironment, AllocateBufferZeroSize) {
    ASSERT_OK_AND_ASSIGN(auto buf, mm_->AllocateBuffer(0));
    EXPECT_NE(buf, nullptr);
    EXPECT_EQ(buf->device_type(), DeviceAllocationType::kMETAL);
    // Arrow contract: zero-length request → size()==0 and capacity()==0,
    // even though the underlying MTLBuffer is 1 byte (Apple rejects 0).
    EXPECT_EQ(buf->size(), 0);
    EXPECT_EQ(buf->capacity(), 0);
}
/** --------------------------------------------------------------------------------------------- DispatchByteDouble
 * @brief Run a tiny Metal compute kernel that multiplies each byte by 2.
 *
 * Used to prove GPU coherence: CPU writes through the MetalBuffer's
 * data() pointer, GPU reads via MTLBuffer, doubles, writes back, and
 * CPU re-reads to confirm the GPU saw the original data and produced
 * the doubled output in the same memory.
 */
static Status DispatchByteDouble(MetalDevice& device, MetalBuffer& target) {
    @autoreleasepool {
        id<MTLDevice> dev = internal::FromOpaqueBorrowed<id<MTLDevice>>(device.mtl_device());
        // root_mtl_buffer() walks the parent_ chain so this works for both
        // root buffers and arbitrarily nested slices; the encoder sees the
        // root MTLBuffer and the absolute byte offset.
        id<MTLBuffer> buf = internal::FromOpaqueBorrowed<id<MTLBuffer>>(
            target.root_mtl_buffer());
        if (dev == nil || buf == nil) {
            return Status::Invalid("DispatchByteDouble: nil handles");
        }
        NSError* err = nil;
        NSString* src = @"#include <metal_stdlib>\n"
                        @"using namespace metal;\n"
                        @"kernel void double_bytes(device uchar* data [[buffer(0)]],\n"
                        @"                         constant uint& n [[buffer(1)]],\n"
                        @"                         uint gid [[thread_position_in_grid]]) {\n"
                        @"  if (gid < n) data[gid] = data[gid] * 2;\n"
                        @"}\n";
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (lib == nil) {
            return Status::Invalid("newLibraryWithSource failed: ",
                                   err ? [[err localizedDescription] UTF8String]
                                       : "unknown");
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"double_bytes"];
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) {
            return Status::Invalid("newComputePipelineState failed: ",
                                   err ? [[err localizedDescription] UTF8String]
                                       : "unknown");
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        // Slices report a non-zero `offset()`; pass that to the encoder
        // so kernels operate on the right window of the root MTLBuffer.
        [enc setBuffer:buf offset:static_cast<NSUInteger>(target.offset()) atIndex:0];
        const uint32_t n = static_cast<uint32_t>(target.size());
        [enc setBytes:&n length:sizeof(n) atIndex:1];
        const NSUInteger tg = pso.maxTotalThreadsPerThreadgroup < 256
                                  ? pso.maxTotalThreadsPerThreadgroup
                                  : 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error != nil) {
            return Status::Invalid("command buffer error: ",
                                   [[cb.error localizedDescription] UTF8String]);
        }
        return Status::OK();
    }
}
TEST_F(MetalEnvironment, GPUSeesCPUWrites) {
    constexpr int64_t kSize = 1024;
    ASSERT_OK_AND_ASSIGN(auto unique_buf, mm_->AllocateBuffer(kSize));
    auto buf = std::shared_ptr<Buffer>(std::move(unique_buf));
    auto* mb = static_cast<MetalBuffer*>(buf.get());
    // CPU writes 0..255 cycling
    for (int64_t i = 0; i < kSize; ++i) {
        buf->mutable_data()[i] = static_cast<uint8_t>(i & 0xFF);
    }
    ASSERT_OK(DispatchByteDouble(*device_, *mb));
    // CPU verifies GPU doubled in place. mod-256 arithmetic.
    for (int64_t i = 0; i < kSize; ++i) {
        const uint8_t expected = static_cast<uint8_t>((i & 0xFF) * 2);
        ASSERT_EQ(buf->data()[i], expected) << "mismatch at byte " << i;
    }
}
TEST_F(MetalEnvironment, ZeroCopyHostView) {
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, page, page), 0);
    std::memset(aligned, 0xAB, page);
    {
        // Inner scope: view + cpu_buf must drop their MTLBuffer refs
        // BEFORE std::free runs, otherwise the no-copy MTLBuffer's
        // backing pages get freed while Metal still references them.
        auto cpu_buf = std::make_shared<Buffer>(static_cast<const uint8_t*>(aligned),
                                                static_cast<int64_t>(page));
        ASSERT_OK_AND_ASSIGN(auto view, MemoryManager::ViewBuffer(cpu_buf, mm_));
        ASSERT_NE(view, nullptr);  ///< zero-copy view succeeded
        ASSERT_EQ(view->device_type(), DeviceAllocationType::kMETAL);
        EXPECT_EQ(view->data(), cpu_buf->data());  ///< same physical pointer
        // Verify GPU sees the host-written pattern via the wrap path.
        auto* mview = static_cast<MetalBuffer*>(view.get());
        ASSERT_OK(DispatchByteDouble(*device_, *mview));
        // 0xAB * 2 = 0x156 → low byte 0x56
        EXPECT_EQ(static_cast<uint8_t>(cpu_buf->data()[0]), 0x56);
    }
    std::free(aligned);
}
TEST_F(MetalEnvironment, ZeroCopyHostViewUnalignedFallsBack) {
    // Off-by-one offset breaks page alignment; ViewBuffer must return
    // a non-Metal view (or a copy) rather than crashing or hard-failing.
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, page, page * 2), 0);
    {
        auto* misaligned = static_cast<uint8_t*>(aligned) + 1;
        auto cpu_buf = std::make_shared<Buffer>(misaligned, static_cast<int64_t>(page - 2));
        ASSERT_OK_AND_ASSIGN(auto view, MemoryManager::ViewBuffer(cpu_buf, mm_));
        // The base contract: ViewBuffer either returns a Metal-side view OR
        // returns the source unchanged. Either way, view must be non-null.
        EXPECT_NE(view, nullptr);
    }
    std::free(aligned);
}
TEST_F(MetalEnvironment, ViewBufferTo) {
    ASSERT_OK_AND_ASSIGN(auto unique_buf, mm_->AllocateBuffer(64));
    auto src = std::shared_ptr<Buffer>(std::move(unique_buf));
    std::memset(src->mutable_data(), 0x11, src->size());
    ASSERT_OK_AND_ASSIGN(auto cpu_view, MemoryManager::ViewBuffer(src, default_cpu_memory_manager()));
    ASSERT_NE(cpu_view, nullptr);
    EXPECT_EQ(cpu_view->data(), src->data());  ///< unified memory: same pointer
    EXPECT_EQ(cpu_view->device_type(), DeviceAllocationType::kCPU);
    EXPECT_EQ(cpu_view->data()[0], 0x11);
}
TEST_F(MetalEnvironment, CopyBufferRoundTrip) {
    ASSERT_OK_AND_ASSIGN(auto src, AllocateBuffer(128));
    std::memset(src->mutable_data(), 0x33, src->size());
    auto src_shared = std::shared_ptr<Buffer>(std::move(src));
    ASSERT_OK_AND_ASSIGN(auto on_metal, MemoryManager::CopyBuffer(src_shared, mm_));
    ASSERT_EQ(on_metal->device_type(), DeviceAllocationType::kMETAL);
    EXPECT_EQ(std::memcmp(on_metal->data(), src_shared->data(), src_shared->size()), 0);
    ASSERT_OK_AND_ASSIGN(auto back, MemoryManager::CopyBuffer(on_metal, default_cpu_memory_manager()));
    EXPECT_EQ(std::memcmp(back->data(), src_shared->data(), src_shared->size()), 0);
}
TEST_F(MetalEnvironment, SyncEventRoundTrip) {
    ASSERT_OK_AND_ASSIGN(auto event, mm_->MakeDeviceSyncEvent());
    ASSERT_NE(event, nullptr);
    ASSERT_OK_AND_ASSIGN(auto stream, device_->MakeStream());
    ASSERT_NE(stream, nullptr);
    ASSERT_OK(event->Record(*stream));
    ASSERT_OK(event->Wait());
}
TEST_F(MetalEnvironment, FromBufferDowncast) {
    ASSERT_OK_AND_ASSIGN(auto unique_buf, mm_->AllocateBuffer(8));
    auto buf = std::shared_ptr<Buffer>(std::move(unique_buf));
    ASSERT_OK_AND_ASSIGN(auto mb, MetalBuffer::FromBuffer(buf));
    EXPECT_EQ(mb->data(), buf->data());
}
TEST_F(MetalEnvironment, MemoryPoolAllocateFreeBalances) {
    auto* pool = MetalMemoryPool::Instance();
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->backend_name(), "metal");
    const int64_t starting = pool->bytes_allocated();
    uint8_t* p1 = nullptr;
    uint8_t* p2 = nullptr;
    ASSERT_OK(pool->Allocate(1024, &p1));
    ASSERT_OK(pool->Allocate(2048, &p2));
    EXPECT_EQ(pool->bytes_allocated(), starting + 1024 + 2048);
    // Pointer is the MTLBuffer .contents — write to it; it's CPU-coherent.
    std::memset(p1, 0xCC, 1024);
    std::memset(p2, 0xDD, 2048);
    EXPECT_EQ(p1[100], 0xCC);
    EXPECT_EQ(p2[100], 0xDD);
    pool->Free(p1, 1024);
    pool->Free(p2, 2048);
    EXPECT_EQ(pool->bytes_allocated(), starting);
    EXPECT_GE(pool->num_allocations(), 2);
}
struct WrapTestState {
    int count;
};
static void WrapTestRelease(void* p, int64_t /*sz*/, void* ud) {
    auto* st = static_cast<WrapTestState*>(ud);
    ++st->count;
    std::free(p);
}
TEST_F(MetalEnvironment, WrapInvokesDeallocatorOnce) {
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, page, page), 0);
    std::memset(aligned, 0x77, page);
    WrapTestState st{0};
    {
        ASSERT_OK_AND_ASSIGN(auto wrapped,
                             MetalBuffer::Wrap(mm_, aligned, static_cast<int64_t>(page),
                                               &WrapTestRelease, &st));
        EXPECT_EQ(wrapped->data(), aligned);
        EXPECT_EQ(wrapped->size(), static_cast<int64_t>(page));
        EXPECT_EQ(wrapped->parent(), nullptr);  ///< sole-owner contract
        // Verify GPU sees the bytes through the wrapped MTLBuffer.
        auto* mb = static_cast<MetalBuffer*>(wrapped.get());
        ASSERT_OK(DispatchByteDouble(*device_, *mb));
        EXPECT_EQ(static_cast<uint8_t>(wrapped->data()[0]), static_cast<uint8_t>(0x77 * 2));
    }
    // After the MetalBuffer drops the deallocator must have run exactly once.
    EXPECT_EQ(st.count, 1);
}
TEST_F(MetalEnvironment, WrapRejectsMisaligned) {
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, page, page * 2), 0);
    auto* misaligned = static_cast<uint8_t*>(aligned) + 1;
    auto result = MetalBuffer::Wrap(mm_, misaligned, static_cast<int64_t>(page) - 2,
                                    /*release_fn=*/nullptr);
    EXPECT_FALSE(result.ok());
    std::free(aligned);
}
TEST_F(MetalEnvironment, MemoryPoolReallocate) {
    auto* pool = MetalMemoryPool::Instance();
    ASSERT_NE(pool, nullptr);
    uint8_t* p = nullptr;
    ASSERT_OK(pool->Allocate(64, &p));
    std::memset(p, 0xEE, 64);
    ASSERT_OK(pool->Reallocate(64, 256, &p));
    // Old contents preserved.
    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(p[i], 0xEE) << "lost data on Reallocate at byte " << i;
    }
    pool->Free(p, 256);
}
TEST_F(MetalEnvironment, NestedSliceGPUDispatch) {
    constexpr int64_t kSize = 4096;
    ASSERT_OK_AND_ASSIGN(auto unique, mm_->AllocateBuffer(kSize));
    auto buf = std::shared_ptr<Buffer>(std::move(unique));
    ASSERT_OK_AND_ASSIGN(auto root, MetalBuffer::FromBuffer(buf));
    // Initialize root: each byte = i mod 256.
    for (int64_t i = 0; i < kSize; ++i) {
        root->mutable_data()[i] = static_cast<uint8_t>(i & 0xFF);
    }
    // Slice the middle 1024 bytes starting at offset 1024.
    auto slice = std::make_shared<MetalBuffer>(root, /*offset=*/1024, /*size=*/1024);
    EXPECT_EQ(slice->offset(), 1024);
    EXPECT_EQ(slice->mtl_buffer(), nullptr);  ///< slice does not own MTLBuffer
    EXPECT_EQ(slice->root_mtl_buffer(), root->mtl_buffer());
    // Slice-of-slice: take middle 512 bytes of the outer slice.
    auto sub = std::make_shared<MetalBuffer>(slice, /*offset=*/256, /*size=*/512);
    EXPECT_EQ(sub->offset(), 1280);  ///< 1024 + 256, offsets compose
    EXPECT_EQ(sub->root_mtl_buffer(), root->mtl_buffer());
    // Dispatch byte-double against `sub`. DispatchByteDouble uses
    // root_mtl_buffer() + offset() so this works for nested slices.
    ASSERT_OK(DispatchByteDouble(*device_, *sub));
    // Verify only the sub's bytes were doubled; surrounding bytes untouched.
    for (int64_t i = 0; i < kSize; ++i) {
        const auto orig = static_cast<uint8_t>(i & 0xFF);
        const auto expected = (i >= 1280 && i < 1280 + 512)
                                  ? static_cast<uint8_t>(orig * 2)
                                  : orig;
        ASSERT_EQ(root->data()[i], expected) << "mismatch at " << i;
    }
}
TEST_F(MetalEnvironment, SyncEventConcurrentRecords) {
    ASSERT_OK_AND_ASSIGN(auto event, mm_->MakeDeviceSyncEvent());
    ASSERT_NE(event, nullptr);
    auto* mev = static_cast<MetalDevice::SyncEvent*>(event.get());
    constexpr int kThreads = 8;
    constexpr int kPerThread = 16;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    std::atomic<int> errors{0};
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto stream_result = device_->MakeStream();
            if (!stream_result.ok()) {
                errors.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            auto stream = stream_result.ValueOrDie();
            for (int i = 0; i < kPerThread; ++i) {
                if (!event->Record(*stream).ok()) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
            (void)t;
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(errors.load(), 0);
    // After kThreads * kPerThread Record() calls, the atomic counter
    // must equal exactly that count — proves no lost updates.
    EXPECT_EQ(mev->signal_value(), static_cast<uint64_t>(kThreads * kPerThread));
    // Final Wait should complete cleanly.
    ASSERT_OK(event->Wait());
}

}  // namespace metal
}  // namespace arrow
