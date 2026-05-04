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

#include <vulkan/vulkan.h>

#include "gtest/gtest.h"

#include "arrow/buffer.h"
#include "arrow/device.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/macros.h"
#include "arrow/vulkan/vulkan_api.h"
#include "arrow/vulkan/vulkan_internal.hh"

namespace arrow {
namespace vulkan {
/** --------------------------------------------------------------------------------------------- kDoubleUintSPV
 * @brief Pre-compiled SPIR-V for the GPU-coherence kernel.
 *
 * Source GLSL (compiled with glslangValidator -V):
 *   #version 450
 *   layout(local_size_x = 64) in;
 *   layout(set = 0, binding = 0) buffer Data { uint data[]; };
 *   layout(push_constant) uniform Pc { uint n; };
 *   void main() {
 *       uint gid = gl_GlobalInvocationID.x;
 *       if (gid < n) data[gid] = data[gid] * 2u;
 *   }
 *
 * Embedding the bytecode (286 uint32 = 1144 bytes) lets the test run
 * without invoking glslangValidator at runtime. Same kernel works on
 * RADV / Intel ANV / MoltenVK.
 */
static const uint32_t kDoubleUintSPV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000002c, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00646967, 0x00080005, 0x0000000b,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00030005, 0x00000011,
    0x00006350, 0x00040006, 0x00000011, 0x00000000, 0x0000006e, 0x00030005, 0x00000013, 0x00000000,
    0x00040005, 0x0000001e, 0x61746144, 0x00000000, 0x00050006, 0x0000001e, 0x00000000, 0x61746164,
    0x00000000, 0x00030005, 0x00000020, 0x00000000, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c,
    0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000,
    0x00040047, 0x0000001d, 0x00000006, 0x00000004, 0x00030047, 0x0000001e, 0x00000003, 0x00050048,
    0x0000001e, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000020, 0x00000021, 0x00000000,
    0x00040047, 0x00000020, 0x00000022, 0x00000000, 0x00040047, 0x0000002b, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000009, 0x00000006,
    0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b,
    0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001,
    0x00000006, 0x0003001e, 0x00000011, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011,
    0x0004003b, 0x00000012, 0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001,
    0x0004002b, 0x00000014, 0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000009, 0x00000006,
    0x00020014, 0x00000019, 0x0003001d, 0x0000001d, 0x00000006, 0x0003001e, 0x0000001e, 0x0000001d,
    0x00040020, 0x0000001f, 0x00000002, 0x0000001e, 0x0004003b, 0x0000001f, 0x00000020, 0x00000002,
    0x00040020, 0x00000023, 0x00000002, 0x00000006, 0x0004002b, 0x00000006, 0x00000026, 0x00000002,
    0x0004002b, 0x00000006, 0x00000029, 0x00000040, 0x0004002b, 0x00000006, 0x0000002a, 0x00000001,
    0x0006002c, 0x00000009, 0x0000002b, 0x00000029, 0x0000002a, 0x0000002a, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008,
    0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006,
    0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000010,
    0x00000008, 0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006,
    0x00000018, 0x00000017, 0x000500b0, 0x00000019, 0x0000001a, 0x00000010, 0x00000018, 0x000300f7,
    0x0000001c, 0x00000000, 0x000400fa, 0x0000001a, 0x0000001b, 0x0000001c, 0x000200f8, 0x0000001b,
    0x0004003d, 0x00000006, 0x00000021, 0x00000008, 0x0004003d, 0x00000006, 0x00000022, 0x00000008,
    0x00060041, 0x00000023, 0x00000024, 0x00000020, 0x00000015, 0x00000022, 0x0004003d, 0x00000006,
    0x00000025, 0x00000024, 0x00050084, 0x00000006, 0x00000027, 0x00000025, 0x00000026, 0x00060041,
    0x00000023, 0x00000028, 0x00000020, 0x00000015, 0x00000021, 0x0003003e, 0x00000028, 0x00000027,
    0x000200f9, 0x0000001c, 0x000200f8, 0x0000001c, 0x000100fd, 0x00010038,
};
static constexpr size_t kDoubleUintSPVBytes = sizeof(kDoubleUintSPV);
/** --------------------------------------------------------------------------------------------- MetalEnvironment
 * @brief Test fixture that resolves the system default VulkanDevice and exposes its
 *        default MemoryManager. Skips gracefully on hosts without a Vulkan device.
 */
class VulkanEnvironment : public ::testing::Test {
 protected:
    void SetUp() override {
        auto dev_result = VulkanDevice::Default();
        if (!dev_result.ok()) {
            GTEST_SKIP() << "No Vulkan device available: " << dev_result.status();
        }
        device_ = dev_result.ValueOrDie();
        mm_ = std::static_pointer_cast<VulkanMemoryManager>(device_->default_memory_manager());
    }
    std::shared_ptr<VulkanDevice> device_;
    std::shared_ptr<VulkanMemoryManager> mm_;
};
/** --------------------------------------------------------------------------------------------- DispatchUintDouble
 * @brief Run the embedded SPIR-V kernel that doubles each uint32 in the buffer.
 *
 * Used to prove GPU coherence: CPU writes through VulkanBuffer::data(),
 * GPU reads via VkBuffer, doubles, writes back, CPU re-reads to confirm
 * the GPU saw the original values and produced doubled output in the
 * same memory.
 */
static Status DispatchUintDouble(VulkanDevice& device, VulkanBuffer& target) {
    VkDevice dev = internal::FromOpaque<VkDevice>(device.vk_device());
    VkBuffer buf = internal::FromOpaque<VkBuffer>(target.root_vk_buffer());
    if (dev == VK_NULL_HANDLE || buf == VK_NULL_HANDLE) {
        return Status::Invalid("DispatchUintDouble: nil handles");
    }
    // Build descriptor set layout: single storage buffer at binding 0.
    VkDescriptorSetLayoutBinding dslb{};
    dslb.binding = 0;
    dslb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dslb.descriptorCount = 1;
    dslb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dsli{};
    dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = 1;
    dsli.pBindings = &dslb;
    VkDescriptorSetLayout dsl;
    ARROW_VK_RETURN_NOT_OK(vkCreateDescriptorSetLayout(dev, &dsli, nullptr, &dsl),
                           "vkCreateDescriptorSetLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    ARROW_VK_RETURN_NOT_OK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl),
                           "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = kDoubleUintSPVBytes;
    smi.pCode = kDoubleUintSPV;
    VkShaderModule sm;
    ARROW_VK_RETURN_NOT_OK(vkCreateShaderModule(dev, &smi, nullptr, &sm),
                           "vkCreateShaderModule");

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = pl;
    VkPipeline pipeline;
    ARROW_VK_RETURN_NOT_OK(
        vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline),
        "vkCreateComputePipelines");

    VkDescriptorPoolSize dps{};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    ARROW_VK_RETURN_NOT_OK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool),
                           "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsa{};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    ARROW_VK_RETURN_NOT_OK(vkAllocateDescriptorSets(dev, &dsa, &ds),
                           "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo dbi{};
    dbi.buffer = buf;
    dbi.offset = static_cast<VkDeviceSize>(target.offset());
    dbi.range = static_cast<VkDeviceSize>(target.size());
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = ds;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = device.queue_family_index();
    VkCommandPool pool;
    ARROW_VK_RETURN_NOT_OK(vkCreateCommandPool(dev, &pci, nullptr, &pool),
                           "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd;
    ARROW_VK_RETURN_NOT_OK(vkAllocateCommandBuffers(dev, &cbi, &cmd),
                           "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    const uint32_t n = static_cast<uint32_t>(target.size() / sizeof(uint32_t));
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);
    const uint32_t groups = (n + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);
    vkEndCommandBuffer(cmd);

    VkQueue q = internal::FromOpaque<VkQueue>(device.vk_queue());
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ARROW_VK_RETURN_NOT_OK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    ARROW_VK_RETURN_NOT_OK(vkQueueWaitIdle(q), "vkQueueWaitIdle");

    vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    return Status::OK();
}
TEST_F(VulkanEnvironment, DeviceProperties) {
    ASSERT_NE(device_, nullptr);
    EXPECT_STREQ(device_->type_name(), "arrow::vulkan::VulkanDevice");
    EXPECT_EQ(device_->device_type(), DeviceAllocationType::kVULKAN);
    EXPECT_TRUE(device_->is_cpu());  ///< integrated/CPU device — true on supported hardware
    EXPECT_TRUE(device_->Equals(*device_));
    EXPECT_NE(device_->device_id(), -1);
    EXPECT_FALSE(device_->device_name().empty());
    EXPECT_GT(device_->total_memory(), 0);
}
TEST_F(VulkanEnvironment, DeviceManagerSingleton) {
    ASSERT_OK_AND_ASSIGN(auto* mgr, VulkanDeviceManager::Instance());
    EXPECT_GE(mgr->num_devices(), 1);
    ASSERT_OK_AND_ASSIGN(auto dev0, mgr->GetDevice(0));
    EXPECT_EQ(dev0->device_type(), DeviceAllocationType::kVULKAN);
}
TEST_F(VulkanEnvironment, AllocateBuffer) {
    ASSERT_OK_AND_ASSIGN(auto buf, mm_->AllocateBuffer(4096));
    ASSERT_NE(buf->data(), nullptr);
    EXPECT_EQ(buf->size(), 4096);
    EXPECT_TRUE(buf->is_mutable());
    EXPECT_TRUE(buf->is_cpu());
    EXPECT_EQ(buf->device_type(), DeviceAllocationType::kVULKAN);
    EXPECT_TRUE(IsVulkanBuffer(*buf));
}
TEST_F(VulkanEnvironment, AllocateBufferZeroSize) {
    ASSERT_OK_AND_ASSIGN(auto buf, mm_->AllocateBuffer(0));
    EXPECT_NE(buf, nullptr);
    EXPECT_EQ(buf->device_type(), DeviceAllocationType::kVULKAN);
    EXPECT_EQ(buf->size(), 0);
    EXPECT_EQ(buf->capacity(), 0);
}
TEST_F(VulkanEnvironment, GPUSeesCPUWrites) {
    constexpr int64_t kElems = 256;
    constexpr int64_t kSize = kElems * sizeof(uint32_t);
    ASSERT_OK_AND_ASSIGN(auto unique_buf, mm_->AllocateBuffer(kSize));
    auto buf = std::shared_ptr<Buffer>(std::move(unique_buf));
    auto* vb = static_cast<VulkanBuffer*>(buf.get());
    auto* p = reinterpret_cast<uint32_t*>(buf->mutable_data());
    for (int64_t i = 0; i < kElems; ++i) {
        p[i] = static_cast<uint32_t>(i);
    }
    ASSERT_OK(DispatchUintDouble(*device_, *vb));
    for (int64_t i = 0; i < kElems; ++i) {
        ASSERT_EQ(p[i], static_cast<uint32_t>(i * 2)) << "mismatch at " << i;
    }
}
TEST_F(VulkanEnvironment, ZeroCopyHostView) {
    if (!device_->supports_imported_host_memory()) {
        GTEST_SKIP() << "VK_EXT_external_memory_host not supported on this device";
    }
    const int64_t align = device_->imported_host_pointer_alignment();
    ASSERT_GT(align, 0);
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, static_cast<size_t>(align),
                               static_cast<size_t>(align)), 0);
    auto* p = static_cast<uint32_t*>(aligned);
    const int64_t n = align / sizeof(uint32_t);
    for (int64_t i = 0; i < n; ++i) {
        p[i] = static_cast<uint32_t>(i);
    }
    {
        auto cpu_buf = std::make_shared<Buffer>(static_cast<const uint8_t*>(aligned),
                                                static_cast<int64_t>(align));
        ASSERT_OK_AND_ASSIGN(auto view, MemoryManager::ViewBuffer(cpu_buf, mm_));
        ASSERT_NE(view, nullptr);
        ASSERT_EQ(view->device_type(), DeviceAllocationType::kVULKAN);
        EXPECT_EQ(view->data(), cpu_buf->data());  ///< unified memory: identical pointer
        auto* mview = static_cast<VulkanBuffer*>(view.get());
        ASSERT_OK(DispatchUintDouble(*device_, *mview));
        // Verify host sees the doubled values.
        for (int64_t i = 0; i < n; ++i) {
            ASSERT_EQ(p[i], static_cast<uint32_t>(i * 2)) << "mismatch at " << i;
        }
    }
    std::free(aligned);
}
TEST_F(VulkanEnvironment, ViewBufferTo) {
    ASSERT_OK_AND_ASSIGN(auto unique_buf, mm_->AllocateBuffer(64));
    auto src = std::shared_ptr<Buffer>(std::move(unique_buf));
    std::memset(src->mutable_data(), 0x11, src->size());
    ASSERT_OK_AND_ASSIGN(auto cpu_view,
                         MemoryManager::ViewBuffer(src, default_cpu_memory_manager()));
    ASSERT_NE(cpu_view, nullptr);
    EXPECT_EQ(cpu_view->data(), src->data());
    EXPECT_EQ(cpu_view->device_type(), DeviceAllocationType::kCPU);
    EXPECT_EQ(cpu_view->data()[0], 0x11);
}
TEST_F(VulkanEnvironment, CopyBufferRoundTrip) {
    ASSERT_OK_AND_ASSIGN(auto src_owned, AllocateBuffer(128));
    std::memset(src_owned->mutable_data(), 0x33, src_owned->size());
    auto src = std::shared_ptr<Buffer>(std::move(src_owned));
    ASSERT_OK_AND_ASSIGN(auto on_vk, MemoryManager::CopyBuffer(src, mm_));
    EXPECT_EQ(on_vk->device_type(), DeviceAllocationType::kVULKAN);
    EXPECT_EQ(std::memcmp(on_vk->data(), src->data(), src->size()), 0);
    ASSERT_OK_AND_ASSIGN(auto back, MemoryManager::CopyBuffer(on_vk, default_cpu_memory_manager()));
    EXPECT_EQ(std::memcmp(back->data(), src->data(), src->size()), 0);
}
TEST_F(VulkanEnvironment, SyncEventRoundTrip) {
    ASSERT_OK_AND_ASSIGN(auto event, mm_->MakeDeviceSyncEvent());
    ASSERT_NE(event, nullptr);
    ASSERT_OK_AND_ASSIGN(auto stream, device_->MakeStream());
    ASSERT_NE(stream, nullptr);
    ASSERT_OK(event->Record(*stream));
    ASSERT_OK(event->Wait());
}
TEST_F(VulkanEnvironment, FromBufferDowncast) {
    ASSERT_OK_AND_ASSIGN(auto unique_buf, mm_->AllocateBuffer(8));
    auto buf = std::shared_ptr<Buffer>(std::move(unique_buf));
    ASSERT_OK_AND_ASSIGN(auto vb, VulkanBuffer::FromBuffer(buf));
    EXPECT_EQ(vb->data(), buf->data());
}
struct WrapTestState {
    int count;
};
static void WrapTestRelease(void* p, int64_t /*sz*/, void* ud) {
    auto* st = static_cast<WrapTestState*>(ud);
    ++st->count;
    std::free(p);
}
TEST_F(VulkanEnvironment, WrapInvokesDeallocatorOnce) {
    if (!device_->supports_imported_host_memory()) {
        GTEST_SKIP() << "VK_EXT_external_memory_host not supported on this device";
    }
    const int64_t align = device_->imported_host_pointer_alignment();
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, static_cast<size_t>(align),
                               static_cast<size_t>(align)), 0);
    WrapTestState st{0};
    {
        ASSERT_OK_AND_ASSIGN(auto wrapped,
                             VulkanBuffer::Wrap(mm_, aligned, align,
                                                &WrapTestRelease, &st));
        EXPECT_EQ(wrapped->data(), aligned);
        EXPECT_EQ(wrapped->size(), align);
        EXPECT_EQ(wrapped->parent(), nullptr);  ///< sole-owner contract
    }
    EXPECT_EQ(st.count, 1);
}
TEST_F(VulkanEnvironment, WrapRejectsMisaligned) {
    if (!device_->supports_imported_host_memory()) {
        GTEST_SKIP();
    }
    const int64_t align = device_->imported_host_pointer_alignment();
    void* aligned = nullptr;
    ASSERT_EQ(::posix_memalign(&aligned, static_cast<size_t>(align),
                               static_cast<size_t>(align * 2)), 0);
    auto* misaligned = static_cast<uint8_t*>(aligned) + 1;
    auto result = VulkanBuffer::Wrap(mm_, misaligned, align - 2, nullptr);
    EXPECT_FALSE(result.ok());
    std::free(aligned);
}
TEST_F(VulkanEnvironment, MemoryPoolAllocateFreeBalances) {
    auto* pool = VulkanMemoryPool::Instance();
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->backend_name(), "vulkan");
    const int64_t starting = pool->bytes_allocated();
    uint8_t* p1 = nullptr;
    uint8_t* p2 = nullptr;
    ASSERT_OK(pool->Allocate(1024, &p1));
    ASSERT_OK(pool->Allocate(2048, &p2));
    EXPECT_EQ(pool->bytes_allocated(), starting + 1024 + 2048);
    std::memset(p1, 0xCC, 1024);
    std::memset(p2, 0xDD, 2048);
    EXPECT_EQ(p1[100], 0xCC);
    EXPECT_EQ(p2[100], 0xDD);
    pool->Free(p1, 1024);
    pool->Free(p2, 2048);
    EXPECT_EQ(pool->bytes_allocated(), starting);
    EXPECT_GE(pool->num_allocations(), 2);
}
TEST_F(VulkanEnvironment, MemoryPoolReallocate) {
    auto* pool = VulkanMemoryPool::Instance();
    ASSERT_NE(pool, nullptr);
    uint8_t* p = nullptr;
    ASSERT_OK(pool->Allocate(64, &p));
    std::memset(p, 0xEE, 64);
    ASSERT_OK(pool->Reallocate(64, 256, &p));
    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(p[i], 0xEE) << "lost data on Reallocate at byte " << i;
    }
    pool->Free(p, 256);
}
TEST_F(VulkanEnvironment, NestedSliceGPUDispatch) {
    constexpr int64_t kElems = 1024;
    constexpr int64_t kSize = kElems * sizeof(uint32_t);
    ASSERT_OK_AND_ASSIGN(auto unique, mm_->AllocateBuffer(kSize));
    auto buf = std::shared_ptr<Buffer>(std::move(unique));
    ASSERT_OK_AND_ASSIGN(auto root, VulkanBuffer::FromBuffer(buf));
    auto* p = reinterpret_cast<uint32_t*>(root->mutable_data());
    for (int64_t i = 0; i < kElems; ++i) {
        p[i] = static_cast<uint32_t>(i);
    }
    // Slice the middle 256 elements (offset 256, size 256).
    auto slice = std::make_shared<VulkanBuffer>(
        root, /*offset=*/256 * sizeof(uint32_t),
        /*size=*/256 * sizeof(uint32_t));
    EXPECT_EQ(slice->offset(), 256 * sizeof(uint32_t));
    EXPECT_EQ(slice->vk_buffer(), nullptr);
    EXPECT_EQ(slice->root_vk_buffer(), root->vk_buffer());
    // Slice-of-slice: take middle 64 elements.
    auto sub = std::make_shared<VulkanBuffer>(
        slice, /*offset=*/96 * sizeof(uint32_t),
        /*size=*/64 * sizeof(uint32_t));
    EXPECT_EQ(sub->offset(), 352 * sizeof(uint32_t));  ///< 256 + 96
    EXPECT_EQ(sub->root_vk_buffer(), root->vk_buffer());
    ASSERT_OK(DispatchUintDouble(*device_, *sub));
    for (int64_t i = 0; i < kElems; ++i) {
        const uint32_t orig = static_cast<uint32_t>(i);
        const uint32_t expected = (i >= 352 && i < 352 + 64) ? orig * 2 : orig;
        ASSERT_EQ(p[i], expected) << "mismatch at " << i;
    }
}
TEST_F(VulkanEnvironment, SyncEventConcurrentRecords) {
    ASSERT_OK_AND_ASSIGN(auto event, mm_->MakeDeviceSyncEvent());
    ASSERT_NE(event, nullptr);
    auto* vev = static_cast<VulkanDevice::SyncEvent*>(event.get());
    constexpr int kThreads = 4;
    constexpr int kPerThread = 8;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    std::atomic<int> errors{0};
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
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
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(errors.load(), 0);
    EXPECT_EQ(vev->signal_value(), static_cast<uint64_t>(kThreads * kPerThread));
    ASSERT_OK(event->Wait());
}

}  // namespace vulkan
}  // namespace arrow
