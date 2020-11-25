//
//  VulkanBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanBuffer.hpp"
#include <string.h>
namespace MNN {

VulkanBuffer::VulkanBuffer(const VulkanMemoryPool& pool, bool seperate, size_t size, const void* hostData,
                           VkBufferUsageFlags usage, VkSharingMode shared, VkFlags requirements_mask)
    : mPool(pool) {
    MNN_ASSERT(size > 0);
    mSize = size;
    mShared = shared;
    mBuffer = const_cast<VulkanMemoryPool&>(mPool).allocBuffer(size, usage, shared);
    mUsage = usage;

    VkMemoryRequirements memReq;
    mPool.device().getBufferMemoryRequirements(mBuffer, memReq);
    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memReq, requirements_mask, seperate);
    //        FUNC_PRINT(mMemory->type());

    if (nullptr != hostData) {
        void* data = nullptr;
        CALL_VK(mPool.device().mapMemory(mMemory->get(), 0, size, 0, &data));
        ::memcpy(data, hostData, size);
        mPool.device().unmapMemory(mMemory->get());
    }
    CALL_VK(mPool.device().bindBufferMemory(mBuffer, mMemory->get()));
}

VulkanBuffer::~VulkanBuffer() {
    const_cast<VulkanMemoryPool&>(mPool).returnBuffer(mBuffer, mSize, mUsage, mShared);
    if (!mReleased) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    }
}
void* VulkanBuffer::map(int start, int size) const {
    const auto& limits = mPool.device().proty().limits;
    if (size < 0) {
        size = mSize;
    }
    size = UP_DIV(size, limits.nonCoherentAtomSize) * limits.nonCoherentAtomSize;
    void* data = nullptr;
    CALL_VK(mPool.device().mapMemory(mMemory->get(), start, size, 0, &data));
    return data;
}
void VulkanBuffer::unmap() const {
    mPool.device().unmapMemory(mMemory->get());
}
void VulkanBuffer::release() {
    if (mReleased) {
        return;
    }
    mReleased = true;
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
}

void VulkanBuffer::flush(bool write, int start, int size) const {
    VkMappedMemoryRange range;
    const auto& limits = mPool.device().proty().limits;
    range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = mMemory->get();
    range.offset = start;
    range.size   = UP_DIV(size, limits.nonCoherentAtomSize) * limits.nonCoherentAtomSize;
    range.pNext  = nullptr;

    if (write) {
        CALL_VK(mPool.device().flushMappedMemoryRanges(&range));
    } else {
        CALL_VK(mPool.device().invalidateMappedMemoryRanges(&range));
    }
}

} // namespace MNN
