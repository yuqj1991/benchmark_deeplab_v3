//
//  VulkanBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBackend.hpp"
#include <mutex>
#include "core/Execution.hpp"
#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "component/VulkanDevice.hpp"
#include "execution/VulkanImageConverter.hpp"
#include "component/VulkanInstance.hpp"
#include "execution/VulkanBasicExecution.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#define MNN_OP_SUPPORT_LOG
//#define MNN_VULKAN_DUMP_MEMORY_USAGE

namespace MNN {

static std::map<OpType, VulkanBackend::Creator*>* gCreator = nullptr;

// Creator
static inline std::map<OpType, VulkanBackend::Creator*>* getCreatorMap() {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, VulkanBackend::Creator*>();
    }
    return gCreator;
}

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source) {
    auto sourcePtr   = source->map();
    auto dataType    = dest->getType();
    //TODO: Support other kind of dataType
    MNN_ASSERT(dataType.bits == 32);
    ::memcpy(dest->host<float>(), sourcePtr, dest->size());
    source->unmap();
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest) {
    auto destPtr     = dest->map();
    auto dataType    = source->getType();
    //TODO: Support other kind of dataType
    MNN_ASSERT(dataType.bits == 32);
    ::memcpy(destPtr, source->host<float>(), source->size());
    dest->unmap();
}

std::pair<float, bool> VulkanBackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    auto creator = getCreatorMap();
    auto iter    = creator->find(op->type());
    if (iter == creator->end()) {
        return std::make_pair(0.0f, false);
    }
#ifndef MNN_BUILD_MINI
    auto flops = SizeComputer::computeFlops(op, inputs, outputs);
#else
    auto flops = 0.0f;
#endif
    const float defaultScheduleCost = 0.001f;
    return std::make_pair(defaultScheduleCost + flops / 1024.0f / mRuntime->mFlops * 1000.0f, true);
}
VulkanBackend::VulkanBackend(const VulkanRuntime* runtime, const Backend::Info& info) : Backend(MNN_FORWARD_VULKAN) {
    mRuntime = runtime;
    mDirect = Backend::Info::INDIRECT != info.mode;

    auto& dev              = device();
    mFence                 = std::make_shared<VulkanFence>(dev);
    if (!mDirect) {
        mCmdBuffer.reset(runtime->mCmdPool->allocBuffer());
    }
}

VulkanBackend::~VulkanBackend() {
    /*keep release order*/
    mCmdBuffer = nullptr;

    mStaticeBuffers.clear();
    mAllBuffers.clear();

    mHostBuffer = nullptr;
    mCmdBuffers.clear();
    mFence = nullptr;
    mConverters.clear();
}
void VulkanBackend::pushCommand(VkCommandBuffer buffer) const {
    mCmdBuffers.emplace_back(buffer);
//    _finish();
}

const VulkanPipeline* VulkanBackend::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                 const std::vector<uint32_t>& localSize) const {
    return mRuntime->mPipelineFactory->getPipeline(key, types, localSize);
}

bool VulkanBackend::_supportImageSize(const Tensor* MTensor) {
    if (MTensor->getType().code != halide_type_float) {
        return false;
    }
    auto format = TensorUtils::getDescribe(MTensor)->dimensionFormat;
    if (format != MNN_DATA_FORMAT_NC4HW4) {
        return true;
    }
    if (MTensor->dimensions() > 4) {
        return false;
    }
    if (UP_DIV(MTensor->channel(), 4) * MTensor->batch() > device().proty().limits.maxImageDimension3D) {
        return false;
    }
    return true;
}
void VulkanBackend::onResizeBegin() {
    if (!mDirect) {
        mCmdBuffer->begin(0);
    }
}
void VulkanBackend::onResizeEnd() {
    if (!mDirect) {
        mCmdBuffer->end();
    }
}

bool VulkanBackend::onAcquireBuffer(const Tensor* tensor, StorageType storageType) {
    //FUNC_PRINT_ALL(tensor, p);

    auto MTensor     = const_cast<Tensor*>(tensor);
    if (Backend::STATIC == storageType) {
        auto newBuffer           = std::make_shared<VulkanTensor>(MTensor, getMemoryPool(), device().proty().limits);
        MTensor->buffer().device = (uint64_t)(newBuffer.get());
        mStaticeBuffers.insert(std::make_pair(MTensor->buffer().device, newBuffer));
    } else {
        bool seperate  = storageType == Backend::DYNAMIC_SEPERATE;
        auto newBuffer = std::make_shared<VulkanTensor>(MTensor, getDynamicMemoryPool(), device().proty().limits, seperate);
        MTensor->buffer().device = (uint64_t)(newBuffer.get());
        mAllBuffers.insert(std::make_pair(MTensor->buffer().device, newBuffer));
    }
    return true;
}
bool VulkanBackend::onReleaseBuffer(const Tensor* tensor, StorageType storageType) {
    auto buffer = (tensor->deviceId());
    if (Backend::DYNAMIC == storageType) {
        auto iter = mAllBuffers.find(buffer);
        MNN_ASSERT(iter != mAllBuffers.end());
        iter->second->release();
    }
    if (Backend::STATIC == storageType) {
        auto iter = mStaticeBuffers.find(buffer);
        MNN_ASSERT(iter != mStaticeBuffers.end());
        iter->second->release();
        mStaticeBuffers.erase(iter);
    }
    return true;
}
bool VulkanBackend::onClearBuffer() {
    mAllBuffers.clear();
    return true;
}
Execution* VulkanBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
    auto creator = getCreatorMap();
    auto iter    = creator->find(op->type());
    std::string name = "";
    if (nullptr != op->name()) {
        name = op->name()->str();
    }
    if (iter == creator->end()) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_PRINT("Vulkan don't support %d, %s: %s\n", op->type(), EnumNameOpType(op->type()),
                name.c_str());
#endif
        return nullptr;
    }
    bool valid = true;
#ifndef MNN_BUILD_MINI
    for (int i=0; i<inputs.size(); ++i) {
        if (!SizeComputer::opNeedContent(op->type(), i)) {
            continue;
        }
        auto t = inputs[i];
        auto inputDes = TensorUtils::getDescribe(t);
        if (inputDes->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
            for (auto& r : inputDes->regions) {
                if (!_supportImageSize(r.origin)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                break;
            }
        } else {
            if (!_supportImageSize(t)) {
                valid = false;
                break;
            }
        }
    }
#endif
    for (auto t : outputs) {
        if (!_supportImageSize(t)) {
            valid = false;
            break;
        }
    }
    if (!valid) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Tensor not support\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }
    auto originExecution = (VulkanBasicExecution*)iter->second->onCreate(inputs, outputs, op, this);
    if (nullptr == originExecution) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Special case\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }
    if (mDirect) {
        return new VulkanBasicExecutionDirect(std::shared_ptr<VulkanBasicExecution>(originExecution));
    }
    return new VulkanBasicExecutionInDirect(std::shared_ptr<VulkanBasicExecution>(originExecution));
}

void VulkanBackend::onExecuteBegin() const {
    if (!mDirect) {
        mCmdBuffers.push_back(mCmdBuffer->get());
    }
    // FUNC_PRINT_ALL(mDynamicMemoryPool->computeSize(), f);
}
void VulkanBackend::onExecuteEnd() const {
    _finish();
}
void VulkanBackend::_finish() const {
    if (mCmdBuffers.empty()) {
        return;
    }
    VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                /* .pNext                = */ nullptr,
                                /* .waitSemaphoreCount   = */ 0,
                                /* .pWaitSemaphores      = */ nullptr,
                                /* .pWaitDstStageMask    = */ nullptr,
                                /* .commandBufferCount   = */ (uint32_t)mCmdBuffers.size(),
                                /* .pCommandBuffers      = */ mCmdBuffers.data(),
                                /* .signalSemaphoreCount = */ 0,
                                /* .pSignalSemaphores    = */ nullptr};
    auto fenceReal           = mFence->get();
    mFence->reset();
    CALL_VK(vkQueueSubmit(device().acquireDefaultDevQueue(), 1, &submit_info, fenceReal));

    auto res = mFence->wait();
    MNN_VK_CHECK(res);
    mCmdBuffers.clear();
}

const VulkanDevice& VulkanBackend::device() const {
    return (* mRuntime->mDevice);
}

void VulkanBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef MNN_VULKAN_DEBUG
    AUTOTIME;
    MNN_PRINT("Src: ");
    for (int i=0; i<srcTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", srcTensor->length(i));
    }
    MNN_PRINT("\n");
    MNN_PRINT("Dst: ");
    for (int i=0; i<dstTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", dstTensor->length(i));
    }
    MNN_PRINT("\n");
#endif
    if (srcTensor->host<float>() != nullptr) {
        _finish();
        auto size = VulkanTensor::getAlignSize(srcTensor) * 4;
        // host->gpu
        _allocHostBuffer(size);
        _copyTensorToBuffer(srcTensor, mHostBuffer.get());
        auto format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
        auto key    = std::make_tuple(dstTensor, true, format);
        auto iter   = mConverters.find(key);
        if (iter == mConverters.end()) {
            auto converter = std::make_shared<VulkanImageConverter>(this);
            std::shared_ptr<VulkanCommandPool::Buffer> convertorBuffer(
                                                                       const_cast<VulkanCommandPool::Buffer*>(getPool().allocBuffer()));
            convertorBuffer->begin(0);
            converter->encodeBufferToTensor(mHostBuffer->buffer(), dstTensor, mHostBuffer->size(), 0,
                                            TensorUtils::getDescribe(srcTensor)->dimensionFormat,
                                            convertorBuffer.get());
            convertorBuffer->end();
            mConverters.insert(std::make_pair(key, std::make_pair(converter, convertorBuffer)));
            iter = mConverters.find(key);
        }
        mCmdBuffers.push_back(iter->second.second->get());
        _finish();
    } else {
        // gpu->host
        auto size = VulkanTensor::getAlignSize(dstTensor) * 4;
        _finish();
        _allocHostBuffer(size);
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        auto key    = std::make_tuple(srcTensor, false, format);

        auto iter = mConverters.find(key);
        if (iter == mConverters.end()) {
            auto converter = std::make_shared<VulkanImageConverter>(this);
            std::shared_ptr<VulkanCommandPool::Buffer> convertorBuffer(
                                                                       const_cast<VulkanCommandPool::Buffer*>(getPool().allocBuffer()));
            convertorBuffer->begin(0);
            converter->encodeTensorToBuffer(srcTensor, mHostBuffer->buffer(), mHostBuffer->size(), 0,
                                            TensorUtils::getDescribe(dstTensor)->dimensionFormat,
                                            convertorBuffer.get());
            convertorBuffer->end();
            mConverters.insert(std::make_pair(key, std::make_pair(converter, convertorBuffer)));
            iter = mConverters.find(key);
        }
        mCmdBuffers.push_back(iter->second.second->get());
        _finish();
        _copyBufferToTensor(dstTensor, mHostBuffer.get());
    }
}

void VulkanBackend::_allocHostBuffer(size_t size) const {
    if (mHostBuffer.get() == nullptr || mHostBuffer->size() < size) {
        mHostBuffer.reset(new VulkanBuffer(getMemoryPool(), false, size, nullptr,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
        mConverters.clear();
    }
}
bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

void VulkanBackend::copyBufferToImage(const VulkanBuffer* buffer, const VulkanImage* image, VkImageLayout finalLayout) const {
    std::vector<int> dimVector = image->dims();
    if (image->format() != VK_FORMAT_R16G16B16A16_SFLOAT) {
        VkBufferImageCopy copyRegions;
        ::memset(&copyRegions, 0, sizeof(copyRegions));
        copyRegions.imageOffset.x                   = 0;
        copyRegions.imageOffset.y                   = 0;
        copyRegions.imageOffset.z                   = 0;
        copyRegions.imageExtent.depth               = image->depth();
        copyRegions.imageExtent.height              = image->height();
        copyRegions.imageExtent.width               = image->width();
        copyRegions.imageSubresource.layerCount     = 1;
        copyRegions.imageSubresource.mipLevel       = 0;
        copyRegions.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegions.imageSubresource.baseArrayLayer = 0;

        std::unique_ptr<VulkanCommandPool::Buffer> cmdbuffer(
                                                             const_cast<VulkanCommandPool::Buffer*>(getPool().allocBuffer()));
        cmdbuffer->begin(0);
        cmdbuffer->barrierImageIfNeeded(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmdbuffer->get(), buffer->buffer(), image->get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &copyRegions);
        if (finalLayout != VK_IMAGE_LAYOUT_UNDEFINED)
            cmdbuffer->barrierImageIfNeeded(image, finalLayout);
        cmdbuffer->end();
        getPool().submitAndWait(cmdbuffer->get());
    }

    const VulkanPipeline* transformPipeline = nullptr;
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    int localX = 16;
    int localY = 16;
    int localZ = 1;
    switch (dimVector.size()) {
        case 1:
            transformPipeline = getPipeline("glsl_buffer2Image1D_comp",
                                            /*glsl_buffer2Image1D_comp, glsl_buffer2Image1D_comp_len,*/ types);
            localX            = 256;
            localY            = 1;
            break;
        case 2:
            transformPipeline = getPipeline("glsl_buffer2Image2D_comp",
                                            /*glsl_buffer2Image2D_comp, glsl_buffer2Image2D_comp_len,*/ types);
            break;
        case 3:
            transformPipeline = getPipeline("glsl_buffer2Image3D_comp",
                                            /*glsl_buffer2Image3D_comp, glsl_buffer2Image3D_comp_len,*/ types);
            break;
        default:
            break;
    }

    std::unique_ptr<VulkanPipeline::DescriptorSet> sets(transformPipeline->createSet());
    auto constBuffer = std::make_shared<VulkanBuffer>(getMemoryPool(), false, dimVector.size() * sizeof(int),
                                                      dimVector.data(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    sets->writeImage(image->view(), mRuntime->mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    sets->writeBuffer(buffer->buffer(), 1, buffer->size());
    sets->writeBuffer(constBuffer->buffer(), 2, constBuffer->size());

    std::unique_ptr<VulkanCommandPool::Buffer> cmdbuffer(
        const_cast<VulkanCommandPool::Buffer*>(mRuntime->mCmdPool->allocBuffer()));
    cmdbuffer->begin(0);
    transformPipeline->bind(cmdbuffer->get(), sets->get());
    vkCmdDispatch(cmdbuffer->get(), UP_DIV(image->width(), localX), UP_DIV(image->height(), localY),
                  UP_DIV(image->depth(), localZ));
    if (finalLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        cmdbuffer->barrierImageIfNeeded(image, finalLayout);
    }
    cmdbuffer->end();
    mRuntime->mCmdPool->submitAndWait(cmdbuffer->get());
}


} // namespace MNN
