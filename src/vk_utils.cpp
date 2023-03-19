#include "vk_utils.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <vk_initializers.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void vkutils::generate_mipmaps(VkCommandBuffer& cmd, VkImage image, int32_t width,
                               int32_t height, uint32_t mipLevels)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = width;
    int32_t mipHeight = height;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1,
                              mipHeight > 1 ? mipHeight / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &barrier);

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
}

bool vkutils::load_image_from_memory(EngineData* engineData, void* pixels, int width,
                                     int height, AllocatedImage& outImage,
                                     uint32_t& outMipLevels)
{
    outMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    VkDeviceSize imageSize = width * height * 4;

    // the format R8G8B8A8 matches exactly with the pixels loaded from stb_image lib
    VkFormat image_format = VK_FORMAT_R8G8B8A8_UNORM;

    // allocate temporary buffer for holding texture data to upload
    AllocatedBuffer stagingBuffer =
        vkutils::create_buffer(engineData->allocator, imageSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    // copy data to buffer
    void* data;
    vmaMapMemory(engineData->allocator, stagingBuffer._allocation, &data);

    memcpy(data, pixels, static_cast<size_t>(imageSize));

    vmaUnmapMemory(engineData->allocator, stagingBuffer._allocation);

    VkExtent3D imageExtent;
    imageExtent.width = static_cast<uint32_t>(width);
    imageExtent.height = static_cast<uint32_t>(height);
    imageExtent.depth = 1;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(
        image_format,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        imageExtent, outMipLevels);

    AllocatedImage newImage;

    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    // allocate and create the image
    vmaCreateImage(engineData->allocator, &dimg_info, &dimg_allocinfo, &newImage._image,
                   &newImage._allocation, nullptr);

    immediate_submit(engineData, [&](VkCommandBuffer cmd) {
        VkImageSubresourceRange range;
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel = 0;
        range.levelCount = outMipLevels;
        range.baseArrayLayer = 0;
        range.layerCount = 1;

        VkImageMemoryBarrier imageBarrier_toTransfer = {};
        imageBarrier_toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageBarrier_toTransfer.image = newImage._image;
        imageBarrier_toTransfer.subresourceRange = range;

        imageBarrier_toTransfer.srcAccessMask = 0;
        imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        // barrier the image into the transfer-receive layout
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &imageBarrier_toTransfer);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = imageExtent;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, newImage._image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        /*
            VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

            imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            //barrier the image into the shader readable layout
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
           &imageBarrier_toReadable);
            */
        generate_mipmaps(cmd, newImage._image, width, height, outMipLevels);
    });

    // TODO: MEMORY ISSUES
    // engine._mainDeletionQueue.push_function([=]() {
    //
    //	vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
    //	});

    vmaDestroyBuffer(engineData->allocator, stagingBuffer._buffer, stagingBuffer._allocation);

    std::cout << "Texture loaded successfully \n";

    outImage = newImage;

    return true;
}

VkCommandBuffer vkutils::create_command_buffer(VkDevice device, VkCommandPool commandPool,
                                               bool startRecording)
{
    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(commandPool, 1);

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd));

    if (startRecording)
    {
        VkCommandBufferBeginInfo cmdBeginInfo =
            vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    }

    return cmd;
}

void vkutils::submit_and_free_command_buffer(VkDevice device, VkCommandPool commandPool,
                                             VkCommandBuffer cmd, VkQueue queue, VkFence fence)
{
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = vkinit::submit_info(&cmd);
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device, 1, &fence, true, 999999999999));
    VK_CHECK(vkResetFences(device, 1, &fence));

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

AllocatedBuffer vkutils::create_buffer(VmaAllocator allocator, size_t allocSize,
                                       VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                                       VmaAllocationCreateFlags allocationFlags)
{
    // allocate vertex buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;

    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = allocationFlags;

    // experimental
    if (vmaallocInfo.usage == VMA_MEMORY_USAGE_GPU_ONLY)
    {
        vmaallocInfo.requiredFlags =
            VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    AllocatedBuffer newBuffer;

    // allocate the buffer
    vmaCreateBuffer(allocator, &bufferInfo, &vmaallocInfo, &newBuffer._buffer,
                    &newBuffer._allocation, nullptr);

    newBuffer._descriptorBufferInfo.buffer = newBuffer._buffer;
    newBuffer._descriptorBufferInfo.offset = 0;
    newBuffer._descriptorBufferInfo.range = allocSize;

    return newBuffer;
}

void vkutils::cpu_to_gpu(VmaAllocator allocator, AllocatedBuffer& allocatedBuffer, void* data,
                         size_t size)
{
    void* gpuData;
    vmaMapMemory(allocator, allocatedBuffer._allocation, &gpuData);
    memcpy(gpuData, data, size);
    vmaUnmapMemory(allocator, allocatedBuffer._allocation);
}

void vkutils::cpu_to_gpu_staging(VmaAllocator allocator, VkCommandBuffer commandBuffer,
                                 AllocatedBuffer& allocatedBuffer, void* data, size_t size)
{
    AllocatedBuffer stagingBuffer = create_buffer(
        allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    cpu_to_gpu(allocator, stagingBuffer, data, size);

    VkBufferCopy cpy;
    cpy.size = size;
    cpy.srcOffset = 0;
    cpy.dstOffset = 0;

    vkCmdCopyBuffer(commandBuffer, stagingBuffer._buffer, allocatedBuffer._buffer, 1, &cpy);
    // TODO: We have to remove staging buffer
}

bool vkutils::load_shader_module(VkDevice device, const char* filePath,
                                 VkShaderModule* outShaderModule)
{
    // open the file. With cursor at the end
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        return false;
    }

    // find what the size of the file is by looking up the location of the cursor
    // because the cursor is at the end, it gives the size directly in bytes
    size_t fileSize = (size_t)file.tellg();

    // spirv expects the buffer to be on uint32, so make sure to reserve an int vector big
    // enough for the entire file
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    // put file cursor at beginning
    file.seekg(0);

    // load the entire file into the buffer
    file.read((char*)buffer.data(), fileSize);

    // now that the file is loaded into the buffer, we can close it
    file.close();

    // create a new shader module, using the buffer we loaded
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;

    // codeSize has to be in bytes, so multiply the ints in the buffer by size of int to know
    // the real size of the buffer
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    // check that the creation goes well.
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}

void vkutils::cmd_viewport_scissor(VkCommandBuffer cmd, VkExtent2D extent)
{
    VkViewport viewport;
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)extent.width;
    viewport.height = (float)extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor;
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

AllocatedBuffer vkutils::create_upload_buffer(EngineData* engineData, void* buffer_data,
                                              size_t size, VkBufferUsageFlags usage,
                                              VmaMemoryUsage memoryUsage,
                                              VmaAllocationCreateFlags allocationFlags)
{
    AllocatedBuffer stagingBuffer =
        vkutils::create_buffer(engineData->allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VMA_MEMORY_USAGE_CPU_ONLY);

    vkutils::cpu_to_gpu(engineData->allocator, stagingBuffer, buffer_data, size);

    AllocatedBuffer new_buffer = vkutils::create_buffer(
        engineData->allocator, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memoryUsage,
        allocationFlags);

    vkutils::immediate_submit(engineData, [=](VkCommandBuffer cmd) {
        VkBufferCopy copy;
        copy.dstOffset = 0;
        copy.srcOffset = 0;
        copy.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer._buffer, new_buffer._buffer, 1, &copy);
    });

    vmaDestroyBuffer(engineData->allocator, stagingBuffer._buffer, stagingBuffer._allocation);

    return new_buffer;
}

void vkutils::immediate_submit(EngineData* engineData,
                               std::function<void(VkCommandBuffer cmd)>&& function)
{
    // allocate the default command buffer that we will use for the instant commands
    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(engineData->uploadContext.commandPool, 1);

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(engineData->device, &cmdAllocInfo, &cmd));

    // begin the command buffer recording. We will use this command buffer exactly once, so we
    // want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // execute the function
    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = vkinit::submit_info(&cmd);

    // submit command buffer to the queue and execute it.
    //  _uploadFence will now block until the graphic commands finish execution
    VK_CHECK(
        vkQueueSubmit(engineData->graphicsQueue, 1, &submit, engineData->uploadContext.fence));

    vkWaitForFences(engineData->device, 1, &engineData->uploadContext.fence, true, 9999999999);
    vkResetFences(engineData->device, 1, &engineData->uploadContext.fence);

    // clear the command pool. This will free the command buffer too
    vkResetCommandPool(engineData->device, engineData->uploadContext.commandPool, 0);
}

AllocatedImage vkutils::create_image(EngineData* engineData, VkFormat format,
                                     VkImageUsageFlags usageFlags, VkExtent3D extent,
                                     uint32_t mipLevels)
{
    AllocatedImage allocatedImage;

    VkImageCreateInfo dimg_info =
        vkinit::image_create_info(format, usageFlags, extent, mipLevels);
    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vmaCreateImage(engineData->allocator, &dimg_info, &dimg_allocinfo,
                            &allocatedImage._image, &allocatedImage._allocation, nullptr));

    allocatedImage.mips = mipLevels;
    allocatedImage.format = format;

    return allocatedImage;
}

VkImageView vkutils::create_image_view(EngineData* engineData, AllocatedImage& allocatedImage,
                                       VkFormat format, VkImageAspectFlagBits aspectFlags,
                                       uint32_t mipLevels)
{
    VkImageView imageView;
    VkImageViewCreateInfo imageViewInfo =
        vkinit::imageview_create_info(format, allocatedImage._image, aspectFlags, mipLevels);
    VK_CHECK(vkCreateImageView(engineData->device, &imageViewInfo, nullptr, &imageView));
    return imageView;
}

void vkutils::image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                            VkImageLayout newLayout,
                            VkImageSubresourceRange imageSubresourceRange,
                            VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                            VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier imageMemoryBarrier = {};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.oldLayout = oldLayout;
    imageMemoryBarrier.newLayout = newLayout;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = imageSubresourceRange;
    imageMemoryBarrier.srcAccessMask = srcAccess;
    imageMemoryBarrier.dstAccessMask = dstAccess;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                         &imageMemoryBarrier);
}

void vkutils::memory_barrier(VkCommandBuffer cmd, VkBuffer buffer, VkAccessFlags srcAccess,
                             VkAccessFlags dstAccess, VkPipelineStageFlags srcStage,
                             VkPipelineStageFlags dstStage)
{
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.size = VK_WHOLE_SIZE;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void vkutils::setObjectName(VkDevice device, const uint64_t object, const std::string& name,
                            VkObjectType t)
{
    VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                    nullptr, t, object, name.c_str()};
    vkSetDebugUtilsObjectNameEXT(device, &s);
}

void vkutils::setObjectName(VkDevice device, VkBuffer object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_BUFFER);
}
void vkutils::setObjectName(VkDevice device, VkBufferView object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_BUFFER_VIEW);
}
void vkutils::setObjectName(VkDevice device, VkCommandBuffer object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_BUFFER);
}
void vkutils::setObjectName(VkDevice device, VkCommandPool object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_POOL);
}
void vkutils::setObjectName(VkDevice device, VkDescriptorPool object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
}
void vkutils::setObjectName(VkDevice device, VkDescriptorSet object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET);
}
void vkutils::setObjectName(VkDevice device, VkDescriptorSetLayout object,
                            const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
}
void vkutils::setObjectName(VkDevice device, VkDevice object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DEVICE);
}
void vkutils::setObjectName(VkDevice device, VkDeviceMemory object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_DEVICE_MEMORY);
}
void vkutils::setObjectName(VkDevice device, VkFramebuffer object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_FRAMEBUFFER);
}
void vkutils::setObjectName(VkDevice device, VkImage object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_IMAGE);
}
void vkutils::setObjectName(VkDevice device, VkImageView object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_IMAGE_VIEW);
}
void vkutils::setObjectName(VkDevice device, VkPipeline object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE);
}
void vkutils::setObjectName(VkDevice device, VkPipelineLayout object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE_LAYOUT);
}
void vkutils::setObjectName(VkDevice device, VkQueryPool object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_QUERY_POOL);
}
void vkutils::setObjectName(VkDevice device, VkQueue object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_QUEUE);
}
void vkutils::setObjectName(VkDevice device, VkRenderPass object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_RENDER_PASS);
}
void vkutils::setObjectName(VkDevice device, VkSampler object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SAMPLER);
}
void vkutils::setObjectName(VkDevice device, VkSemaphore object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SEMAPHORE);
}
void vkutils::setObjectName(VkDevice device, VkShaderModule object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SHADER_MODULE);
}
void vkutils::setObjectName(VkDevice device, VkSwapchainKHR object, const std::string& name)
{
    setObjectName(device, (uint64_t)object, name, VK_OBJECT_TYPE_SWAPCHAIN_KHR);
}

void vkutils::screenshot(EngineData* engineData, const char* filename, VkImage srcImage,
                         VkExtent2D size)
{
    AllocatedImage dstImage;

    VkImageCreateInfo dimg_info =
        vkinit::image_create_info(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                  {size.width, size.height, 1});
    dimg_info.tiling = VK_IMAGE_TILING_LINEAR;
    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    vmaCreateImage(engineData->allocator, &dimg_info, &dimg_allocinfo, &dstImage._image,
                   &dstImage._allocation, nullptr);

    vkutils::immediate_submit(engineData, [&](VkCommandBuffer cmd) {
        {
            VkImageMemoryBarrier imageMemoryBarrier = {};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.image = dstImage._image;
            imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            imageMemoryBarrier.srcAccessMask = 0;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &imageMemoryBarrier);
        }

        {
            VkImageMemoryBarrier imageMemoryBarrier = {};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            imageMemoryBarrier.image = srcImage;
            imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &imageMemoryBarrier);
        }

        // Define the region to blit (we will blit the whole swapchain image)
        // Otherwise use image copy (requires us to manually flip components)
        VkImageCopy imageCopyRegion{};
        imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopyRegion.srcSubresource.layerCount = 1;
        imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopyRegion.dstSubresource.layerCount = 1;
        imageCopyRegion.extent.width = size.width;
        imageCopyRegion.extent.height = size.height;
        imageCopyRegion.extent.depth = 1;

        // Issue the copy command
        vkCmdCopyImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage._image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);

        // Transition destination image to general layout, which is the required layout for
        // mapping the image memory later on
        {
            VkImageMemoryBarrier imageMemoryBarrier = {};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageMemoryBarrier.image = dstImage._image;
            imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &imageMemoryBarrier);
        }

        // Transition back the swap chain image after the blit is done
        {
            VkImageMemoryBarrier imageMemoryBarrier = {};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            imageMemoryBarrier.image = srcImage;
            imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &imageMemoryBarrier);
        }
    });

    // Get layout of the image (including row pitch)
    VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(engineData->device, dstImage._image, &subResource,
                                &subResourceLayout);

    // Map image memory so we can start copying from it
    const char* data;
    vmaMapMemory(engineData->allocator, dstImage._allocation, (void**)&data);
    data += subResourceLayout.offset;

    char* newdata = new char[size.width * size.height * 4];
    memcpy(newdata, data, size.width * size.height * 4);

    vmaUnmapMemory(engineData->allocator, dstImage._allocation);

    for (int i = 0; i < size.width * size.height * 4; i += 4)
    {
        char temp = newdata[i + 0];
        newdata[i + 0] = newdata[i + 2];
        newdata[i + 2] = temp;
    }

    // save data
    stbi_write_png(filename, size.width, size.height, 4, newdata, size.width * 4);

    delete[] newdata;
    vmaDestroyImage(engineData->allocator, dstImage._image, dstImage._allocation);
}