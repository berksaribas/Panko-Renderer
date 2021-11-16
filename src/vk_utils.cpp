#include "vk_utils.h"
#include <vk_initializers.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <vk_initializers.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void generate_mipmaps(VkCommandBuffer& cmd, VkImage image, int32_t width, int32_t height, uint32_t mipLevels) {
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

	for (uint32_t i = 1; i < mipLevels; i++) {
		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VkImageBlit blit{};
		blit.srcOffsets[0] = { 0, 0, 0 };
		blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = i - 1;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = 1;
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = i;
		blit.dstSubresource.baseArrayLayer = 0;
		blit.dstSubresource.layerCount = 1;

		vkCmdBlitImage(cmd,
			image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit,
			VK_FILTER_LINEAR);

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}

	barrier.subresourceRange.baseMipLevel = mipLevels - 1;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		0, nullptr,
		0, nullptr,
		1, &barrier);
}

bool vkutils::load_image_from_memory(VulkanEngine& engine, void* pixels, int width, int height, AllocatedImage& outImage, uint32_t& outMipLevels)
{
	outMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
	
	VkDeviceSize imageSize = width * height * 4;
	
	//the format R8G8B8A8 matches exactly with the pixels loaded from stb_image lib
	VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB;
	
	//allocate temporary buffer for holding texture data to upload
	AllocatedBuffer stagingBuffer = vkutils::create_buffer(engine._allocator, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	
	//copy data to buffer
	void* data;
	vmaMapMemory(engine._allocator, stagingBuffer._allocation, &data);
	
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	
	vmaUnmapMemory(engine._allocator, stagingBuffer._allocation);
	
	VkExtent3D imageExtent;
	imageExtent.width = static_cast<uint32_t>(width);
	imageExtent.height = static_cast<uint32_t>(height);
	imageExtent.depth = 1;
	
	VkImageCreateInfo dimg_info = vkinit::image_create_info(image_format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, imageExtent, outMipLevels);
	
	AllocatedImage newImage;
	
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	
	//allocate and create the image
	vmaCreateImage(engine._allocator, &dimg_info, &dimg_allocinfo, &newImage._image, &newImage._allocation, nullptr);
	
	engine.immediate_submit([&](VkCommandBuffer cmd) {
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
		
		//barrier the image into the transfer-receive layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);
		
		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = imageExtent;
		
		//copy the buffer into the image
		vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, newImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
		
		/*
			VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;
			
			imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			
			imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			
			//barrier the image into the shader readable layout
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
			*/
		generate_mipmaps(cmd, newImage._image, width, height, outMipLevels);
	});
	
	// TODO: MEMORY ISSUES
	//engine._mainDeletionQueue.push_function([=]() {
	//
	//	vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
	//	});
	
	vmaDestroyBuffer(engine._allocator, stagingBuffer._buffer, stagingBuffer._allocation);
	
	std::cout << "Texture loaded successfully \n";
	
	outImage = newImage;
	
	return true;
}

VkCommandBuffer vkutils::create_command_buffer(VkDevice device, VkCommandPool commandPool, bool startRecording)
{
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(commandPool, 1);

	VkCommandBuffer cmd;
	VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd));

	if (startRecording) {
		VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
	}

	return cmd;
}

void vkutils::submit_and_free_command_buffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer cmd, VkQueue queue, VkFence fence)
{
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));

	vkWaitForFences(device, 1, &fence, true, 9999999999);
	vkResetFences(device, 1, &fence);

	vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

AllocatedBuffer vkutils::create_buffer(VmaAllocator allocator, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags allocationFlags)
{
	//allocate vertex buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;

	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;

	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = memoryUsage;
	vmaallocInfo.flags = allocationFlags;

	//experimental
	if (vmaallocInfo.usage == VMA_MEMORY_USAGE_GPU_ONLY) {
		vmaallocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	AllocatedBuffer newBuffer;

	//allocate the buffer
	vmaCreateBuffer(allocator, &bufferInfo, &vmaallocInfo,
		&newBuffer._buffer,
		&newBuffer._allocation,
		nullptr);

	
	newBuffer._descriptorBufferInfo.buffer = newBuffer._buffer;
	newBuffer._descriptorBufferInfo.offset = 0;
	newBuffer._descriptorBufferInfo.range = allocSize;

	return newBuffer;
}

void vkutils::cpu_to_gpu(VmaAllocator allocator, AllocatedBuffer& allocatedBuffer, void* data, size_t size)
{
	void* gpuData;
	vmaMapMemory(allocator, allocatedBuffer._allocation, &gpuData);
	memcpy(gpuData, data, size);
	vmaUnmapMemory(allocator, allocatedBuffer._allocation);
}

void vkutils::cpu_to_gpu_staging(VmaAllocator allocator, VkCommandBuffer commandBuffer, AllocatedBuffer& allocatedBuffer, void* data, size_t size)
{
	AllocatedBuffer stagingBuffer = create_buffer(allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	cpu_to_gpu(allocator, stagingBuffer, data, size);

	VkBufferCopy cpy;
	cpy.size = size;
	cpy.srcOffset = 0;
	cpy.dstOffset = 0;

	vkCmdCopyBuffer(commandBuffer, stagingBuffer._buffer, allocatedBuffer._buffer, 1, &cpy);
	//TODO: We have to remove staging buffer
}

bool vkutils::load_shader_module(VkDevice device, const char* filePath, VkShaderModule* outShaderModule)
{
	//open the file. With cursor at the end
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		return false;
	}

	//find what the size of the file is by looking up the location of the cursor
//because the cursor is at the end, it gives the size directly in bytes
	size_t fileSize = (size_t)file.tellg();

	//spirv expects the buffer to be on uint32, so make sure to reserve an int vector big enough for the entire file
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	//put file cursor at beginning
	file.seekg(0);

	//load the entire file into the buffer
	file.read((char*)buffer.data(), fileSize);

	//now that the file is loaded into the buffer, we can close it
	file.close();

	//create a new shader module, using the buffer we loaded
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pNext = nullptr;

	//codeSize has to be in bytes, so multiply the ints in the buffer by size of int to know the real size of the buffer
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	//check that the creation goes well.
	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		return false;
	}
	*outShaderModule = shaderModule;
	return true;
}