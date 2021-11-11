#include "vk_utils.h"
#include <vk_initializers.h>

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
