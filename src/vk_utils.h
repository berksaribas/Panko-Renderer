#pragma once

#include <vk_types.h>
#include "vk_engine.h"

#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			printf("Detected Vulkan error: %d\n", err); \
			abort();                                                \
		}                                                           \
	} while (0)

namespace vkutils {
	VkCommandBuffer create_command_buffer(VkDevice device, VkCommandPool commandPool, bool startRecording = false);
	void submit_and_free_command_buffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer cmd, VkQueue queue, VkFence fence);
	
	AllocatedBuffer create_buffer(VmaAllocator allocator, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags allocationFlags = 0);
	void cpu_to_gpu(VmaAllocator allocator, AllocatedBuffer& allocatedBuffer, void* data, size_t size);
	void cpu_to_gpu_staging(VmaAllocator allocator, VkCommandBuffer commandBuffer, AllocatedBuffer& allocatedBuffer, void* data, size_t size);

	bool load_image_from_memory(EngineData* engineData, void* pixels, int width, int height, AllocatedImage& outImage, uint32_t& outMipLevels);
	bool load_shader_module(VkDevice device, const char* filePath, VkShaderModule* outShaderModule);
	void cmd_viewport_scissor(VkCommandBuffer cmd, VkExtent2D extent);
	AllocatedBuffer create_upload_buffer(EngineData* engineData, void* buffer_data, size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void immediate_submit(EngineData* engineData, std::function<void(VkCommandBuffer cmd)>&& function);
}