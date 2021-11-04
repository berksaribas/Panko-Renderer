#pragma once

#include <vk_types.h>
#include <vector>

enum ComputeBufferType {
	UNIFORM,
	STORAGE
};

struct ComputeInstance {
	// PIPELINE
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;

	// BUFFERS
	VkDescriptorSet descriptorSet;
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	std::vector<size_t> sizes;
	std::vector<VmaMemoryUsage> memoryUsages;
	std::vector<ComputeBufferType> bufferTypes;
	std::vector<AllocatedBuffer> buffers;
};

class VulkanCompute {
public:
	void init(VkDevice device, VmaAllocator allocator, VkQueue computeQueue, uint32_t computeQueueFamily);
	void add_buffer(ComputeInstance& computeInstance, ComputeBufferType bufferType, VmaMemoryUsage memoryUsage, size_t size);
	void build(ComputeInstance& computeInstance, VkDescriptorPool descriptorPool, const char* computeShader);
	void compute(ComputeInstance computeInstance, int x, int y, int z);
	void destroy_compute_instance(ComputeInstance& computeInstance);
private:
	VkDevice _device;
	VkQueue _computeQueue;
	uint32_t _computeQueueFamily;
	CommandContext _computeContext;
	VmaAllocator _allocator;

	VkDescriptorType computeDescriptorTypes[2] = {
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
	//VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
	};

	VkBufferUsageFlagBits computeBufferUsageBits[2] = {
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	};
};