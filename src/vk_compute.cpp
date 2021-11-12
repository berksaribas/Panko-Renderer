#include "vk_compute.h"
#include "vk_initializers.h"
#include <vk_shader.h>
#include <VkBootstrap.h>
#include <vk_engine.h>
#include <vk_utils.h>

// TODO: This only supports storage and uniform buffers at the moment.
// Potentially it will be able to write to textures as well

void VulkanCompute::init(VkDevice device, VmaAllocator allocator, VkQueue computeQueue, uint32_t computeQueueFamily)
{
	_device = device;
	_computeQueue = computeQueue;
	_computeQueueFamily = computeQueueFamily;
	_allocator = allocator;

	//create pool for compute context
	VkCommandPoolCreateInfo computeCommandPoolInfo = vkinit::command_pool_create_info(_computeQueueFamily);
	VK_CHECK(vkCreateCommandPool(_device, &computeCommandPoolInfo, nullptr, &_computeContext._commandPool));

	//create fence
	VkFenceCreateInfo computeFenceCreateInfo = vkinit::fence_create_info();
	VK_CHECK(vkCreateFence(_device, &computeFenceCreateInfo, nullptr, &_computeContext._fence));
}

void VulkanCompute::add_buffer(ComputeInstance& computeInstance, ComputeBufferType bufferType, VmaMemoryUsage memoryUsage, size_t size)
{
	computeInstance.bindings.push_back(
		vkinit::descriptorset_layout_binding(computeDescriptorTypes[bufferType],
			VK_SHADER_STAGE_COMPUTE_BIT, computeInstance.bindings.size())
	);

	computeInstance.sizes.push_back(size);
	computeInstance.memoryUsages.push_back(memoryUsage);
	computeInstance.bufferTypes.push_back(bufferType);
}

void VulkanCompute::build(ComputeInstance& computeInstance, VkDescriptorPool descriptorPool, const char* computeShader)
{
	VkShaderModule shader;
	if (!vkutil::load_shader_module(_device, computeShader, &shader))
	{
		assert("Compute Shader Loading Issue");
	}

	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(computeInstance.bindings.data(), computeInstance.bindings.size());
	VK_CHECK(vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &computeInstance.descriptorSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(descriptorPool, &computeInstance.descriptorSetLayout, 1);

	vkAllocateDescriptorSets(_device, &allocInfo, &computeInstance.descriptorSet);

	std::vector<VkWriteDescriptorSet> descriptorSets;

	for (int i = 0; i < computeInstance.bindings.size(); i++) {
		size_t size = computeInstance.sizes[i];
		VmaMemoryUsage memoryUsage = computeInstance.memoryUsages[i];
		VkBufferUsageFlagBits bufferUsage = computeBufferUsageBits[computeInstance.bufferTypes[i]];
		VkDescriptorType descriptorType = computeDescriptorTypes[computeInstance.bufferTypes[i]];
		computeInstance.buffers.push_back(
			vkutils::create_buffer(_allocator, size, bufferUsage, memoryUsage));

		VkWriteDescriptorSet descriptorSet = vkinit::write_descriptor_buffer(descriptorType, computeInstance.descriptorSet, &computeInstance.buffers[i]._descriptorBufferInfo, i);
		vkUpdateDescriptorSets(_device, 1, &descriptorSet, 0, nullptr);
	}


	VkDescriptorSetLayout setLayouts[] = { computeInstance.descriptorSetLayout };
	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 1);
	VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &computeInstance.pipelineLayout));

	VkPipelineShaderStageCreateInfo stage =
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, shader);

	VkComputePipelineCreateInfo computePipelineCreateInfo = {};
	computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCreateInfo.stage = stage;
	computePipelineCreateInfo.layout = computeInstance.pipelineLayout;

	VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &computeInstance.pipeline));

	vkDestroyShaderModule(_device, shader, nullptr);
}

void VulkanCompute::compute(ComputeInstance computeInstance, int x, int y, int z)
{
	VkCommandBuffer cmd = vkutils::create_command_buffer(_device, _computeContext._commandPool, true);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeInstance.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeInstance.pipelineLayout, 0, 1, &computeInstance.descriptorSet, 0, nullptr);
	vkCmdDispatch(cmd, x, y, z);

	vkutils::submit_and_free_command_buffer(_device, _computeContext._commandPool, cmd, _computeQueue, _computeContext._fence);
}

void VulkanCompute::destroy_compute_instance(ComputeInstance& computeInstance)
{
	vkDestroyPipeline(_device, computeInstance.pipeline, nullptr);
	vkDestroyPipelineLayout(_device, computeInstance.pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(_device, computeInstance.descriptorSetLayout, nullptr);

	for (int i = 0; i < computeInstance.buffers.size(); i++)
	{
		vmaDestroyBuffer(_allocator, computeInstance.buffers[i]._buffer, computeInstance.buffers[i]._allocation);
	}
}