#include "vk_compute.h"
#include "vk_initializers.h"
#include <vk_shader.h>
#include <VkBootstrap.h>
#include <vk_engine.h>

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

	VkDescriptorSetLayoutCreateInfo setinfo = {};
	setinfo.flags = 0;
	setinfo.pNext = nullptr;
	setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setinfo.pBindings = computeInstance.bindings.data();
	setinfo.bindingCount = computeInstance.bindings.size();
	VK_CHECK(vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &computeInstance.descriptorSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.pNext = nullptr;
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &computeInstance.descriptorSetLayout;
	vkAllocateDescriptorSets(_device, &allocInfo, &computeInstance.descriptorSet);

	std::vector<VkWriteDescriptorSet> descriptorSets;

	for (int i = 0; i < computeInstance.bindings.size(); i++) {
		size_t size = computeInstance.sizes[i];
		VmaMemoryUsage memoryUsage = computeInstance.memoryUsages[i];
		VkBufferUsageFlagBits bufferUsage = computeBufferUsageBits[computeInstance.bufferTypes[i]];
		VkDescriptorType descriptorType = computeDescriptorTypes[computeInstance.bufferTypes[i]];
		computeInstance.buffers.push_back(
			vkinit::create_buffer(_allocator, size, bufferUsage, memoryUsage));

		VkDescriptorBufferInfo bufferInfo;
		bufferInfo.buffer = computeInstance.buffers[i]._buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = size;

		VkWriteDescriptorSet descriptorSet = vkinit::write_descriptor_buffer(descriptorType, computeInstance.descriptorSet, &bufferInfo, i);
		vkUpdateDescriptorSets(_device, 1, &descriptorSet, 0, nullptr);
	}


	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
	VkDescriptorSetLayout setLayouts[] = { computeInstance.descriptorSetLayout };
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts = setLayouts;
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
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_computeContext._commandPool, 1);

	VkCommandBuffer cmd;
	VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &cmd));

	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeInstance.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeInstance.pipelineLayout, 0, 1, &computeInstance.descriptorSet, 0, nullptr);
	vkCmdDispatch(cmd, x, y, z);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VK_CHECK(vkQueueSubmit(_computeQueue, 1, &submit, _computeContext._fence));

	vkWaitForFences(_device, 1, &_computeContext._fence, true, 9999999999);
	vkResetFences(_device, 1, &_computeContext._fence);

	vkResetCommandPool(_device, _computeContext._commandPool, 0);
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