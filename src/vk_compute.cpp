#include "vk_compute.h"
#include "vk_initializers.h"
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

void VulkanCompute::create_buffer(ComputeInstance& computeInstance, ComputeBufferType bufferType, VmaMemoryUsage memoryUsage, size_t size)
{
	//Creating texture is currently not supported
	assert(bufferType != TEXTURE_STORAGE && bufferType != TEXTURE_SAMPLED);

	ComputeBinding computeBinding = {};
	computeBinding.binding = vkinit::descriptorset_layout_binding(computeDescriptorTypes[bufferType],
		VK_SHADER_STAGE_COMPUTE_BIT, computeInstance.bindings.size());
	computeBinding.size = size;
	computeBinding.memoryUsage = memoryUsage;
	computeBinding.bufferType = bufferType;
	computeBinding.isExternal = false;

	computeInstance.bindings.push_back(computeBinding);
}

void VulkanCompute::add_buffer_binding(ComputeInstance& computeInstance, ComputeBufferType bufferType, AllocatedBuffer buffer)
{
	assert(bufferType != TEXTURE_STORAGE && bufferType != TEXTURE_SAMPLED);
	
	ComputeBinding computeBinding = {};
	computeBinding.binding = vkinit::descriptorset_layout_binding(computeDescriptorTypes[bufferType],
		VK_SHADER_STAGE_COMPUTE_BIT, computeInstance.bindings.size());
	computeBinding.bufferType = bufferType;
	computeBinding.isExternal = true;
	computeBinding.buffer = buffer;

	computeInstance.bindings.push_back(computeBinding);
}

void VulkanCompute::add_texture_binding(ComputeInstance& computeInstance, ComputeBufferType bufferType, VkSampler sampler, VkImageView imageView)
{
	assert(bufferType != UNIFORM && bufferType != STORAGE);

	ComputeBinding computeBinding = {};
	computeBinding.binding = vkinit::descriptorset_layout_binding(computeDescriptorTypes[bufferType],
		VK_SHADER_STAGE_COMPUTE_BIT, computeInstance.bindings.size());
	computeBinding.bufferType = bufferType;
	computeBinding.isExternal = true;

	VkDescriptorImageInfo imageBufferInfo = {};
	if (bufferType == TEXTURE_SAMPLED) {
		imageBufferInfo.sampler = sampler;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}
	else {
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}
	imageBufferInfo.imageView = imageView;

	computeBinding.imageInfo = imageBufferInfo;

	computeInstance.bindings.push_back(computeBinding);
}

void VulkanCompute::add_descriptor_set_layout(ComputeInstance& computeInstance, VkDescriptorSetLayout descriptorSetLayout)
{
	computeInstance.extraDescriptorSetLayouts.push_back(descriptorSetLayout);
}

void VulkanCompute::build(ComputeInstance& computeInstance, VkDescriptorPool descriptorPool, const char* computeShader)
{
	VkShaderModule shader;
	if (!vkutils::load_shader_module(_device, computeShader, &shader))
	{
		assert("Compute Shader Loading Issue");
	}

	std::vector<VkDescriptorSetLayoutBinding> bindingsData;
	for (int i = 0; i < computeInstance.bindings.size(); i++) {
		bindingsData.push_back(computeInstance.bindings[i].binding);
	}

	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindingsData.data(), computeInstance.bindings.size());
	VK_CHECK(vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &computeInstance.descriptorSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(descriptorPool, &computeInstance.descriptorSetLayout, 1);
	vkAllocateDescriptorSets(_device, &allocInfo, &computeInstance.descriptorSet);

	std::vector<VkWriteDescriptorSet> descriptorSets;

	for (int i = 0; i < computeInstance.bindings.size(); i++) {
		auto& binding = computeInstance.bindings[i];
		VkDescriptorType descriptorType = computeDescriptorTypes[binding.bufferType];

		if (!binding.isExternal) {
			VkBufferUsageFlagBits bufferUsage = computeBufferUsageBits[binding.bufferType];
			binding.buffer = vkutils::create_buffer(_allocator, binding.size, bufferUsage, binding.memoryUsage);
		}

		VkWriteDescriptorSet writeDescriptorSet;

		if (binding.bufferType != TEXTURE_STORAGE && binding.bufferType != TEXTURE_SAMPLED) {
			writeDescriptorSet = vkinit::write_descriptor_buffer(descriptorType, computeInstance.descriptorSet, &binding.buffer._descriptorBufferInfo, i);
		}
		else {
			writeDescriptorSet = vkinit::write_descriptor_image(descriptorType, computeInstance.descriptorSet, &binding.imageInfo, i, 1);
		}

		vkUpdateDescriptorSets(_device, 1, &writeDescriptorSet, 0, nullptr);
	}

	std::vector<VkDescriptorSetLayout> setLayouts;
	setLayouts.push_back(computeInstance.descriptorSetLayout);

	for (int i = 0; i < computeInstance.extraDescriptorSetLayouts.size(); i++) {
		setLayouts.push_back(computeInstance.extraDescriptorSetLayouts[i]);
	}

	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts.data(), setLayouts.size());
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

void VulkanCompute::compute(ComputeInstance& computeInstance, int x, int y, int z)
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

	for (int i = 0; i < computeInstance.bindings.size(); i++)
	{
		if (!computeInstance.bindings[i].isExternal) {
			vmaDestroyBuffer(_allocator, computeInstance.bindings[i].buffer._buffer, computeInstance.bindings[i].buffer._allocation);
		}
	}
}