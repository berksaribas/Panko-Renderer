#include <vk_rendergraph.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include "vk_pipeline.h"
#include "gltf_scene.hpp"

using namespace Vrg;

inline static VkAccessFlags get_access_flags(ResourceAccessType type) {
	switch (type) {
	case ResourceAccessType::COLOR_WRITE:
		return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	case ResourceAccessType::DEPTH_WRITE:
		return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	case ResourceAccessType::COMPUTE_WRITE:
		return VK_ACCESS_SHADER_WRITE_BIT;
	case ResourceAccessType::COMPUTE_READ:
		return VK_ACCESS_SHADER_READ_BIT;
	case ResourceAccessType::FRAGMENT_READ:
		return VK_ACCESS_SHADER_READ_BIT;
	case ResourceAccessType::RAYTRACING_WRITE:
		return VK_ACCESS_SHADER_WRITE_BIT;
	case ResourceAccessType::RAYTRACING_READ:
		return VK_ACCESS_SHADER_READ_BIT;
	}

	return 0;
}

inline static VkPipelineStageFlags get_stage_flags(ResourceAccessType type) {
	switch (type) {
	case ResourceAccessType::COLOR_WRITE:
		return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	case ResourceAccessType::DEPTH_WRITE:
		return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	case ResourceAccessType::COMPUTE_WRITE:
		return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	case ResourceAccessType::COMPUTE_READ:
		return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	case ResourceAccessType::FRAGMENT_READ:
		return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	case ResourceAccessType::RAYTRACING_WRITE:
		return VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
	case ResourceAccessType::RAYTRACING_READ:
		return VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
	}
	return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
}

inline static VkImageLayout get_image_layout(ResourceAccessType type) {
	switch (type) {
	case ResourceAccessType::COLOR_WRITE:
		return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	case ResourceAccessType::DEPTH_WRITE:
		return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
	case ResourceAccessType::COMPUTE_WRITE:
		return VK_IMAGE_LAYOUT_GENERAL;
	case ResourceAccessType::COMPUTE_READ:
		return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	case ResourceAccessType::FRAGMENT_READ:
		return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	case ResourceAccessType::RAYTRACING_WRITE:
		return VK_IMAGE_LAYOUT_GENERAL;
	case ResourceAccessType::RAYTRACING_READ:
		return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}
	return VK_IMAGE_LAYOUT_UNDEFINED;
}

inline static bool is_read_access(ResourceAccessType type) {
	return type == ResourceAccessType::COMPUTE_READ || type == ResourceAccessType::FRAGMENT_READ || type == ResourceAccessType::RAYTRACING_READ;
}

inline static bool is_buffer_binding(BindType type) {
	return type == BindType::STORAGE || type == BindType::UNIFORM;
}

inline static bool is_image_binding(BindType type) {
	return type == BindType::IMAGE_VIEW;
}

template <typename T>
inline static void copy_memory(Slice<T>& slice, FrameAllocator& frameAllocator) {
	T* newPtr = frameAllocator.allocate_array<T>(slice.size());
	memcpy(newPtr, slice.m_data, slice.byte_size());
	slice.m_data = newPtr;
}

RenderGraph::RenderGraph(EngineData* _engineData)
{
	renderPasses.reserve(128);
	bindings.reserve(256);
	engineData = _engineData;

	frameAllocator.initialize(1024 * 1024 * 64); //64mb


	//Create a linear sampler
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 1.0f;
	VK_CHECK(vkCreateSampler(_engineData->device, &samplerInfo, nullptr, &samplers[(int)Sampler::LINEAR]));

	//Create a nearest sampler
	VkSamplerCreateInfo samplerInfo2 = vkinit::sampler_create_info(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	samplerInfo2.mipLodBias = 0.0f;
	samplerInfo2.maxAnisotropy = 1.0f;
	samplerInfo2.minLod = 0.0f;
	samplerInfo2.maxLod = 1.0f;
	VK_CHECK(vkCreateSampler(_engineData->device, &samplerInfo2, nullptr, &samplers[(int)Sampler::NEAREST]));
}

void Vrg::RenderGraph::enable_raytracing(VulkanRaytracing* _vulkanRaytracing)
{
	vulkanRaytracing = _vulkanRaytracing;
}

void RenderGraph::add_render_pass(RenderPass renderPass)
{
	auto& newPass = renderPass;

	copy_memory(newPass.writes, frameAllocator);
	copy_memory(newPass.reads, frameAllocator);
	copy_memory(newPass.extraDescriptorSets, frameAllocator);

	copy_memory(newPass.constants, frameAllocator);

	for (int i = 0; i < newPass.constants.size(); i++) {
		void* newPtr = frameAllocator.allocate(newPass.constants[i].size, 8);
		memcpy(newPtr, newPass.constants[i].data, newPass.constants[i].size);
		newPass.constants[i].data = newPtr;
	}

	if (newPass.pipelineType == PipelineType::RASTER_TYPE) {
		copy_memory(newPass.rasterPipeline.blendAttachmentStates, frameAllocator);
		copy_memory(newPass.rasterPipeline.vertexBuffers, frameAllocator);
		copy_memory(newPass.rasterPipeline.colorOutputs, frameAllocator);
	}

	for (int i = 0; i < newPass.writes.size(); i++) {
		newPass.descriptorSetCount = newPass.descriptorSetCount > (newPass.writes[i].set_index + 1) ? newPass.descriptorSetCount : (newPass.writes[i].set_index + 1);
	}

	for (int i = 0; i < newPass.reads.size(); i++) {
		newPass.descriptorSetCount = newPass.descriptorSetCount > (newPass.reads[i].set_index + 1) ? newPass.descriptorSetCount : (newPass.reads[i].set_index + 1);
	}

	for (int i = 0; i < newPass.extraDescriptorSets.size(); i++) {
		newPass.descriptorSetCount = newPass.descriptorSetCount > (newPass.extraDescriptorSets[i].set_index + 1) ? newPass.descriptorSetCount : (newPass.extraDescriptorSets[i].set_index + 1);
	}

	renderPasses.push_back(renderPass);
}

Bindable* RenderGraph::register_image_view(AllocatedImage* image, ImageView imageView, std::string resourceName)
{
	uint32_t id = bindings.size();

	Bindable bindable = {
		.image = image,
		.imageView = imageView,
		.format = image->format,
		.type = BindType::IMAGE_VIEW,
		.id = id
	};

	bindings.push_back(bindable);

	return &bindings[bindings.size() - 1];
}

Bindable* RenderGraph::register_storage_buffer(AllocatedBuffer* buffer, std::string resourceName)
{
	uint32_t id = bindings.size();

	Bindable bindable = {
		.buffer = buffer,
		.type = BindType::STORAGE,
		.id = id
	};

	bindings.push_back(bindable);

	return &bindings[bindings.size() - 1];
}

Bindable* RenderGraph::register_uniform_buffer(AllocatedBuffer* buffer, std::string resourceName)
{
	uint32_t id = bindings.size();

	Bindable bindable = {
		.buffer = buffer,
		.type = BindType::UNIFORM,
		.id = id
	};

	bindings.push_back(bindable);

	return &bindings[bindings.size() - 1];
}

Bindable* RenderGraph::register_vertex_buffer(AllocatedBuffer* buffer, VkFormat format, std::string resourceName)
{
	uint32_t id = bindings.size();

	Bindable bindable = {
		.buffer = buffer,
		.format = format,
		.type = BindType::VERTEX,
		.id = id
	};

	bindings.push_back(bindable);

	return &bindings[bindings.size() - 1];
}

Bindable* RenderGraph::register_index_buffer(AllocatedBuffer* buffer, VkFormat format, std::string resourceName)
{
	uint32_t id = bindings.size();

	Bindable bindable = {
		.buffer = buffer,
		.format = format,
		.type = BindType::INDEX,
		.id = id
	};

	bindings.push_back(bindable);

	return &bindings[bindings.size() - 1];
}


void RenderGraph::insert_barrier(VkCommandBuffer cmd, Vrg::Bindable* binding, PipelineType pipelineType, bool isWrite, uint32_t mip) {
	ResourceAccessType prevAccessType = ResourceAccessType::NONE;

	if (is_buffer_binding(binding->type)) {
		if (bufferBindingAccessType.find(binding->buffer->_buffer) != bufferBindingAccessType.end()) {
			prevAccessType = bufferBindingAccessType[binding->buffer->_buffer];
		}
	}
	if (is_image_binding(binding->type)) {
		if (imageBindingAccessType.find({ binding->image->_image, mip }) != imageBindingAccessType.end()) {
			prevAccessType = imageBindingAccessType[{ binding->image->_image, mip }];
		}
	}

	VkAccessFlags dstAccess = get_access_flags(ResourceAccessType::NONE);
	VkPipelineStageFlags dstStage = get_stage_flags(ResourceAccessType::NONE);
	VkImageLayout dstLayout = get_image_layout(ResourceAccessType::NONE);

	ResourceAccessType newAccessType;

	if (isWrite) {
		if (pipelineType == PipelineType::COMPUTE_TYPE) {
			newAccessType = ResourceAccessType::COMPUTE_WRITE;
		}
		else if (pipelineType == PipelineType::RASTER_TYPE) {
			//undefined behaviour
		}
		else if (pipelineType == PipelineType::RAYTRACING_TYPE) {
			newAccessType = ResourceAccessType::RAYTRACING_WRITE;
		}
	}
	else {
		if (pipelineType == PipelineType::COMPUTE_TYPE) {
			newAccessType = ResourceAccessType::COMPUTE_READ;
		}
		else if (pipelineType == PipelineType::RASTER_TYPE) {
			newAccessType = ResourceAccessType::FRAGMENT_READ;
		}
		else if (pipelineType == PipelineType::RAYTRACING_TYPE) {
			newAccessType = ResourceAccessType::RAYTRACING_READ;
		}
	}

	dstAccess = get_access_flags(newAccessType);
	dstStage = get_stage_flags(newAccessType);
	dstLayout = get_image_layout(newAccessType);

	bool isImageLayoutDifferent = false;
	VkImageLayout srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (binding->type == BindType::IMAGE_VIEW) {
		srcLayout = get_current_image_layout(binding->image->_image, mip);
		if (srcLayout != dstLayout) {
			isImageLayoutDifferent = true;
		}
	}

	bool condition = prevAccessType != ResourceAccessType::NONE || isImageLayoutDifferent;
	if (!isWrite) {
		condition = (prevAccessType != ResourceAccessType::NONE && !is_read_access(prevAccessType)) || isImageLayoutDifferent;
	}

	if (condition)
	{
		VkAccessFlags srcAccess = get_access_flags(prevAccessType);
		VkPipelineStageFlags srcStage = get_stage_flags(prevAccessType);

		if (binding->type == BindType::IMAGE_VIEW) {
			//TODO BARRIER VKIMAGESUBRESOURCERANGE FIX
			VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;

			if (binding->image->format == VK_FORMAT_D32_SFLOAT) {
				aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
			}

			vkutils::image_barrier(cmd, binding->image->_image, srcLayout, dstLayout, { aspectFlag, mip, 1, 0, 1 }, srcAccess, dstAccess, srcStage, dstStage);
		}
		else {
			vkutils::memory_barrier(cmd, binding->buffer->_buffer, srcAccess, dstAccess, srcStage, dstStage);
		}
	}

	if (is_buffer_binding(binding->type)) {
		bufferBindingAccessType[binding->buffer->_buffer] = newAccessType;
	}
	if (is_image_binding(binding->type)) {
		imageBindingAccessType[{binding->image->_image, mip}] = newAccessType;
		bindingImageLayout[{binding->image->_image, mip}] = dstLayout;
	}
}

void RenderGraph::execute(VkCommandBuffer cmd)
{
	for (int r = 0; r < renderPasses.size(); r++) {
		auto& renderPass = renderPasses[r];

		//WRITE BARRIERS
		for (int i = 0; i < renderPass.writes.size(); i++) {
			auto binding = renderPass.writes[i].binding;
			
			if (is_buffer_binding(binding->type)) {
				insert_barrier(cmd, binding, renderPass.pipelineType, true);
			}
			else if (is_image_binding(binding->type)) {
				for (uint32_t k = binding->imageView.baseMipLevel; k < binding->imageView.baseMipLevel + binding->imageView.mipLevelCount; k++) {
					insert_barrier(cmd, binding, renderPass.pipelineType, true, k);
				}
			}
		}
		//READ BARRIERS
		for (int i = 0; i < renderPass.reads.size(); i++) {
			auto binding = renderPass.reads[i].binding;

			if (is_buffer_binding(binding->type)) {
				insert_barrier(cmd, binding, renderPass.pipelineType, false);
			}
			else if (is_image_binding(binding->type)) {
				for (uint32_t k = binding->imageView.baseMipLevel; k < binding->imageView.baseMipLevel + binding->imageView.mipLevelCount; k++) {
					insert_barrier(cmd, binding, renderPass.pipelineType, false, k);
				}
			}
		}


		if (renderPass.pipelineType == PipelineType::COMPUTE_TYPE) {
			VkDescriptorSet descriptorSet[16];

			for (int i = 0; i < renderPass.descriptorSetCount; i++) {
				descriptorSet[i] = get_descriptor_set(renderPass, i);
			}
			auto pipelineLayout = get_pipeline_layout(renderPass);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get_pipeline(renderPass));
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, renderPass.descriptorSetCount, descriptorSet, 0, nullptr);
			if (renderPass.constants.size() > 0) {
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_ALL, 0,
					renderPass.constants[0].size, renderPass.constants[0].data);
			}
			vkCmdDispatch(cmd, renderPass.computePipeline.dimX, renderPass.computePipeline.dimY, renderPass.computePipeline.dimZ);
		}
		else if (renderPass.pipelineType == PipelineType::RASTER_TYPE) {
			const auto& rasterPipeline = renderPass.rasterPipeline;

			//TODO: Get rid of std::vectors
			std::vector<VkRenderingAttachmentInfoKHR> renderingAttachments;
			renderingAttachments.reserve(rasterPipeline.colorOutputs.size());
			int colorAttachmentCount = rasterPipeline.colorOutputs.size();

			for (int i = 0; i < colorAttachmentCount; i++) {
				VkRenderingAttachmentInfoKHR color_attachment_info{
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
					.imageView = get_image_view(rasterPipeline.colorOutputs[i].binding->image->_image, rasterPipeline.colorOutputs[i].binding->imageView, rasterPipeline.colorOutputs[i].binding->format),
					.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = rasterPipeline.colorOutputs[i].clearValue
				};
				renderingAttachments.push_back(color_attachment_info);
			}

			VkRenderingInfoKHR render_info {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
				.renderArea = {0, 0, rasterPipeline.size.width, rasterPipeline.size.height},
				.layerCount = 1,
				.colorAttachmentCount = (uint32_t) colorAttachmentCount,
				.pColorAttachments = renderingAttachments.data(),
			};

			VkRenderingAttachmentInfoKHR depthStencilAttachment{
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
				.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = rasterPipeline.depthOutput.clearValue
			};


			if (rasterPipeline.depthOutput.binding != nullptr) {
				depthStencilAttachment.imageView = get_image_view(rasterPipeline.depthOutput.binding->image->_image, rasterPipeline.depthOutput.binding->imageView, rasterPipeline.depthOutput.binding->format, true),
				render_info.pDepthAttachment = &depthStencilAttachment;
			}

			//TODO BARRIER VKIMAGESUBRESOURCERANGE FIX

			//Color output barrier
			for (int i = 0; i < colorAttachmentCount; i++) {
				auto& colorAttachment = rasterPipeline.colorOutputs[0];
				uint32_t mipLevel = colorAttachment.binding->imageView.baseMipLevel;
				vkutils::image_barrier(cmd, colorAttachment.binding->image->_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 1, 0, 1 }, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

				imageBindingAccessType[{colorAttachment.binding->image->_image, mipLevel}] = ResourceAccessType::COLOR_WRITE;
				bindingImageLayout[{colorAttachment.binding->image->_image, mipLevel}] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			}
			//Depth output barrier
			if (rasterPipeline.depthOutput.binding != nullptr) {
				uint32_t mipLevel = rasterPipeline.depthOutput.binding->imageView.baseMipLevel;

				vkutils::image_barrier(cmd, rasterPipeline.depthOutput.binding->image->_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, mipLevel, 1, 0, 1}, 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

				imageBindingAccessType[{rasterPipeline.depthOutput.binding->image->_image, mipLevel}] = ResourceAccessType::DEPTH_WRITE;
				bindingImageLayout[{rasterPipeline.depthOutput.binding->image->_image, mipLevel}] = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			}

			vkCmdBeginRenderingKHR(cmd, &render_info);

			vkutils::cmd_viewport_scissor(cmd, rasterPipeline.size);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, get_pipeline(renderPass));

			//TODO: Currently supporting maximum of 16 descriptor sets
			VkDescriptorSet descriptorSet[16];
			 
			for (int i = 0; i < renderPass.descriptorSetCount; i++) {
				descriptorSet[i] = get_descriptor_set(renderPass, i);
			}
			auto pipelineLayout = get_pipeline_layout(renderPass);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, renderPass.descriptorSetCount, descriptorSet, 0, nullptr);

			if (renderPass.constants.size() > 0) {
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_ALL, 0,
					renderPass.constants[0].size, renderPass.constants[0].data);
			}

			//TODO: Get rid of std::vectors
			std::vector<VkDeviceSize> offsets;
			std::vector<VkBuffer> buffers;
			offsets.reserve(rasterPipeline.vertexBuffers.size());
			buffers.reserve(rasterPipeline.vertexBuffers.size());

			for (int i = 0; i < rasterPipeline.vertexBuffers.size(); i++) {
				buffers.push_back(rasterPipeline.vertexBuffers[i]->buffer->_buffer);
				offsets.push_back(0);
			}

			if (buffers.size() > 0) {
				vkCmdBindVertexBuffers(cmd, 0, buffers.size(), buffers.data(), offsets.data());
			}

			if(rasterPipeline.indexBuffer != nullptr) {
				vkCmdBindIndexBuffer(cmd, rasterPipeline.indexBuffer->buffer->_buffer, 0, VK_INDEX_TYPE_UINT32); //TODO: support different index types
			}
			//TODO: DRAW
			renderPass.execute(cmd);

			vkCmdEndRenderingKHR(cmd);

			for (int i = 0; i < colorAttachmentCount; i++) {
				auto& colorAttachment = rasterPipeline.colorOutputs[0];
				if (colorAttachment.isSwapChain) {
					vkutils::image_barrier(cmd, colorAttachment.binding->image->_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
				}
			}
		}
		else if (renderPass.pipelineType == PipelineType::RAYTRACING_TYPE) {
			VkDescriptorSet descriptorSet[16];

			for (int i = 0; i < renderPass.descriptorSetCount; i++) {
				descriptorSet[i] = get_descriptor_set(renderPass, i);
			}

			auto raytracingPipeline = get_raytracing_pipeline(renderPass);
			auto pipelineLayout = get_pipeline_layout(renderPass);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, raytracingPipeline->pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, renderPass.descriptorSetCount, descriptorSet, 0, nullptr);
			
			if (renderPass.constants.size() > 0) {
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_ALL, 0,
					renderPass.constants[0].size, renderPass.constants[0].data);
			}

			vkCmdTraceRaysKHR(cmd, &raytracingPipeline->rgenRegion, &raytracingPipeline->missRegion, &raytracingPipeline->hitRegion, &raytracingPipeline->callRegion, renderPass.raytracingPipeline.width, renderPass.raytracingPipeline.height, renderPass.raytracingPipeline.depth);
		}
	}

	imageBindingAccessType.clear();
	bufferBindingAccessType.clear();
	renderPasses.clear();
	frameAllocator.reset();
}

void RenderGraph::compile()
{
	//for (int r = 0; r < renderPasses.size(); r++) {
	//	auto& renderPass = renderPasses[r];
	//	if (renderPass.isDescriptorsDirty) {
	//		renderPass.create_descriptor_set();
	//		renderPass.create_pipeline_layout();
	//
	//		renderPass.isPipelineDirty = true;
	//	}
	//
	//	if (renderPass.isPipelineDirty) {
	//		renderPass.create_pipeline();
	//	}
	//}
}

/*
RenderPass* RenderPass::add_constant(void* data, size_t size)
{
	uint32_t offset = 0;
	if (pushConstantRanges.size() > 0) {
		offset = pushConstantRanges[pushConstantRanges.size() - 1].offset
			+ pushConstantRanges[pushConstantRanges.size() - 1].size;
	}

	VkPushConstantRange range = { VK_SHADER_STAGE_ALL, offset, size };
	pushConstantRanges.push_back(range);

	return this;
}
*/

VkPipeline RenderGraph::get_pipeline(RenderPass& renderPass)
{
	//TODO: A better pipeline cache!
	if (pipelineCache.find(renderPass.name) != pipelineCache.end()) {
		return pipelineCache[renderPass.name];
	}

	if (renderPass.pipelineType == PipelineType::COMPUTE_TYPE) {
		VkShaderModule computeShader;
		if (!vkutils::load_shader_module(engineData->device, renderPass.computePipeline.shader.c_str(), &computeShader))
		{
			assert("Compute Shader Loading Issue");
		}

		VkPipelineShaderStageCreateInfo stage =
			vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShader);

		VkComputePipelineCreateInfo computePipelineCreateInfo = {};
		computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computePipelineCreateInfo.stage = stage;
		computePipelineCreateInfo.layout = get_pipeline_layout(renderPass);

		VkPipeline computePipeline;

		VK_CHECK(vkCreateComputePipelines(engineData->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &computePipeline));
		pipelineCache[renderPass.name] = computePipeline;

		vkDestroyShaderModule(engineData->device, computeShader, nullptr);

		return computePipeline;
	}
	else if (renderPass.pipelineType == PipelineType::RASTER_TYPE) {
		//dynamic rendering
		std::vector<VkFormat> colorAttachments;
		colorAttachments.reserve(renderPass.rasterPipeline.colorOutputs.size());

		for (int i = 0; i < renderPass.rasterPipeline.colorOutputs.size(); i++) {
			colorAttachments.push_back(renderPass.rasterPipeline.colorOutputs[i].binding->format);
		}

		VkFormat depthFormat = VK_FORMAT_UNDEFINED;

		if (renderPass.rasterPipeline.depthOutput.binding != nullptr) {
			depthFormat = renderPass.rasterPipeline.depthOutput.binding->format;
		}

		VkPipelineRenderingCreateInfoKHR pipeline_rendering_create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = (uint32_t)renderPass.rasterPipeline.colorOutputs.size(),
			.pColorAttachmentFormats = colorAttachments.data(),
			.depthAttachmentFormat = depthFormat
		};

		//compile shaders
		VkShaderModule vertexShader;
		if (!vkutils::load_shader_module(engineData->device, renderPass.rasterPipeline.vertexShader.c_str(), &vertexShader))
		{
			assert("Vertex Shader Loading Issue");
		}

		VkShaderModule fragmentShader;
		if (!vkutils::load_shader_module(engineData->device, renderPass.rasterPipeline.fragmentShader.c_str(), &fragmentShader))
		{
			assert("Fragment Shader Loading Issue");
		}

		PipelineBuilder pipelineBuilder;

		//VERTEX INFO
		std::vector<VkVertexInputBindingDescription> bindings;
		std::vector<VkVertexInputAttributeDescription> attributes;
		bindings.reserve(renderPass.rasterPipeline.vertexBuffers.size());
		attributes.reserve(renderPass.rasterPipeline.vertexBuffers.size());
		{
			pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

			for (int i = 0; i < renderPass.rasterPipeline.vertexBuffers.size(); i++) {
				VkFormat format = renderPass.rasterPipeline.vertexBuffers[i]->format;

				VkVertexInputBindingDescription vertexBinding = {};
				vertexBinding.binding = i;
				if (format == VK_FORMAT_R32G32_SFLOAT) {
					vertexBinding.stride = sizeof(float) * 2;
				}
				else if (format == VK_FORMAT_R32G32B32_SFLOAT) {
					vertexBinding.stride = sizeof(float) * 3;
				}
				else if (format == VK_FORMAT_R32G32B32A32_SFLOAT) {
					vertexBinding.stride = sizeof(float) * 4;
				}
				vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
				bindings.push_back(vertexBinding);

				VkVertexInputAttributeDescription vertexAttribute = {};
				vertexAttribute.binding = i;
				vertexAttribute.location = i;
				vertexAttribute.format = format;
				vertexAttribute.offset = 0;

				attributes.push_back(vertexAttribute);
			}

			pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = attributes.data();
			pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();
			pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = bindings.data();
			pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = bindings.size();
		}

		//TOPOLOGY
		{
			VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
			if (renderPass.rasterPipeline.inputAssembly == InputAssembly::TRIANGLE) {
				topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			}
			else if (renderPass.rasterPipeline.inputAssembly == InputAssembly::POINT) {
				topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
			}
			else if (renderPass.rasterPipeline.inputAssembly == InputAssembly::LINE) {
				topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
			}
			pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(topology);
		}

		//RASTERIZER
		{
			VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
			if (renderPass.rasterPipeline.polygonMode == PolygonMode::FILL) {
				polygonMode = VK_POLYGON_MODE_FILL;
			}
			else if (renderPass.rasterPipeline.polygonMode == PolygonMode::POINT) {
				polygonMode = VK_POLYGON_MODE_POINT;
			}
			else if (renderPass.rasterPipeline.polygonMode == PolygonMode::LINE) {
				polygonMode = VK_POLYGON_MODE_LINE;
			}
			pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(polygonMode);

			if (renderPass.rasterPipeline.cullMode == CullMode::NONE) {
				pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;
			}
			else {
				pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
			}
			
			if (renderPass.rasterPipeline.cullMode == CullMode::CLOCK_WISE) {
				pipelineBuilder._rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
			}
			else {
				pipelineBuilder._rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			}
		}

		//CONSERVATIVE RASTERIZATION
		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterStateCI{};
		conservativeRasterStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
		conservativeRasterStateCI.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
		conservativeRasterStateCI.extraPrimitiveOverestimationSize = 1.0;

		if (renderPass.rasterPipeline.enableConservativeRasterization) {
			pipelineBuilder._rasterizer.pNext = &conservativeRasterStateCI;
		}

		//MULTISAMPLING
		{
			pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();
		}

		//COLOR BLENDING
		{
			pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(renderPass.rasterPipeline.blendAttachmentStates.size(), renderPass.rasterPipeline.blendAttachmentStates.m_data);
		}

		//DEPTH TEST
		{
			pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(renderPass.rasterPipeline.depthState.depthTest, renderPass.rasterPipeline.depthState.depthWrite, renderPass.rasterPipeline.depthState.compareOp);
		}

		//SHADER STAGES
		{
			pipelineBuilder._shaderStages.push_back(
				vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader));
			pipelineBuilder._shaderStages.push_back(
				vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));
		}
		

		pipelineBuilder._pipelineLayout = get_pipeline_layout(renderPass);
		VkPipeline newPipeline = pipelineBuilder.build_pipeline(engineData->device, nullptr, &pipeline_rendering_create_info);
		pipelineCache[renderPass.name] = newPipeline;

		vkDestroyShaderModule(engineData->device, vertexShader, nullptr);
		vkDestroyShaderModule(engineData->device, fragmentShader, nullptr);

		vkutils::setObjectName(engineData->device, newPipeline, renderPass.name + " - Pipeline");

		return newPipeline;
	}
}

RaytracingPipeline* Vrg::RenderGraph::get_raytracing_pipeline(RenderPass& renderPass)
{
	if (raytracingPipelineCache.find(renderPass.name) != raytracingPipelineCache.end()) {
		return &raytracingPipelineCache[renderPass.name];
	}
	RaytracingPipeline pipeline;
	RayPipeline rayPipeline = renderPass.raytracingPipeline;

	vulkanRaytracing->create_new_pipeline(pipeline, get_pipeline_layout(renderPass), rayPipeline.rgenShader.c_str(), rayPipeline.missShader.c_str(), rayPipeline.hitShader.c_str(), rayPipeline.recursionDepth, &rayPipeline.rgenSpecialization, &rayPipeline.missSpecialization, &rayPipeline.hitSpecialization);

	raytracingPipelineCache[renderPass.name] = pipeline;
	return &raytracingPipelineCache[renderPass.name];
}

VkPipelineLayout RenderGraph::get_pipeline_layout(RenderPass& renderPass)
{
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.resize(renderPass.descriptorSetCount);

	for (int i = 0; i < renderPass.descriptorSetCount; i++) {
		descriptorSetLayouts[i] = get_descriptor_set_layout(renderPass, i);
	}

	std::vector<VkPushConstantRange> pushRanges;
	for (int i = 0; i < renderPass.constants.size(); i++) {
		VkPushConstantRange range = {
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = (uint32_t) renderPass.constants[i].size
		};
		pushRanges.push_back(range);
	}
	
	PipelineLayoutCache cache1 = { descriptorSetLayouts, pushRanges };

	if (pipelineLayoutCache.find(cache1) != pipelineLayoutCache.end()) {
		return pipelineLayoutCache[cache1];
	}

	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(descriptorSetLayouts.data(), descriptorSetLayouts.size());

	if (pushRanges.size() > 0) {
		pipeline_layout_info.pushConstantRangeCount = pushRanges.size();
		pipeline_layout_info.pPushConstantRanges = pushRanges.data();
	}

	VkPipelineLayout newLayout;
	VK_CHECK(vkCreatePipelineLayout(engineData->device, &pipeline_layout_info, nullptr, &newLayout));
	pipelineLayoutCache[cache1] = newLayout;

	vkutils::setObjectName(engineData->device, newLayout, renderPass.name + " - Pipeline Layout");

	return newLayout;
}

VkDescriptorSet RenderGraph::get_descriptor_set(RenderPass& renderPass, int set)
{
	for (int i = 0; i < renderPass.extraDescriptorSets.size(); i++) {
		if (renderPass.extraDescriptorSets[i].set_index == set) {
			return renderPass.extraDescriptorSets[i].descriptorSet;
		}
	}

	std::vector<DescriptorSet> descriptorSets;

	auto prepare = [&](const Slice<DescriptorBinding>& bindings, bool isWrite) {
		for (int i = 0; i < bindings.size(); i++) {
			if (bindings[i].set_index != set) {
				continue;
			}

			DescriptorSet descriptorSet = {};
			switch (bindings[i].binding->type) {
			case BindType::UNIFORM:
				descriptorSet.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorSet.buffer = bindings[i].binding->buffer;
				break;
			case BindType::STORAGE:
				descriptorSet.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorSet.buffer = bindings[i].binding->buffer;
				break;
			case BindType::IMAGE_VIEW:
				descriptorSet.type = isWrite ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorSet.image = bindings[i].binding->image->_image;
				descriptorSet.imageView = bindings[i].binding->imageView;
				descriptorSet.imageLayout = isWrite ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				descriptorSet.format = bindings[i].binding->format;
				break;
			default:
				break;
			}

			descriptorSets.push_back(descriptorSet);
		}
	};

	prepare(renderPass.reads, false);
	prepare(renderPass.writes, true);

	DescriptorSetCache cache1 = { descriptorSets };

	//printf("Render pass: %s, set: %d, descriptor count: %d\n", renderPass.name.c_str(), set, descriptorSets.size());
	if (descriptorSetCache.find(cache1) != descriptorSetCache.end()) {
		return descriptorSetCache[cache1];
	}

	VkDescriptorSetLayout descriptorSetLayout = get_descriptor_set_layout(renderPass, set);
	VkDescriptorSet descriptorSet;

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData->descriptorPool, &descriptorSetLayout, 1);
	vkAllocateDescriptorSets(engineData->device, &allocInfo, &descriptorSet);

	descriptorSetCache[cache1] = descriptorSet;

	// TODO: Update allocations
	for (int i = 0; i < descriptorSets.size(); i++) {
		VkWriteDescriptorSet writeDescriptorSet;
		VkDescriptorImageInfo imageInfo;

		if (descriptorSets[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || descriptorSets[i].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
			writeDescriptorSet = vkinit::write_descriptor_buffer(descriptorSets[i].type, descriptorSet, &descriptorSets[i].buffer->_descriptorBufferInfo, i);
		}
		else if(descriptorSets[i].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
			imageInfo = {
				.sampler = samplers[(int)descriptorSets[i].imageView.sampler],
				.imageView = get_image_view(descriptorSets[i].image, descriptorSets[i].imageView, descriptorSets[i].format),
				.imageLayout = descriptorSets[i].imageLayout
			};
			writeDescriptorSet = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorSet, &imageInfo, i, 1);
		}
		else if (descriptorSets[i].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
			imageInfo = {
				.sampler = samplers[(int)descriptorSets[i].imageView.sampler],
				.imageView = get_image_view(descriptorSets[i].image, descriptorSets[i].imageView, descriptorSets[i].format),
				.imageLayout = descriptorSets[i].imageLayout
			};
			writeDescriptorSet = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorSet, &imageInfo, i, 1);
		}
		vkUpdateDescriptorSets(engineData->device, 1, &writeDescriptorSet, 0, nullptr);
	}

	vkutils::setObjectName(engineData->device, descriptorSet, renderPass.name + " - DescriptorSet" + std::to_string(set));
	return descriptorSet;
}

VkDescriptorSetLayout RenderGraph::get_descriptor_set_layout(RenderPass& renderPass, int set)
{
	for (int i = 0; i < renderPass.extraDescriptorSets.size(); i++) {
		if (renderPass.extraDescriptorSets[i].set_index == set) {
			return renderPass.extraDescriptorSets[i].descriptorSetLayout;
		}
	}

	int bindingCounter = 0;
	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
	std::vector<VkDescriptorType> bindingDescriptorTypes;

	auto prepare = [&](const Slice<DescriptorBinding>& bindings, bool isWrite) {
		for (int i = 0; i < bindings.size(); i++) {
			if (bindings[i].set_index != set) {
				continue;
			}

			switch (bindings[i].binding->type) {
			case BindType::UNIFORM:
				layoutBindings.push_back(vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, bindingCounter));
				bindingDescriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
				break;
			case BindType::STORAGE:
				layoutBindings.push_back(vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, bindingCounter));
				bindingDescriptorTypes.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
				break;
			case BindType::IMAGE_VIEW:
				layoutBindings.push_back(vkinit::descriptorset_layout_binding(isWrite ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL, bindingCounter));
				bindingDescriptorTypes.push_back(isWrite ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
				break;
			default:
				break;
			}

			bindingCounter++;
		}
	};

	prepare(renderPass.reads, false);
	prepare(renderPass.writes, true);

	DescriptorSetLayoutCache cache1 = { bindingDescriptorTypes };

	if (descriptorSetLayoutCache.find(cache1) != descriptorSetLayoutCache.end()) {
		return descriptorSetLayoutCache[cache1];
	}
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(layoutBindings.data(), layoutBindings.size());
	VkDescriptorSetLayout newDescriptorSetLayout;

	VK_CHECK(vkCreateDescriptorSetLayout(engineData->device, &setinfo, nullptr, &newDescriptorSetLayout));

	descriptorSetLayoutCache[cache1] = newDescriptorSetLayout;
	vkutils::setObjectName(engineData->device, newDescriptorSetLayout, renderPass.name + " - DescriptorSetLayout" + std::to_string(set));

	return newDescriptorSetLayout;
}

VkImageView Vrg::RenderGraph::get_image_view(VkImage image, ImageView& imageView, VkFormat format, bool isDepth)
{
	ImageViewCache cache1 = { image, imageView };

	if (imageViewCache.find(cache1) != imageViewCache.end()) {
		return imageViewCache[cache1];
	}

	VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(format, image, isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT  : VK_IMAGE_ASPECT_COLOR_BIT);
	imageViewInfo.subresourceRange.baseMipLevel = imageView.baseMipLevel;
	imageViewInfo.subresourceRange.levelCount = imageView.mipLevelCount;
	VkImageView newImageView;

	VK_CHECK(vkCreateImageView(engineData->device, &imageViewInfo, nullptr, &newImageView));
	imageViewCache[cache1] = newImageView;

	return newImageView;
	//vkutils::setObjectName(engineData->device, newImageView, );
}

VkImageLayout Vrg::RenderGraph::get_current_image_layout(VkImage image, uint32_t mip)
{
	if (bindingImageLayout.find({image, mip}) != bindingImageLayout.end()) {
		return bindingImageLayout[{image, mip}];
	}

	return VK_IMAGE_LAYOUT_UNDEFINED;
}
