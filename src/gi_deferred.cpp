#include <gi_deferred.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include <vk_pipeline.h>

void Deferred::init_images(EngineData& engineData, VkExtent2D imageSize)
{
	_imageSize = imageSize;

	{
		VkExtent3D extent3D = {
			_imageSize.width,
			_imageSize.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_deferredColorImage._image, &_deferredColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _deferredColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_deferredColorImageView));
		}

		VkImageView attachments[1] = { _deferredColorImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(engineData.colorRenderPass, _imageSize);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(engineData.device, &fb_info, nullptr, &_deferredFramebuffer));
	}
}

void Deferred::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = engineData.nearestSampler; 
		imageBufferInfo.imageView = _deferredColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_deferredColorTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _deferredColorTextureDescriptor, &imageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}
}

void Deferred::init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, bool rebuild)
{
	VkShaderModule deferredFragShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/deferred.frag.spv", &deferredFragShader))
	{
		assert("Deferred Fragment Shader Loading Issue");
	}

	VkShaderModule fullscreenVertShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/fullscreen.vert.spv", &fullscreenVertShader))
	{
		assert("Deferred vertex Shader Loading Issue");
	}

	if (!rebuild)
	{
		VkDescriptorSetLayout setLayouts[] = { sceneDescriptors.globalSetLayout, gbuffer._gbufferDescriptorSetLayout, sceneDescriptors.textureSetLayout, sceneDescriptors.materialSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout };
		VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 7);
		VK_CHECK(vkCreatePipelineLayout(engineData.device, &pipeline_layout_info, nullptr, &_deferredPipelineLayout));
	}
	else {
		vkDestroyPipeline(engineData.device, _deferredPipeline, nullptr);
	}

	PipelineBuilder pipelineBuilder;
	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

	auto blendAttachmentState = vkinit::color_blend_attachment_state();
	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &blendAttachmentState);

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, fullscreenVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, deferredFragShader));
	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
	pipelineBuilder._rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	VkPipelineVertexInputStateCreateInfo emptyInputState = {};
	emptyInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	emptyInputState.vertexAttributeDescriptionCount = 0;
	emptyInputState.pVertexAttributeDescriptions = nullptr;
	emptyInputState.vertexBindingDescriptionCount = 0;
	emptyInputState.pVertexBindingDescriptions = nullptr;
	pipelineBuilder._vertexInputInfo = emptyInputState;
	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(false, false, VK_COMPARE_OP_LESS_OR_EQUAL);
	pipelineBuilder._pipelineLayout = _deferredPipelineLayout;

	_deferredPipeline = pipelineBuilder.build_pipeline(engineData.device, engineData.colorRenderPass);

	vkDestroyShaderModule(engineData.device, deferredFragShader, nullptr);
	vkDestroyShaderModule(engineData.device, fullscreenVertShader, nullptr);
}

void Deferred::render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination, GlossyIllumination& glossyIllumination)
{
	VkClearValue clearValue;
	clearValue.color = { { 0, 0, 0, 0 } };

	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(engineData.colorRenderPass, _imageSize, _deferredFramebuffer);

	rpInfo.clearValueCount = 1;
	VkClearValue clearValues[] = { clearValue };
	rpInfo.pClearValues = clearValues;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkutils::cmd_viewport_scissor(cmd, _imageSize);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 0, 1, &sceneDescriptors.globalDescriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 1, 1, &gbuffer._gbufferDescriptorSet, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 2, 1, &sceneDescriptors.textureDescriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 3, 1, &sceneDescriptors.materialDescriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 4, 1, &shadow._shadowMapTextureDescriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 5, 1, &diffuseIllumination._dilatedGiIndirectLightTextureDescriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _deferredPipelineLayout, 6, 1, &glossyIllumination._glossyReflectionsColorTextureDescriptor, 0, nullptr);

	vkCmdDraw(cmd, 3, 1, 0, 0);

	//finalize the render pass
	vkCmdEndRenderPass(cmd);
}
