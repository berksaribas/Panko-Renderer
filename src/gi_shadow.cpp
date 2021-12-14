#include "gi_shadow.h"
#include <vk_initializers.h>
#include <vk_utils.h>
#include <vk_pipeline.h>

void Shadow::init_images(EngineData& engineData)
{
	// Create shadowmap framebuffer
	{
		VkExtent3D depthImageExtent3D = {
			_shadowMapExtent.width,
			_shadowMapExtent.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_shadowMapColorImage._image, &_shadowMapColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _shadowMapColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_shadowMapColorImageView));
		}

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.depth32Format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_shadowMapDepthImage._image, &_shadowMapDepthImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.depth32Format, _shadowMapDepthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_shadowMapDepthImageView));
		}

		VkImageView attachments[2] = { _shadowMapColorImageView, _shadowMapDepthImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(engineData.colorDepthRenderPass, _shadowMapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(engineData.device, &fb_info, nullptr, &_shadowMapFramebuffer));
	}
}

void Shadow::init_buffers(EngineData& engineData)
{
	_shadowMapDataBuffer = vkutils::create_buffer(engineData.allocator, sizeof(GPUShadowMapData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
}

void Shadow::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	//binding set for shadow map data
	VkDescriptorSetLayoutBinding shadowMapDataBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(&shadowMapDataBind, 1);
	vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_shadowMapDataSetLayout);

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_shadowMapDataSetLayout, 1);
	vkAllocateDescriptorSets(engineData.device, &allocInfo, &_shadowMapDataDescriptor);

	VkWriteDescriptorSet shadowMapDataWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _shadowMapDataDescriptor, &_shadowMapDataBuffer._descriptorBufferInfo, 0);

	VkWriteDescriptorSet setWrites[] = { shadowMapDataWrite };

	vkUpdateDescriptorSets(engineData.device, 1, setWrites, 0, nullptr);

	//Shadow map texture descriptor
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = engineData.linearSampler;
		imageBufferInfo.imageView = _shadowMapColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_shadowMapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _shadowMapTextureDescriptor, &imageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}
}

void Shadow::init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, bool rebuild)
{
	VkShaderModule shadowMapVertShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/evsm.vert.spv", &shadowMapVertShader))
	{
		assert("Shadow Vertex Shader Loading Issue");
	}

	VkShaderModule shadowMapFragShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/evsm.frag.spv", &shadowMapFragShader))
	{
		assert("Shadow Fragment Shader Loading Issue");
	}

	//SHADOWMAP PIPELINE LAYOUT INFO
	if (!rebuild)
	{
		VkDescriptorSetLayout setLayouts[] = { _shadowMapDataSetLayout, sceneDescriptors.objectSetLayout };
		VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 2);
		VK_CHECK(vkCreatePipelineLayout(engineData.device, &pipeline_layout_info, nullptr, &_shadowMapPipelineLayout));
	}
	else {
		vkDestroyPipeline(engineData.device, _shadowMapPipeline, nullptr);
	}

	//build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
	PipelineBuilder pipelineBuilder;

	//vertex input controls how to read vertices from vertex buffers. We aren't using it yet
	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

	//input assembly is the configuration for drawing triangle lists, strips, or individual points.
	//we are just going to draw triangle list
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	//configure the rasterizer to draw filled triangles
	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

	//we don't use multisampling, so just run the default one
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

	//a single blend attachment with no blending and writing to RGBA

	auto blendAttachmentState = vkinit::color_blend_attachment_state();
	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &blendAttachmentState);

	//build the mesh pipeline
	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, shadowMapVertShader));

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, shadowMapFragShader));

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
	pipelineBuilder._pipelineLayout = _shadowMapPipelineLayout;

	//build the mesh triangle pipeline
	_shadowMapPipeline = pipelineBuilder.build_pipeline(engineData.device, engineData.colorDepthRenderPass);

	vkDestroyShaderModule(engineData.device, shadowMapVertShader, nullptr);
	vkDestroyShaderModule(engineData.device, shadowMapFragShader, nullptr);
}

void Shadow::prepare_rendering(EngineData& engineData)
{
	vkutils::cpu_to_gpu(engineData.allocator, _shadowMapDataBuffer, &_shadowMapData, sizeof(GPUShadowMapData));
}

void Shadow::render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, std::function<void(VkCommandBuffer cmd)>&& function)
{
	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

	VkClearValue depthClear;
	depthClear.depthStencil = { 1.0f, 0 };
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(engineData.colorDepthRenderPass, _shadowMapExtent, _shadowMapFramebuffer);

	rpInfo.clearValueCount = 2;
	VkClearValue clearValues[] = { clearValue, depthClear };
	rpInfo.pClearValues = clearValues;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkutils::cmd_viewport_scissor(cmd, _shadowMapExtent);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipelineLayout, 0, 1, &_shadowMapDataDescriptor, 0, nullptr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipelineLayout, 1, 1, &sceneDescriptors.objectDescriptor, 0, nullptr);

	function(cmd);

	vkCmdEndRenderPass(cmd);
}

void Shadow::cleanup(EngineData& engineData)
{
	vkDestroyPipeline(engineData.device, _shadowMapPipeline, nullptr);
	vkDestroyPipelineLayout(engineData.device, _shadowMapPipelineLayout, nullptr);

	vkDestroyDescriptorSetLayout(engineData.device, _shadowMapDataSetLayout, nullptr);
	vmaDestroyBuffer(engineData.allocator, _shadowMapDataBuffer._buffer, _shadowMapDataBuffer._allocation);

	vkDestroyFramebuffer(engineData.device, _shadowMapFramebuffer, nullptr);

	vkDestroyImageView(engineData.device, _shadowMapDepthImageView, nullptr);
	vmaDestroyImage(engineData.allocator, _shadowMapDepthImage._image, _shadowMapDepthImage._allocation);

	vkDestroyImageView(engineData.device, _shadowMapColorImageView, nullptr);
	vmaDestroyImage(engineData.allocator, _shadowMapColorImage._image, _shadowMapColorImage._allocation);
}
