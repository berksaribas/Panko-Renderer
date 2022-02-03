#include <gi_glossy.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include "../shaders/common.glsl"
#include <vk_pipeline.h>

uint32_t mipmapLevels;

void GlossyIllumination::init(VulkanRaytracing& vulkanRaytracing)
{
	_vulkanRaytracing = &vulkanRaytracing;
}

void GlossyIllumination::init_images(EngineData& engineData, VkExtent2D imageSize)
{
	_imageSize = imageSize;

	VkExtent3D extent3D = {
			_imageSize.width,
			_imageSize.height,
			1
	};

	mipmapLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(_imageSize.width, _imageSize.height)))) + 1;

	// COLOR IMAGE
	{
		VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent3D, mipmapLevels);
		VmaAllocationCreateInfo dimg_allocinfo = {};
		dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_glossyReflectionsColorImage._image, &_glossyReflectionsColorImage._allocation, nullptr);

		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _glossyReflectionsColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT, mipmapLevels);
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_glossyReflectionsColorImageView));
	}

	vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = _glossyReflectionsColorImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	});

	_colorMipmapImageViews = std::vector<VkImageView>(mipmapLevels);
	_colorMipmapFramebuffers = std::vector<VkFramebuffer>(mipmapLevels);

	for (int i = 0; i < mipmapLevels; i++) {
		//create image views and their framebuffers.
		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _glossyReflectionsColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
		imageViewInfo.subresourceRange.baseMipLevel = i;
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_colorMipmapImageViews[i]));

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(engineData.colorRenderPass, { imageSize.width / (int)pow(2, i), imageSize.height / (int)pow(2, i) });
		fb_info.pAttachments = &_colorMipmapImageViews[i];
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(engineData.device, &fb_info, nullptr, &_colorMipmapFramebuffers[i]));
	}

	// NORMAL IMAGE
	{
		VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent3D, mipmapLevels);
		VmaAllocationCreateInfo dimg_allocinfo = {};
		dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_normalImage._image, &_normalImage._allocation, nullptr);
		
		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _normalImage._image, VK_IMAGE_ASPECT_COLOR_BIT, mipmapLevels);
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_normalImageView));

	}

	vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = _normalImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
		});
	_normalMipmapImageViews = std::vector<VkImageView>(mipmapLevels);
	_normalMipmapFramebuffers = std::vector<VkFramebuffer>(mipmapLevels);

	for (int i = 0; i < mipmapLevels; i++) {
		//create image views and their framebuffers.
		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _normalImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
		imageViewInfo.subresourceRange.baseMipLevel = i;
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_normalMipmapImageViews[i]));

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(engineData.colorRenderPass, { imageSize.width / (int)pow(2, i), imageSize.height / (int)pow(2, i) });
		fb_info.pAttachments = &_normalMipmapImageViews[i];
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(engineData.device, &fb_info, nullptr, &_normalMipmapFramebuffers[i]));
	}

	// TEMP IMAGE
	{
		_tempMipmapImageViews = std::vector<VkImageView>(mipmapLevels);
		_tempMipmapFramebuffers = std::vector<VkFramebuffer>(mipmapLevels);


		VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent3D, mipmapLevels);
		VmaAllocationCreateInfo dimg_allocinfo = {};
		dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_tempImage._image, &_tempImage._allocation, nullptr);

		for (int i = 0; i < mipmapLevels; i++) {
			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _tempImage._image, VK_IMAGE_ASPECT_COLOR_BIT, mipmapLevels);
			imageViewInfo.subresourceRange.baseMipLevel = i;
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_tempMipmapImageViews[i]));

			VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(engineData.colorRenderPass, { imageSize.width / (int)pow(2, i), imageSize.height / (int)pow(2, i) });
			fb_info.pAttachments = &_tempMipmapImageViews[i];
			fb_info.attachmentCount = 1;
			VK_CHECK(vkCreateFramebuffer(engineData.device, &fb_info, nullptr, &_tempMipmapFramebuffers[i]));
		}
	}
}

void GlossyIllumination::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer)
{
	VkSampler sampler;
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = FLT_MAX;
	VK_CHECK(vkCreateSampler(engineData.device, &samplerInfo, nullptr, &sampler));

	VkDescriptorImageInfo storageImageBufferInfo;
	storageImageBufferInfo.sampler = sampler;
	storageImageBufferInfo.imageView = _colorMipmapImageViews[0];
	storageImageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo storageImageBufferInfo2;
	storageImageBufferInfo2.sampler = sampler;
	storageImageBufferInfo2.imageView = _normalMipmapImageViews[0];
	storageImageBufferInfo2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	{
		//Descriptors: Acceleration structure, storage buffer to save results, Materials
		VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0);
		VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
			VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
		VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
			VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);
		VkDescriptorSetLayoutBinding outBuffer = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3);
		VkDescriptorSetLayoutBinding outNormalBuffer = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4);
		VkDescriptorSetLayoutBinding bindings[5] = { tlasBind, sceneDescBind, meshInfoBind, outBuffer, outNormalBuffer };
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 5);
		vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &rtDescriptorSetLayout);

		VkDescriptorSetAllocateInfo allocateInfo =
			vkinit::descriptorset_allocate_info(engineData.descriptorPool, &rtDescriptorSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocateInfo, &rtDescriptorSet);

		std::vector<VkWriteDescriptorSet> writes;

		VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
		descASInfo.accelerationStructureCount = 1;
		descASInfo.pAccelerationStructures = &_vulkanRaytracing->tlas.accel;
		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext = &descASInfo;
		accelerationStructureWrite.dstSet = rtDescriptorSet;
		accelerationStructureWrite.dstBinding = 0;
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

		writes.emplace_back(accelerationStructureWrite);
		writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, rtDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));
		writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));
		writes.emplace_back(vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, rtDescriptorSet, &storageImageBufferInfo, 3, 1));
		writes.emplace_back(vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, rtDescriptorSet, &storageImageBufferInfo2, 4, 1));

		vkUpdateDescriptorSets(engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	storageImageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	storageImageBufferInfo.imageView = _glossyReflectionsColorImageView;

	{
		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_glossyReflectionsColorTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _glossyReflectionsColorTextureDescriptor, &storageImageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}

	{
		storageImageBufferInfo.imageView = _tempMipmapImageViews[0];

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_tempMipmapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _tempMipmapTextureDescriptor, &storageImageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}

	{
		storageImageBufferInfo.imageView = _normalImageView;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_normalMipmapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _normalMipmapTextureDescriptor, &storageImageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}


	vkutils::setObjectName(engineData.device, _glossyReflectionsColorImage._image, "GlossyReflectionsColorImage");
	vkutils::setObjectName(engineData.device, _glossyReflectionsColorImageView, "GlossyReflectionsColorImageView");
	vkutils::setObjectName(engineData.device, _glossyReflectionsColorTextureDescriptor, "GlossyReflectionsColorTextureDescriptor");
}

void GlossyIllumination::init_blur_pipeline(EngineData& engineData, SceneDescriptors& sceneDescriptors) {
	{
		VkDescriptorSetLayout setLayouts[] = { sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout };
		VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 2);

		VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_FRAGMENT_BIT , 0, sizeof(glm::vec3)};
		pipeline_layout_info.pushConstantRangeCount = 1;
		pipeline_layout_info.pPushConstantRanges = &pushConstantRanges;

		VK_CHECK(vkCreatePipelineLayout(engineData.device, &pipeline_layout_info, nullptr, &_blurPipelineLayout));
	}

	VkShaderModule fullscreenVertShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/fullscreen.vert.spv", &fullscreenVertShader))
	{
		assert("Fullscreen vertex Shader Loading Issue");
	}

	VkShaderModule blurFragShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/gaussblur.frag.spv", &blurFragShader))
	{
		assert("Dilation Fragment Shader Loading Issue");
	}

	PipelineBuilder pipelineBuilder;

	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();
	auto blendState = vkinit::color_blend_attachment_state();
	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &blendState);

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, fullscreenVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, blurFragShader));
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
	
	pipelineBuilder._pipelineLayout = _blurPipelineLayout;
	_blurPipeline = pipelineBuilder.build_pipeline(engineData.device, engineData.colorRenderPass);
}

void GlossyIllumination::init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, bool rebuild)
{
	if (rebuild) {
		_vulkanRaytracing->destroy_raytracing_pipeline(rtPipeline);
	}
	
	VkDescriptorSetLayout setLayouts[] = { rtDescriptorSetLayout, sceneDescriptors.globalSetLayout, sceneDescriptors.objectSetLayout, gbuffer._gbufferDescriptorSetLayout, sceneDescriptors.textureSetLayout, sceneDescriptors.materialSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout };
	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 8);

	_vulkanRaytracing->create_new_pipeline(rtPipeline, pipeline_layout_info,
		"../../shaders/reflections_rt.rgen.spv",
		"../../shaders/reflections_rt.rmiss.spv",
		"../../shaders/reflections_rt.rchit.spv");

	vkutils::setObjectName(engineData.device, rtPipeline.pipeline, "GlossyRTPipeline");
	vkutils::setObjectName(engineData.device, rtPipeline.pipelineLayout, "GlossyRTPipelineLayout");

	init_blur_pipeline(engineData, sceneDescriptors);
}

void GlossyIllumination::render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination)
{
	{
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		// We won't be changing the layout of the image
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = _glossyReflectionsColorImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	}

	{
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		// We won't be changing the layout of the image
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = _normalImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipeline);
	
	std::vector<VkDescriptorSet> descSets{ rtDescriptorSet, sceneDescriptors.globalDescriptor, sceneDescriptors.objectDescriptor, gbuffer._gbufferDescriptorSet, sceneDescriptors.textureDescriptor, sceneDescriptors.materialDescriptor, shadow._shadowMapTextureDescriptor, diffuseIllumination._giIndirectLightTextureDescriptor };

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipelineLayout, 0,
		(uint32_t)descSets.size(), descSets.data(), 0, nullptr);

	vkCmdTraceRaysKHR(cmd, &rtPipeline.rgenRegion, &rtPipeline.missRegion, &rtPipeline.hitRegion, &rtPipeline.callRegion, _imageSize.width, _imageSize.height, 1);

	{
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		// We won't be changing the layout of the image
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.image = _glossyReflectionsColorImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	}

	{
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		// We won't be changing the layout of the image
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.image = _normalImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	}

	//Do gaussian blur on each mip chain. Each mip will sample from the previous mip. I'll need a temp image as well
	for (int i = 1; i < mipmapLevels; i++) {
		VkExtent2D previousMipSize = { _imageSize.width / (int)pow(2, i - 1), _imageSize.height / (int)pow(2, i - 1) };
		VkExtent2D mipSize = { _imageSize.width / (int)pow(2, i), _imageSize.height / (int)pow(2, i) };
		VkClearValue clearValue;

		clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		{
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(engineData.colorRenderPass, mipSize, _normalMipmapFramebuffers[i]);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkutils::cmd_viewport_scissor(cmd, mipSize);

			glm::vec3 pushData = { 2, 0, i - 1 };

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipeline);
			vkCmdPushConstants(cmd, _blurPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &pushData);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipelineLayout, 0, 1, &_normalMipmapTextureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipelineLayout, 1, 1, &_normalMipmapTextureDescriptor, 0, nullptr);
			vkCmdDraw(cmd, 3, 1, 0, 0);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}

		
		{
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(engineData.colorRenderPass, previousMipSize, _tempMipmapFramebuffers[i - 1]);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkutils::cmd_viewport_scissor(cmd, previousMipSize);

			glm::vec3 pushData = { 1, 0, i - 1 };

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipeline);
			vkCmdPushConstants(cmd, _blurPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &pushData);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipelineLayout, 0, 1, &_glossyReflectionsColorTextureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipelineLayout, 1, 1, &_normalMipmapTextureDescriptor, 0, nullptr);

			vkCmdDraw(cmd, 3, 1, 0, 0);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}
		

		{
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(engineData.colorRenderPass, mipSize, _colorMipmapFramebuffers[i]);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkutils::cmd_viewport_scissor(cmd, mipSize);

			glm::vec3 pushData = { 0, 1, i - 1 };

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipeline);
			vkCmdPushConstants(cmd, _blurPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &pushData);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipelineLayout, 0, 1, &_tempMipmapTextureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blurPipelineLayout, 1, 1, &_normalMipmapTextureDescriptor, 0, nullptr);

			vkCmdDraw(cmd, 3, 1, 0, 0);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}
	}
}
