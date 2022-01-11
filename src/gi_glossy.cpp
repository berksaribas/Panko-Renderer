#include <gi_glossy.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include "../shaders/common.glsl"

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

	{
		VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);
		VmaAllocationCreateInfo dimg_allocinfo = {};
		dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_glossyReflectionsColorImage._image, &_glossyReflectionsColorImage._allocation, nullptr);

		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _glossyReflectionsColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
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
}

void GlossyIllumination::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer)
{
	VkDescriptorImageInfo storageImageBufferInfo;
	storageImageBufferInfo.sampler = engineData.linearSampler;
	storageImageBufferInfo.imageView = _glossyReflectionsColorImageView;
	storageImageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	{
		//Descriptors: Acceleration structure, storage buffer to save results, Materials
		VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0);
		VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
			VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
		VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
			VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);
		VkDescriptorSetLayoutBinding outBuffer = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3);
		VkDescriptorSetLayoutBinding bindings[4] = { tlasBind, sceneDescBind, meshInfoBind, outBuffer };
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 4);
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

		vkUpdateDescriptorSets(engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	{
		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_glossyReflectionsColorTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _glossyReflectionsColorTextureDescriptor, &storageImageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}

	vkutils::setObjectName(engineData.device, _glossyReflectionsColorImage._image, "GlossyReflectionsColorImage");
	vkutils::setObjectName(engineData.device, _glossyReflectionsColorImageView, "GlossyReflectionsColorImageView");
	vkutils::setObjectName(engineData.device, _glossyReflectionsColorTextureDescriptor, "GlossyReflectionsColorTextureDescriptor");
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
}

void GlossyIllumination::render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination)
{
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
}
