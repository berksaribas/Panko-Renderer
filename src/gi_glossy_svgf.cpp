#include "gi_glossy_svgf.h"
#include <vk_initializers.h>
#include <vk_utils.h>

void GlossyDenoise::init(VulkanCompute* vulkanCompute)
{
	_vulkanCompute = vulkanCompute;
}

void GlossyDenoise::init_images(EngineData& engineData, VkExtent2D imageSize)
{
	_imageSize = imageSize;

	VkExtent3D extent3D = {
			_imageSize.width,
			_imageSize.height,
			1
	};

	for (int i = 0; i < 2; i++) {
		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color16Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_temporalData[i].colorImage._image, &_temporalData[i].colorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color16Format, _temporalData[i].colorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_temporalData[i].colorImageView));
		}

		vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
			vkutils::image_barrier(cmd, _temporalData[i].colorImage._image,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
				0, VK_ACCESS_SHADER_WRITE_BIT);
			});

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color16Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_temporalData[i].momentsImage._image, &_temporalData[i].momentsImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color16Format, _temporalData[i].momentsImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_temporalData[i].momentsImageView));
		}

		vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
			vkutils::image_barrier(cmd, _temporalData[i].momentsImage._image,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
				0, VK_ACCESS_SHADER_WRITE_BIT);
			});
	}

	//atrous
	for (int i = 0; i < 2; i++) {
		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color16Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_atrousData[i].pingImage._image, &_atrousData[i].pingImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color16Format, _atrousData[i].pingImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_atrousData[i].pingImageView));
		}

		vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
			vkutils::image_barrier(cmd, _atrousData[i].pingImage._image,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
				0, VK_ACCESS_SHADER_WRITE_BIT);
			});
	}
}

void GlossyDenoise::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	{
		VkDescriptorSetLayoutBinding data[2]{ vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1) };
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(data, 2);
		vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_temporalStorageDescriptorSetLayout);
	}
	{
		VkDescriptorSetLayoutBinding data[2]{ vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1) };
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(data, 2);
		vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_temporalSampleDescriptorSetLayout);
	}
	{
		VkDescriptorSetLayoutBinding data[1]{ vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0)};
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(data, 1);
		vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_atrousPingStorageDescriptorSetLayout);
	}

	for (int i = 0; i < 2; i++) {
		{
			VkDescriptorImageInfo outputBufferInfo = { engineData.nearestSampler, _temporalData[i].colorImageView, VK_IMAGE_LAYOUT_GENERAL };
			VkDescriptorImageInfo momentsBufferInfo = { engineData.nearestSampler, _temporalData[i].momentsImageView, VK_IMAGE_LAYOUT_GENERAL };

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_temporalSampleDescriptorSetLayout, 1);
			vkAllocateDescriptorSets(engineData.device, &allocInfo, &_temporalData[i].temporalSampleDescriptor);

			VkWriteDescriptorSet textures[2] = {
				vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _temporalData[i].temporalSampleDescriptor, &outputBufferInfo, 0, 1),
				vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _temporalData[i].temporalSampleDescriptor, &momentsBufferInfo, 1, 1),
			};

			vkUpdateDescriptorSets(engineData.device, 2, textures, 0, nullptr);
		}

		{
			VkDescriptorImageInfo outputBufferInfo = { engineData.nearestSampler, _temporalData[i].colorImageView, VK_IMAGE_LAYOUT_GENERAL };
			VkDescriptorImageInfo momentsBufferInfo = { engineData.nearestSampler, _temporalData[i].momentsImageView, VK_IMAGE_LAYOUT_GENERAL };

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_temporalStorageDescriptorSetLayout, 1);
			vkAllocateDescriptorSets(engineData.device, &allocInfo, &_temporalData[i].temporalStorageDescriptor);

			VkWriteDescriptorSet textures[2] = {
				vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _temporalData[i].temporalStorageDescriptor, &outputBufferInfo, 0, 1),
				vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _temporalData[i].temporalStorageDescriptor, &momentsBufferInfo, 1, 1),
			};

			vkUpdateDescriptorSets(engineData.device, 2, textures, 0, nullptr);
		}
	}

	for (int i = 0; i < 2; i++) {
		{
			VkDescriptorImageInfo outputBufferInfo = { engineData.nearestSampler, _atrousData[i].pingImageView, VK_IMAGE_LAYOUT_GENERAL };

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);
			vkAllocateDescriptorSets(engineData.device, &allocInfo, &_atrousData[i].atrousSampleDescriptor);

			VkWriteDescriptorSet textures[1] = {
				vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _atrousData[i].atrousSampleDescriptor, &outputBufferInfo, 0, 1),
			};

			vkUpdateDescriptorSets(engineData.device, 1, textures, 0, nullptr);
		}

		{
			VkDescriptorImageInfo outputBufferInfo = { engineData.nearestSampler, _atrousData[i].pingImageView, VK_IMAGE_LAYOUT_GENERAL };

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_atrousPingStorageDescriptorSetLayout, 1);
			vkAllocateDescriptorSets(engineData.device, &allocInfo, &_atrousData[i].atrousStorageDescriptor);

			VkWriteDescriptorSet textures[1] = {
				vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _atrousData[i].atrousStorageDescriptor, &outputBufferInfo, 0, 1),
			};

			vkUpdateDescriptorSets(engineData.device, 1, textures, 0, nullptr);
		}
	}
}

void GlossyDenoise::init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, bool rebuild)
{
	if (rebuild) {
		_vulkanCompute->rebuildPipeline(_temporalFilter, "../../shaders/svgf_temporal.comp.spv");
		_vulkanCompute->rebuildPipeline(_atrousFilter, "../../shaders/svgf_atrous.comp.spv");

		return;
	}
	_vulkanCompute->add_descriptor_set_layout(_temporalFilter, sceneDescriptors.globalSetLayout); //camera
	_vulkanCompute->add_descriptor_set_layout(_temporalFilter, gbuffer._gbufferDescriptorSetLayout); //gbuffer current
	_vulkanCompute->add_descriptor_set_layout(_temporalFilter, gbuffer._gbufferDescriptorSetLayout); //gbuffer previous
	_vulkanCompute->add_descriptor_set_layout(_temporalFilter, sceneDescriptors.singleImageSetLayout); //raytracing result
	_vulkanCompute->add_descriptor_set_layout(_temporalFilter, _temporalSampleDescriptorSetLayout); //raytracing result
	_vulkanCompute->add_descriptor_set_layout(_temporalFilter, _temporalStorageDescriptorSetLayout); //raytracing result
	_vulkanCompute->build(_temporalFilter, engineData.descriptorPool, "../../shaders/svgf_temporal.comp.spv");


	_vulkanCompute->add_descriptor_set_layout(_atrousFilter, sceneDescriptors.globalSetLayout); //camera
	_vulkanCompute->add_descriptor_set_layout(_atrousFilter, gbuffer._gbufferDescriptorSetLayout); //gbuffer current
	_vulkanCompute->add_descriptor_set_layout(_atrousFilter, sceneDescriptors.singleImageSetLayout); //raytracing result
	_vulkanCompute->add_descriptor_set_layout(_atrousFilter, _temporalStorageDescriptorSetLayout); //raytracing result
	VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_COMPUTE_BIT , 0, sizeof(int) };
	_atrousFilter.pushConstantRangeCount = 1;
	_atrousFilter.pushConstantRange = &pushConstantRanges;
	_vulkanCompute->build(_atrousFilter, engineData.descriptorPool, "../../shaders/svgf_atrous.comp.spv");
}

void GlossyDenoise::render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, GlossyIllumination& glossyIllumination)
{
	_currFrame++;

	auto gbufferCurrent = gbuffer.getGbufferCurrentDescriptorSet();
	auto gbufferPrevious = gbuffer.getGbufferPreviousFrameDescriptorSet();

	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipelineLayout, 0, 1, &sceneDescriptors.globalDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipelineLayout, 1, 1, &gbufferCurrent, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipelineLayout, 2, 1, &gbufferPrevious, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipelineLayout, 3, 1, &glossyIllumination._glossyReflectionsColorTextureDescriptor, 0, nullptr);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipelineLayout, 4, 1, &_temporalData[(_currFrame -1) % 2].temporalSampleDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _temporalFilter.pipelineLayout, 5, 1, &_temporalData[(_currFrame) % 2].temporalStorageDescriptor, 0, nullptr);

		vkCmdDispatch(cmd, static_cast<uint32_t>(ceil(float(_imageSize.width) / float(16))), static_cast<uint32_t>(ceil(float(_imageSize.height) / float(16))), 1);
	}

	vkutils::image_barrier(cmd, _temporalData[(_currFrame) % 2].colorImage._image,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		0, VK_ACCESS_TRANSFER_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	vkutils::image_barrier(cmd, _atrousData[0].pingImage._image,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	// Issue the copy command
	VkImageCopy imageCopyRegion{};
	imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageCopyRegion.srcSubresource.layerCount = 1;
	imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageCopyRegion.dstSubresource.layerCount = 1;
	imageCopyRegion.extent.width = _imageSize.width;
	imageCopyRegion.extent.height = _imageSize.height;
	imageCopyRegion.extent.depth = 1;

	vkCmdCopyImage(
		cmd,
		_temporalData[(_currFrame) % 2].colorImage._image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		_atrousData[0].pingImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&imageCopyRegion);

	vkutils::image_barrier(cmd, _temporalData[(_currFrame) % 2].colorImage._image,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		0, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	vkutils::image_barrier(cmd, _atrousData[0].pingImage._image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		0, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	for(int i = 0; i < 4; i++)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _atrousFilter.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _atrousFilter.pipelineLayout, 0, 1, &sceneDescriptors.globalDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _atrousFilter.pipelineLayout, 1, 1, &gbufferCurrent, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _atrousFilter.pipelineLayout, 2, 1, &_atrousData[(i) % 2].atrousSampleDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _atrousFilter.pipelineLayout, 3, 1, &_atrousData[(i + 1) % 2].atrousStorageDescriptor, 0, nullptr);

		int stepsize = 1u << i;
		vkCmdPushConstants(cmd, _atrousFilter.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &stepsize);
		vkCmdDispatch(cmd, static_cast<uint32_t>(ceil(float(_imageSize.width) / float(16))), static_cast<uint32_t>(ceil(float(_imageSize.height) / float(16))), 1);
	}

	vkutils::image_barrier(cmd, _atrousData[0].pingImage._image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

}

VkDescriptorSet GlossyDenoise::getDenoisedDescriptor()
{
	return _atrousData[0].atrousSampleDescriptor;
}

void GlossyDenoise::cleanup()
{
}
