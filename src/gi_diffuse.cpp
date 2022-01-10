#include "gi_diffuse.h"
#include <vk_initializers.h>
#include <random>
#include <vk_utils.h>

uint wang_hash(uint& seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}

float random_float(uint& state)
{
	return (wang_hash(state) & 0xFFFFFF) / 16777216.0f;
}

float random_float_between(uint& state, float min, float max) {
	return min + (max - min) * random_float(state);
}

vec3 random_unit_vector(uint& state) {
	float a = random_float_between(state, 0.0, 2.0 * 3.14159265358979323846264);
	float z = random_float_between(state, -1.0, 1.0);
	float r = sqrt(1.0 - z * z);
	return vec3(r * cos(a), r * sin(a), z);
}

void DiffuseIllumination::init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, VulkanCompute* vulkanCompute, VulkanRaytracing* vulkanRaytracing, GltfScene& scene, SceneDescriptors& sceneDescriptors, VkImageView lightmapImageView)
{
	_device = engineData.device;
	_allocator = engineData.allocator;
	_descriptorPool = engineData.descriptorPool;
	_colorRenderPass = engineData.colorRenderPass;

	_vulkanCompute = vulkanCompute;
	_vulkanRaytracing = vulkanRaytracing;
	_precalculationInfo = precalculationInfo;
	_precalculationLoadData = precalculationLoadData;
	_precalculationResult = precalculationResult;

	//Config buffer (GPU ONLY)
	_config.probeCount = _precalculationResult->probes.size();
	_config.basisFunctionCount = SPHERICAL_HARMONICS_NUM_COEFF(_precalculationInfo->sphericalHarmonicsOrder);
	_config.rayCount = _precalculationInfo->raysPerProbe;
	_config.clusterCount = _precalculationLoadData->aabbClusterCount;
	_config.lightmapInputSize = glm::vec2(scene.lightmap_width, scene.lightmap_height);
	_config.pcaCoefficient = _precalculationInfo->clusterCoefficientCount;
	_config.maxReceiversInCluster = _precalculationInfo->maxReceiversInCluster;

	_giLightmapExtent.width = precalculationInfo->lightmapResolution;
	_giLightmapExtent.height = precalculationInfo->lightmapResolution;

	VkExtent3D lightmapImageExtent3D = {
		_giLightmapExtent.width,
		_giLightmapExtent.height,
		1
	};

	{
		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_giIndirectLightImage._image, &_giIndirectLightImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _giIndirectLightImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_giIndirectLightImageView));
		}

		{
			VkDescriptorImageInfo imageBufferInfo;
			imageBufferInfo.sampler = engineData.linearSampler;
			imageBufferInfo.imageView = _giIndirectLightImageView;
			imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

			vkAllocateDescriptorSets(_device, &allocInfo, &_giIndirectLightTextureDescriptor);

			VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _giIndirectLightTextureDescriptor, &imageBufferInfo, 0, 1);

			vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
		}
	}

	vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = _giIndirectLightImage._image;
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

	{
		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engineData.color32Format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_dilatedGiIndirectLightImage._image, &_dilatedGiIndirectLightImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engineData.color32Format, _dilatedGiIndirectLightImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_dilatedGiIndirectLightImageView));
		}

		VkImageView attachments[1] = { _dilatedGiIndirectLightImageView };
		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_colorRenderPass, _giLightmapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_dilatedGiIndirectLightFramebuffer));

		{
			VkDescriptorImageInfo imageBufferInfo;
			imageBufferInfo.sampler = engineData.linearSampler;
			imageBufferInfo.imageView = _dilatedGiIndirectLightImageView;
			imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);

			vkAllocateDescriptorSets(_device, &allocInfo, &_dilatedGiIndirectLightTextureDescriptor);

			VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _dilatedGiIndirectLightTextureDescriptor, &imageBufferInfo, 0, 1);

			vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
		}
	}

	_configBuffer = vkutils::create_upload_buffer(&engineData, &_config, sizeof(GIConfig), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	//GPUProbeRaycastResult buffer (GPU ONLY)
	auto probeRaycastResultBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probeRaycastResult, sizeof(GPUProbeRaycastResult) * _config.probeCount * _config.rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//ProbeBasisFunctions buffer (GPU ONLY)
	auto probeBasisBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probeRaycastBasisFunctions, sizeof(glm::vec4) * (_config.rayCount * _config.basisFunctionCount / 4 + 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_probe_projection Temp buffer (GPU ONLY)
	auto probeRelightTempBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * _config.probeCount * _config.rayCount * _config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_probe_projection output buffer (GPU ONLY)
	_probeRelightOutputBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * _config.probeCount * _config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	//Create compute instances
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, probeRaycastResultBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, probeBasisBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, probeRelightTempBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, _probeRelightOutputBuffer);
	_vulkanCompute->add_texture_binding(_probeRelight, ComputeBufferType::TEXTURE_SAMPLED, engineData.linearSampler, lightmapImageView);
	_vulkanCompute->add_texture_binding(_probeRelight, ComputeBufferType::TEXTURE_SAMPLED, engineData.linearSampler, _dilatedGiIndirectLightImageView);

	_vulkanCompute->add_descriptor_set_layout(_probeRelight, sceneDescriptors.objectSetLayout);
	_vulkanCompute->add_descriptor_set_layout(_probeRelight, sceneDescriptors.materialSetLayout);
	_vulkanCompute->add_descriptor_set_layout(_probeRelight, sceneDescriptors.textureSetLayout);

	_vulkanCompute->build(_probeRelight, _descriptorPool, "../../shaders/gi_probe_projection.comp.spv");

	//Cluster projection matrices (GPU ONLY)
	auto clusterProjectionMatricesBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterProjectionMatrices, (_precalculationLoadData->totalProbesPerCluster * _config.basisFunctionCount * _config.pcaCoefficient / 4 + 1) * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_cluster_projection output buffer (GPU ONLY)
	_clusterProjectionOutputBuffer = vkutils::create_buffer(_allocator, _config.clusterCount * _config.pcaCoefficient * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterReceiverInfos = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterReceiverInfos, _config.clusterCount * sizeof(ClusterReceiverInfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterProbes = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterProbes, (_precalculationLoadData->totalProbesPerCluster / 4 + 1) * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, _probeRelightOutputBuffer);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, clusterProjectionMatricesBuffer);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, clusterReceiverInfos);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, clusterProbes);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, _clusterProjectionOutputBuffer);
	_vulkanCompute->build(_clusterProjection, _descriptorPool, "../../shaders/gi_cluster_projection.comp.spv");


	auto receiverReconstructionMatricesBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->receiverCoefficientMatrices, (_precalculationLoadData->totalClusterReceiverCount * _config.pcaCoefficient / 4 + 1) * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterReceiverUvs = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterReceiverUvs, _precalculationLoadData->totalClusterReceiverCount * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);


	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, _clusterProjectionOutputBuffer);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, receiverReconstructionMatricesBuffer);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, clusterReceiverInfos);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, clusterReceiverUvs);
	_vulkanCompute->add_texture_binding(_receiverReconstruction, ComputeBufferType::TEXTURE_STORAGE, 0, _giIndirectLightImageView);
	_vulkanCompute->build(_receiverReconstruction, _descriptorPool, "../../shaders/gi_receiver_reconstruction.comp.spv");
	
	vkutils::setObjectName(engineData.device, _receiverReconstruction.pipeline, "DiffuseReceiverReconstructionPipeline");
	vkutils::setObjectName(engineData.device, _receiverReconstruction.pipelineLayout, "DiffuseReceiverReconstructionPipelineLayout");

	vkutils::setObjectName(engineData.device, _clusterProjection.pipeline, "DiffuseClusterProjectionPipeline");
	vkutils::setObjectName(engineData.device, _clusterProjection.pipelineLayout, "DiffuseClusterProjectionPipelineLayout");

	vkutils::setObjectName(engineData.device, _probeRelight.pipeline, "DiffuseProbeRelightPipeline");
	vkutils::setObjectName(engineData.device, _probeRelight.pipelineLayout, "DiffuseProbeRelightPipelineLayout");

	vkutils::setObjectName(engineData.device, _giIndirectLightImage._image, "DiffuseIndirectLightImage");
	vkutils::setObjectName(engineData.device, _giIndirectLightImageView, "DiffuseIndirectLightImageView");

	vkutils::setObjectName(engineData.device, _dilatedGiIndirectLightImage._image, "DilatedDiffuseIndirectLightImage");
	vkutils::setObjectName(engineData.device, _dilatedGiIndirectLightImageView, "DilatedDiffuseIndirectLightImageView");

	vkutils::setObjectName(engineData.device, _giIndirectLightTextureDescriptor, "DiffuseIndirectLightTextureDescriptor");
	vkutils::setObjectName(engineData.device, _dilatedGiIndirectLightTextureDescriptor, "DilatedDiffuseIndirectLightTextureDescriptor");		
}

void DiffuseIllumination::render(VkCommandBuffer cmd, VkPipeline dilationPipeline, VkPipelineLayout dilationPipelineLayout, SceneDescriptors& sceneDescriptors)
{
	//GI - Probe relight
	{
		int groupcount = ((_precalculationResult->probes.size() * _precalculationInfo->raysPerProbe) / 256) + 1;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipelineLayout, 0, 1, &_probeRelight.descriptorSet, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipelineLayout, 1, 1, &sceneDescriptors.objectDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipelineLayout, 2, 1, &sceneDescriptors.materialDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipelineLayout, 3, 1, &sceneDescriptors.textureDescriptor, 0, nullptr);

		vkCmdDispatch(cmd, groupcount, 1, 1);
	}

	{
		VkBufferMemoryBarrier barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.offset = 0;
		barrier.size = VK_WHOLE_SIZE;
		barrier.buffer = _probeRelightOutputBuffer._buffer;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 1, &barrier, 0, nullptr);
	}

	//GI - Cluster Projection
	{

		int groupcount = ((_precalculationLoadData->aabbClusterCount * _precalculationInfo->clusterCoefficientCount) / 256) + 1;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _clusterProjection.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _clusterProjection.pipelineLayout, 0, 1, &_clusterProjection.descriptorSet, 0, nullptr);
		vkCmdDispatch(cmd, groupcount, 1, 1);
	}

	{
		VkBufferMemoryBarrier barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.offset = 0;
		barrier.size = VK_WHOLE_SIZE;
		barrier.buffer = _clusterProjectionOutputBuffer._buffer;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 1, &barrier, 0, nullptr);
	}

	//GI - Receiver Projection
	{
		int groupcount = ((_precalculationLoadData->aabbClusterCount * _precalculationInfo->maxReceiversInCluster) / 256) + 1;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _receiverReconstruction.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _receiverReconstruction.pipelineLayout, 0, 1, &_receiverReconstruction.descriptorSet, 0, nullptr);
		vkCmdDispatch(cmd, groupcount, 1, 1);
	}


	{
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		// We won't be changing the layout of the image
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = _giIndirectLightImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);
	}

	// GI LIGHTMAP DILATION RENDERIN
	{
		VkClearValue clearValue;
		clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

		VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_colorRenderPass, _giLightmapExtent, _dilatedGiIndirectLightFramebuffer);

		rpInfo.clearValueCount = 1;
		VkClearValue clearValues[] = { clearValue };
		rpInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkutils::cmd_viewport_scissor(cmd, _giLightmapExtent);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dilationPipeline);
		vkCmdPushConstants(cmd, dilationPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::ivec2), &_giLightmapExtent);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dilationPipelineLayout, 0, 1, &_giIndirectLightTextureDescriptor, 0, nullptr);
		vkCmdDraw(cmd, 3, 1, 0, 0);

		//finalize the render pass
		vkCmdEndRenderPass(cmd);
	}
}

void DiffuseIllumination::rebuild_shaders()
{
	_vulkanCompute->rebuildPipeline(_probeRelight, "../../shaders/gi_probe_projection.comp.spv");
	_vulkanCompute->rebuildPipeline(_clusterProjection, "../../shaders/gi_cluster_projection.comp.spv");
	_vulkanCompute->rebuildPipeline(_receiverReconstruction, "../../shaders/gi_receiver_reconstruction.comp.spv");
}

void DiffuseIllumination::debug_draw_probes(VulkanDebugRenderer& debugRenderer, bool showProbeRays, float sceneScale)
{
	for (int i = 0; i < _precalculationResult->probes.size(); i++) {
		debugRenderer.draw_point(glm::vec3(_precalculationResult->probes[i]) * sceneScale, { 1, 0, 0 });
		if (showProbeRays) {
			for (int j = 0; j < _precalculationInfo->raysPerProbe; j += 400) {
				auto& ray = _precalculationResult->probeRaycastResult[_precalculationInfo->raysPerProbe * i + j];
				if (ray.objectId != -1) {
					debugRenderer.draw_line(glm::vec3(_precalculationResult->probes[i]) * sceneScale,
						glm::vec3(ray.worldPos) * sceneScale,
						{ 0, 0, 1 });

					debugRenderer.draw_point(glm::vec3(ray.worldPos) * sceneScale, { 0, 0, 1 });
				}
				else {
					debugRenderer.draw_line(glm::vec3(_precalculationResult->probes[i]) * sceneScale,
						glm::vec3(_precalculationResult->probes[i]) * sceneScale + glm::vec3(ray.direction) * 10.f,
						{ 0, 0, 1 });

				}
			}
		}
	}
}

void DiffuseIllumination::debug_draw_receivers(VulkanDebugRenderer& debugRenderer, float sceneScale)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	rng.seed(0);
	std::uniform_real_distribution<> dist(0, 1);

	for (int i = 0; i < _precalculationLoadData->aabbClusterCount; i += 1) {
		glm::vec3 color = { dist(rng), dist(rng) , dist(rng) };
		int receiverCount = _precalculationResult->clusterReceiverInfos[i].receiverCount;
		int receiverOffset = _precalculationResult->clusterReceiverInfos[i].receiverOffset;

		for (int j = receiverOffset; j < receiverOffset + receiverCount; j++) {
			debugRenderer.draw_point(_precalculationResult->aabbReceivers[j].position * sceneScale, color);
			//debugRenderer.draw_line(precalculation._aabbClusters[i].receivers[j].position * _sceneScale, (precalculation._aabbClusters[i].receivers[j].position + precalculation._aabbClusters[i].receivers[j].normal * 2.f) * _sceneScale, color);
		}
	}
}

void DiffuseIllumination::debug_draw_specific_receiver(VulkanDebugRenderer& debugRenderer, int specificCluster, int specificReceiver, int specificReceiverRaySampleCount, bool* enabledProbes, bool showSpecificProbeRays, float sceneScale)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	rng.seed(0);
	std::uniform_real_distribution<> dist(0, 1);

	int receiverCount = _precalculationResult->clusterReceiverInfos[specificCluster].receiverCount;
	int receiverOffset = _precalculationResult->clusterReceiverInfos[specificCluster].receiverOffset;

	auto receiverPos = _precalculationResult->aabbReceivers[receiverOffset + specificReceiver].position * sceneScale;
	auto receiverNormal = _precalculationResult->aabbReceivers[receiverOffset + specificReceiver].normal;
	debugRenderer.draw_point(receiverPos, { 1, 0, 0 });

	debugRenderer.draw_line(receiverPos, receiverPos + receiverNormal * 50.0f, { 0, 1, 0 });

	for (int abc = 0; abc < specificReceiverRaySampleCount; abc++) {
		uint random_state = (specificReceiver * 1973 + 9277 * abc + specificReceiver * 26699) | 1;
		vec3 direction = normalize(receiverNormal + random_unit_vector(random_state));

		debugRenderer.draw_line(receiverPos, receiverPos + direction * 100.0f, { 0, 1, 1 });
	}


	for (int i = 0; i < _precalculationResult->probes.size(); i++) {
		if (_precalculationResult->receiverProbeWeightData[(receiverOffset + specificReceiver) * _precalculationResult->probes.size() + i] > 0.000001) {
			if (enabledProbes[i]) {
				debugRenderer.draw_point(glm::vec3(_precalculationResult->probes[i]) * sceneScale, { 1, 0, 1 });

				if (showSpecificProbeRays) {
					for (int j = 0; j < _precalculationInfo->raysPerProbe; j += 1) {
						auto& ray = _precalculationResult->probeRaycastResult[_precalculationInfo->raysPerProbe * i + j];
						if (ray.objectId != -1) {
							debugRenderer.draw_line(glm::vec3(_precalculationResult->probes[i]) * sceneScale,
								glm::vec3(ray.worldPos) * sceneScale,
								{ 0, 0, 1 });

							debugRenderer.draw_point(glm::vec3(ray.worldPos) * sceneScale, { 0, 0, 1 });
						}
						else {
							debugRenderer.draw_line(glm::vec3(_precalculationResult->probes[i]) * sceneScale,
								glm::vec3(_precalculationResult->probes[i]) * sceneScale + glm::vec3(ray.direction) * 1000.f,
								{ 0, 0, 1 });

						}
					}
				}
			}
		}
	}
}
