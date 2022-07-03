#include "gi_diffuse.h"
#include <vk_initializers.h>
#include <random>
#include <vk_utils.h>
#include <vk_pipeline.h>
#include <unordered_set>

glm::vec3 calculate_barycentric(glm::vec2 p, glm::vec2 a, glm::vec2 b, glm::vec2 c);
glm::vec3 apply_barycentric(glm::vec3 barycentricCoordinates, glm::vec3 a, glm::vec3 b, glm::vec3 c);

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

void DiffuseIllumination::init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, VulkanCompute* vulkanCompute, VulkanRaytracing* vulkanRaytracing, GltfScene& scene, SceneDescriptors& sceneDescriptors)
{
	// Create lightmap framebuffer and its sampler
	{
		VkExtent3D lightmapImageExtent3D = {
			_lightmapExtent.width,
			_lightmapExtent.height,
			1
		};

		_lightmapColorImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
	}

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
			_giIndirectLightImage = vkutils::create_image(&engineData, engineData.color32Format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			_giIndirectLightImageView = vkutils::create_image_view(&engineData, _giIndirectLightImage, engineData.color32Format);
			vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
				vkutils::image_barrier(cmd, _giIndirectLightImage._image, VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_GENERAL, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }, 
					0, VK_ACCESS_SHADER_WRITE_BIT);
			});
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

	{
		{
			_dilatedGiIndirectLightImage = vkutils::create_image(&engineData, engineData.color32Format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			_dilatedGiIndirectLightImageView = vkutils::create_image_view(&engineData, _giIndirectLightImage, engineData.color32Format);
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
	//same but for realtime
	_probeRaycastResultBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * _config.probeCount * _config.rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	//ProbeBasisFunctions buffer (GPU ONLY)
	auto probeBasisBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probeRaycastBasisFunctions, sizeof(glm::vec4) * (_config.rayCount * _config.basisFunctionCount / 4 + 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_probe_projection Temp buffer (GPU ONLY)
	//auto probeRelightTempBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * _config.probeCount * _config.rayCount * _config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_probe_projection output buffer (GPU ONLY)
	_probeRelightOutputBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * _config.probeCount * _config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

	//Create compute instances
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, probeRaycastResultBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, probeBasisBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelight, ComputeBufferType::STORAGE, _probeRelightOutputBuffer);
	_vulkanCompute->add_texture_binding(_probeRelight, ComputeBufferType::TEXTURE_SAMPLED, engineData.linearSampler, _lightmapColorImageView);

	_vulkanCompute->build(_probeRelight, _descriptorPool, "../../shaders/gi_probe_projection.comp.spv");

	_vulkanCompute->add_buffer_binding(_probeRelightRealtime, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelightRealtime, ComputeBufferType::STORAGE, _probeRaycastResultBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelightRealtime, ComputeBufferType::STORAGE, probeBasisBuffer);
	_vulkanCompute->add_buffer_binding(_probeRelightRealtime, ComputeBufferType::STORAGE, _probeRelightOutputBuffer);
	_vulkanCompute->build(_probeRelightRealtime, _descriptorPool, "../../shaders/gi_probe_projection_realtime.comp.spv");

	//Cluster projection matrices (GPU ONLY)
	auto clusterProjectionMatricesBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterProjectionMatrices, _precalculationLoadData->projectionMatricesSize * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
	//gi_cluster_projection output buffer (GPU ONLY)
	_clusterProjectionOutputBuffer = vkutils::create_buffer(_allocator, _precalculationLoadData->totalSvdCoeffCount * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterReceiverInfos = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterReceiverInfos, _config.clusterCount * sizeof(ClusterReceiverInfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterProbes = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterProbes, (_precalculationLoadData->totalProbesPerCluster / 4 + 1) * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, _probeRelightOutputBuffer);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, clusterProjectionMatricesBuffer);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, clusterReceiverInfos);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, clusterProbes);
	_vulkanCompute->add_buffer_binding(_clusterProjection, ComputeBufferType::STORAGE, _clusterProjectionOutputBuffer);
	_vulkanCompute->build(_clusterProjection, _descriptorPool, "../../shaders/gi_cluster_projection.comp.spv");


	auto receiverReconstructionMatricesBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->receiverCoefficientMatrices, (_precalculationLoadData->reconstructionMatricesSize / 4 + 1) * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
	auto clusterReceiverUvs = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterReceiverUvs, _precalculationLoadData->totalClusterReceiverCount * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::UNIFORM, _configBuffer);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, _clusterProjectionOutputBuffer);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, receiverReconstructionMatricesBuffer);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, clusterReceiverInfos);
	_vulkanCompute->add_buffer_binding(_receiverReconstruction, ComputeBufferType::STORAGE, clusterReceiverUvs);
	_vulkanCompute->add_texture_binding(_receiverReconstruction, ComputeBufferType::TEXTURE_STORAGE, 0, _giIndirectLightImageView);
	_vulkanCompute->build(_receiverReconstruction, _descriptorPool, "../../shaders/gi_receiver_reconstruction.comp.spv");
}

void DiffuseIllumination::render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, Shadow& shadow, BRDF& brdfUtils, std::function<void(VkCommandBuffer cmd)>&& function, bool realtimeProbeRaycast, VkPipeline dilationPipeline, VkPipelineLayout dilationPipelineLayout)
{
	//GI - Probe relight
	if (!realtimeProbeRaycast) {
		// LIGHTMAP RENDERING
		{
			VkClearValue clearValue;
			clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(engineData.colorRenderPass, _lightmapExtent, _lightmapFramebuffer);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkutils::cmd_viewport_scissor(cmd, _lightmapExtent);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 0, 1, &sceneDescriptors.globalDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 1, 1, &sceneDescriptors.objectDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 2, 1, &sceneDescriptors.textureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 3, 1, &sceneDescriptors.materialDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 4, 1, &shadow._shadowMapTextureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 5, 1, &_giIndirectLightTextureDescriptor, 0, nullptr);

			function(cmd);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}

		{
			int groupcount = ((_precalculationResult->probes.size() * _precalculationInfo->raysPerProbe) / 64) + 1;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelight.pipelineLayout, 0, 1, &_probeRelight.descriptorSet, 0, nullptr);

			vkCmdDispatch(cmd, groupcount, 1, 1);
		}
	}
	else {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _probeRTPipeline.pipeline);

		std::vector<VkDescriptorSet> descSets{ _probeRTDescriptorSet, sceneDescriptors.globalDescriptor, sceneDescriptors.objectDescriptor, _giIndirectLightTextureDescriptor, sceneDescriptors.textureDescriptor, sceneDescriptors.materialDescriptor, shadow._shadowMapTextureDescriptor, _giIndirectLightTextureDescriptor,  brdfUtils._brdfLutTextureDescriptor };

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _probeRTPipeline.pipelineLayout, 0,
			(uint32_t)descSets.size(), descSets.data(), 0, nullptr);

		vkCmdTraceRaysKHR(cmd, &_probeRTPipeline.rgenRegion, &_probeRTPipeline.missRegion, &_probeRTPipeline.hitRegion, &_probeRTPipeline.callRegion, _config.rayCount, _config.probeCount, 1);

		{
			VkBufferMemoryBarrier barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.offset = 0;
			barrier.size = VK_WHOLE_SIZE;
			barrier.buffer = _probeRaycastResultBuffer._buffer;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		{
			int groupcount = ((_precalculationResult->probes.size() * _precalculationInfo->raysPerProbe) / 64) + 1;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelightRealtime.pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _probeRelightRealtime.pipelineLayout, 0, 1, &_probeRelightRealtime.descriptorSet, 0, nullptr);

			vkCmdDispatch(cmd, groupcount, 1, 1);
		}
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

		int groupcount = ((_precalculationLoadData->aabbClusterCount * _precalculationInfo->clusterCoefficientCount) / 64) + 1;
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
		int groupcount = ((_precalculationLoadData->aabbClusterCount * _precalculationInfo->maxReceiversInCluster) / 64) + 1;
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

void DiffuseIllumination::render_ground_truth(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, Shadow& shadow, BRDF& brdfUtils, VkPipeline dilationPipeline, VkPipelineLayout dilationPipelineLayout)
{
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _gtDiffuseRTPipeline.pipeline);

	std::vector<VkDescriptorSet> descSets{ _gtDiffuseRTDescriptorSet, sceneDescriptors.globalDescriptor, sceneDescriptors.objectDescriptor, _dilatedGiIndirectLightTextureDescriptor, sceneDescriptors.textureDescriptor, sceneDescriptors.materialDescriptor, shadow._shadowMapTextureDescriptor, _giIndirectLightTextureDescriptor,  brdfUtils._brdfLutTextureDescriptor };

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _gtDiffuseRTPipeline.pipelineLayout, 0,
		(uint32_t)descSets.size(), descSets.data(), 0, nullptr);

	vkCmdTraceRaysKHR(cmd, &_gtDiffuseRTPipeline.rgenRegion, &_gtDiffuseRTPipeline.missRegion, &_gtDiffuseRTPipeline.hitRegion, &_gtDiffuseRTPipeline.callRegion, _gpuReceiverCount, 1, 1);

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
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
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

void DiffuseIllumination::build_lightmap_pipeline(EngineData& engineData)
{
	VkShaderModule lightmapVertShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/lightmap.vert.spv", &lightmapVertShader))
	{
		assert("Lightmap Vertex Shader Loading Issue");
	}

	VkShaderModule lightmapFragShader;
	if (!vkutils::load_shader_module(engineData.device, "../../shaders/lightmap.frag.spv", &lightmapFragShader))
	{
		assert("Lightmap Vertex Shader Loading Issue");
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
	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;

	//we don't use multisampling, so just run the default one
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

	//a single blend attachment with no blending and writing to RGBA

	auto blendState = vkinit::color_blend_attachment_state();
	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &blendState);

	//build the mesh pipeline
	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	//add the other shaders
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, lightmapVertShader));

	//make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, lightmapFragShader));

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(false, false, VK_COMPARE_OP_LESS_OR_EQUAL);

	pipelineBuilder._pipelineLayout = _lightmapPipelineLayout;


	VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterStateCI{};
	conservativeRasterStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
	conservativeRasterStateCI.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
	conservativeRasterStateCI.extraPrimitiveOverestimationSize = 1.0;
	pipelineBuilder._rasterizer.pNext = &conservativeRasterStateCI;

	//build the mesh triangle pipeline
	_lightmapPipeline = pipelineBuilder.build_pipeline(engineData.device, engineData.colorRenderPass);

	vkutils::setObjectName(engineData.device, _lightmapPipeline, "LightmapPipeline");
	//destroy all shader modules, outside of the queue
	vkDestroyShaderModule(engineData.device, lightmapVertShader, nullptr);
	vkDestroyShaderModule(engineData.device, lightmapFragShader, nullptr);
}

void DiffuseIllumination::build_proberaycast_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer)
{
	VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);
	VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
	VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);
	VkDescriptorSetLayoutBinding probeLocationsBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3);
	VkDescriptorSetLayoutBinding outColorBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4);
	VkDescriptorSetLayoutBinding bindings[5] = { tlasBind, sceneDescBind, meshInfoBind, probeLocationsBind, outColorBind };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 5);
	vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_probeRTDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_probeRTDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engineData.device, &allocateInfo, &_probeRTDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &_vulkanRaytracing->tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = _probeRTDescriptorSet;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	writes.emplace_back(accelerationStructureWrite);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _probeRTDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _probeRTDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));

	_probeLocationsBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probes.data(), sizeof(glm::vec4) * _config.probeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _probeRTDescriptorSet, &_probeLocationsBuffer._descriptorBufferInfo, 3));

	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _probeRTDescriptorSet, &_probeRaycastResultBuffer._descriptorBufferInfo, 4));
	
	vkUpdateDescriptorSets(engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void DiffuseIllumination::build_realtime_proberaycast_pipeline(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	VkDescriptorSetLayout setLayouts[] = { _probeRTDescriptorSetLayout, sceneDescriptors.globalSetLayout, sceneDescriptors.objectSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.textureSetLayout, sceneDescriptors.materialSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout };
	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 9);

	VkSpecializationMapEntry specializationMapEntry = { 0, 0, sizeof(uint32_t) };
	uint32_t maxRecursion = 1;
	VkSpecializationInfo specializationInfo = { 1, &specializationMapEntry, sizeof(maxRecursion), &maxRecursion };

	_vulkanRaytracing->create_new_pipeline(_probeRTPipeline, pipeline_layout_info,
		"../../shaders/proberaycast_realtime.rgen.spv",
		"../../shaders/reflections_rt.rmiss.spv",
		"../../shaders/reflections_rt.rchit.spv"
		, 2, nullptr, nullptr, &specializationInfo);

	vkutils::setObjectName(engineData.device, _probeRTPipeline.pipeline, "ProbeRTPipeline");
	vkutils::setObjectName(engineData.device, _probeRTPipeline.pipelineLayout, "ProbeRTPipelineLayout");
}

void DiffuseIllumination::build_radiance_coefficients_descriptor(EngineData& engineData)
{
	VkDescriptorSetLayoutBinding probeLocationsBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutBinding outColorBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	VkDescriptorSetLayoutBinding bindings[2] = { probeLocationsBind, outColorBind };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 2);
	vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_radianceCoefficientsDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_radianceCoefficientsDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engineData.device, &allocateInfo, &_radianceCoefficientsDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _radianceCoefficientsDescriptorSet, &_probeLocationsBuffer._descriptorBufferInfo, 0));
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _radianceCoefficientsDescriptorSet, &_probeRelightOutputBuffer._descriptorBufferInfo, 1));

	vkUpdateDescriptorSets(engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

}

void DiffuseIllumination::build_groundtruth_gi_raycast_descriptors(EngineData& engineData, GltfScene& scene, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer)
{
	VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);
	VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
	VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);
	VkDescriptorSetLayoutBinding receiversBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3);
	VkDescriptorSetLayoutBinding outColorBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4);
	VkDescriptorSetLayoutBinding bindings[5] = { tlasBind, sceneDescBind, meshInfoBind, receiversBind, outColorBind };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 5);
	vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_gtDiffuseRTDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_gtDiffuseRTDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engineData.device, &allocateInfo, &_gtDiffuseRTDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &_vulkanRaytracing->tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = _gtDiffuseRTDescriptorSet;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	writes.emplace_back(accelerationStructureWrite);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _gtDiffuseRTDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _gtDiffuseRTDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));

	{
		std::vector<GPUReceiverDataUV>* lm = new std::vector<GPUReceiverDataUV>[_precalculationInfo->lightmapResolution * _precalculationInfo->lightmapResolution];
		int receiverCount = 0;
		for (int nodeIndex = 0; nodeIndex < scene.nodes.size(); nodeIndex++) {
			auto& mesh = scene.prim_meshes[scene.nodes[nodeIndex].prim_mesh];
			for (int triangle = 0; triangle < mesh.idx_count; triangle += 3) {
				glm::vec3 worldVertices[3];
				glm::vec3 worldNormals[3];
				glm::vec2 texVertices[3];
				int minX = _precalculationInfo->lightmapResolution, minY = _precalculationInfo->lightmapResolution;
				int maxX = 0, maxY = 0;
				for (int i = 0; i < 3; i++) {
					int vertexIndex = mesh.vtx_offset + scene.indices[mesh.first_idx + triangle + i];

					glm::vec4 vertex = scene.nodes[nodeIndex].world_matrix * glm::vec4(scene.positions[vertexIndex], 1.0);
					worldVertices[i] = glm::vec3(vertex / vertex.w);


					worldNormals[i] = glm::mat3(glm::transpose(glm::inverse(scene.nodes[nodeIndex].world_matrix))) * scene.normals[vertexIndex];

					texVertices[i] = scene.lightmapUVs[vertexIndex] * glm::vec2(_precalculationInfo->lightmapResolution / (float)scene.lightmap_width, _precalculationInfo->lightmapResolution / (float)scene.lightmap_height);

					if (texVertices[i].x < minX) {
						minX = texVertices[i].x;
					}
					if (texVertices[i].x > maxX) {
						maxX = std::ceil(texVertices[i].x);
					}
					if (texVertices[i].y < minY) {
						minY = texVertices[i].y;
					}
					if (texVertices[i].y > maxY) {
						maxY = std::ceil(texVertices[i].y);
					}
				}

				for (int j = minY; j <= maxY; j++) {
					for (int i = minX; i <= maxX; i++) {
						int maxSample = TEXEL_SAMPLES;
						for (int sample = 0; sample < maxSample * maxSample; sample++) {
							glm::vec2 pixelMiddle;
							if (maxSample > 1) {
								pixelMiddle = { i + (sample / maxSample) / ((float)(maxSample - 1)), j + (sample % maxSample) / ((float)(maxSample - 1)) };
								//printf("%f , %f\n", pixelMiddle.x, pixelMiddle.y);
							}
							else {
								pixelMiddle = { i + 0.5, j + 0.5 };
							}
							glm::vec3 barycentric = calculate_barycentric(pixelMiddle,
								texVertices[0], texVertices[1], texVertices[2]);
							if (barycentric.x >= 0 && barycentric.y >= 0 && barycentric.z >= 0) {
								GPUReceiverDataUV receiverData = {};
								receiverData.pos = apply_barycentric(barycentric, worldVertices[0], worldVertices[1], worldVertices[2]);
								receiverData.normal = apply_barycentric(barycentric, worldNormals[0], worldNormals[1], worldNormals[2]);
								receiverData.uvPad = { i, j, 0, 0 };
								bool exists = false;
								for (int checker = 0; checker < lm[i + j * _precalculationInfo->lightmapResolution].size(); checker++) {
									if (lm[i + j * _precalculationInfo->lightmapResolution][checker].pos == receiverData.pos) {
										exists = true;
										break;
									}
								}
								if (!exists) {
									lm[i + j * _precalculationInfo->lightmapResolution].push_back(receiverData);
								}
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < _precalculationInfo->lightmapResolution * _precalculationInfo->lightmapResolution; i++) {
			if (lm[i].size() > 0) {
				receiverCount++;
				int targetSize = (TEXEL_SAMPLES * TEXEL_SAMPLES) > lm[i].size() ? lm[i].size() : (TEXEL_SAMPLES * TEXEL_SAMPLES);
				for (int j = 0; j < targetSize; j++) {
					lm[i][j].uvPad.b = lm[i].size();
					receiverDataVector.push_back(lm[i][j]);
				}
				int remainingSize = TEXEL_SAMPLES * TEXEL_SAMPLES - targetSize;
				for (int j = 0; j < remainingSize; j++) {
					receiverDataVector.push_back(lm[i][0]);
				}
			}
		}

		delete[] lm;
		_gpuReceiverCount = receiverCount;
		_receiverBuffer = vkutils::create_upload_buffer(&engineData, receiverDataVector.data(), sizeof(GPUReceiverDataUV) * receiverDataVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _gtDiffuseRTDescriptorSet, &_receiverBuffer._descriptorBufferInfo, 3));
	}

	{
		VkDescriptorImageInfo storageImageBufferInfo;
		storageImageBufferInfo.sampler = engineData.linearSampler;
		storageImageBufferInfo.imageView = _giIndirectLightImageView;
		storageImageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		writes.emplace_back(vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _gtDiffuseRTDescriptorSet, &storageImageBufferInfo, 4, 1));
	}
	
	vkUpdateDescriptorSets(engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

}

void DiffuseIllumination::build_groundtruth_gi_raycast_pipeline(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	VkDescriptorSetLayout setLayouts[] = { _gtDiffuseRTDescriptorSetLayout, sceneDescriptors.globalSetLayout, sceneDescriptors.objectSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.textureSetLayout, sceneDescriptors.materialSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout, sceneDescriptors.singleImageSetLayout };
	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 9);

	_vulkanRaytracing->create_new_pipeline(_gtDiffuseRTPipeline, pipeline_layout_info,
		"../../shaders/diffusegi_groundtruth.rgen.spv",
		"../../shaders/reflections_rt.rmiss.spv",
		"../../shaders/reflections_rt.rchit.spv");

	vkutils::setObjectName(engineData.device, _gtDiffuseRTPipeline.pipeline, "gtDiffuseRTPipeline");
	vkutils::setObjectName(engineData.device, _gtDiffuseRTPipeline.pipelineLayout, "gtDiffuseRTPipelineLayout");
}

void DiffuseIllumination::rebuild_shaders(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	_vulkanCompute->rebuildPipeline(_probeRelight, "../../shaders/gi_probe_projection.comp.spv");
	_vulkanCompute->rebuildPipeline(_probeRelightRealtime, "../../shaders/gi_probe_projection_realtime.comp.spv");
	_vulkanCompute->rebuildPipeline(_clusterProjection, "../../shaders/gi_cluster_projection.comp.spv");
	_vulkanCompute->rebuildPipeline(_receiverReconstruction, "../../shaders/gi_receiver_reconstruction.comp.spv");

	vkDestroyPipeline(engineData.device, _lightmapPipeline, nullptr);
	build_lightmap_pipeline(engineData);

	_vulkanRaytracing->destroy_raytracing_pipeline(_probeRTPipeline);
	build_realtime_proberaycast_pipeline(engineData, sceneDescriptors);

	_vulkanRaytracing->destroy_raytracing_pipeline(_gtDiffuseRTPipeline);
	build_groundtruth_gi_raycast_pipeline(engineData, sceneDescriptors);

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
	int probeCount = _precalculationResult->clusterReceiverInfos[specificCluster].probeCount;
	int probeOffset = _precalculationResult->clusterReceiverInfos[specificCluster].probeOffset;

	auto receiverPos = _precalculationResult->aabbReceivers[receiverOffset + specificReceiver].position * sceneScale;
	auto receiverNormal = _precalculationResult->aabbReceivers[receiverOffset + specificReceiver].normal;
	debugRenderer.draw_point(receiverPos, { 1, 0, 0 });

	debugRenderer.draw_line(receiverPos, receiverPos + receiverNormal * 50.0f, { 0, 1, 0 });

	for (int abc = 0; abc < specificReceiverRaySampleCount; abc++) {
		uint random_state = (specificReceiver * 1973 + 9277 * abc + specificReceiver * 26699) | 1;
		vec3 direction = normalize(receiverNormal + random_unit_vector(random_state));

		debugRenderer.draw_line(receiverPos, receiverPos + direction * 100.0f, { 0, 1, 1 });
	}
	
	for (int probe = 0; probe < probeCount; probe++) {
		int i = _precalculationResult->clusterProbes[probeOffset + probe];
		if (_precalculationResult->receiverProbeWeightData[(receiverOffset + specificReceiver) * _precalculationLoadData->maxProbesPerCluster + probe] > 0.000001) {
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

void DiffuseIllumination::cleanup(EngineData& engineData)
{
	vkDestroyPipeline(engineData.device, _lightmapPipeline, nullptr);
	vkDestroyPipelineLayout(engineData.device, _lightmapPipelineLayout, nullptr);
}