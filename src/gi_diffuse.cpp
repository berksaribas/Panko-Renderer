#include "gi_diffuse.h"
#include <vk_initializers.h>
#include <random>
#include <vk_utils.h>
#include <vk_pipeline.h>
#include <vk_rendergraph.h>
#include <vk_debug_renderer.h>
#include <gi_shadow.h>
#include <gi_brdf.h>
#include "gltf_scene.hpp"

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

void DiffuseIllumination::init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, GltfScene& scene)
{
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
		_lightmapExtent.width,
		_lightmapExtent.height,
		1
	};

	VkExtent3D giLightmapImageExtent3D = {
		_giLightmapExtent.width,
		_giLightmapExtent.height,
		1
	};

	//IMAGES
	_lightmapColorImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
	_lightmapColorImageBinding = engineData.renderGraph->register_image_view(&_lightmapColorImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = 1
	}, "DiffuseLightmapImage");

	_giIndirectLightImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, giLightmapImageExtent3D);
	_giIndirectLightImageBinding = engineData.renderGraph->register_image_view(&_giIndirectLightImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = 1
	}, "DiffuseIndirectImage");

	_dilatedGiIndirectLightImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, giLightmapImageExtent3D);
	_dilatedGiIndirectLightImageBinding = engineData.renderGraph->register_image_view(&_dilatedGiIndirectLightImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = 1
	}, "DiffuseDilatedIndirectImage");

	//COMMON
	_configBuffer = vkutils::create_upload_buffer(&engineData, &_config, sizeof(GIConfig), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_configBufferBinding = engineData.renderGraph->register_uniform_buffer(&_configBuffer, "DiffuseConfigBuffer");

	//Probe relighting buffers
	_probeRaycastResultOfflineBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probeRaycastResult, sizeof(GPUProbeRaycastResult) * _config.probeCount * _config.rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_probeRaycastResultOfflineBufferBinding = engineData.renderGraph->register_storage_buffer(&_probeRaycastResultOfflineBuffer, "DiffuseProbeRaycatResultOfflineBuffer");

	_probeRaycastResultOnlineBuffer = vkutils::create_buffer(engineData.allocator, sizeof(glm::vec4) * _config.probeCount * _config.rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_probeRaycastResultOnlineBufferBinding = engineData.renderGraph->register_storage_buffer(&_probeRaycastResultOnlineBuffer, "DiffuseProbeRaycastResultOnlineBuffer");

	_probeBasisBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probeRaycastBasisFunctions, sizeof(glm::vec4) * (_config.rayCount * _config.basisFunctionCount / 4 + 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_probeBasisBufferBinding = engineData.renderGraph->register_storage_buffer(&_probeBasisBuffer, "DiffuseProbeBasisBuffer");

	_probeRelightOutputBuffer = vkutils::create_buffer(engineData.allocator, sizeof(glm::vec4) * _config.probeCount * _config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
	_probeRelightOutputBufferBinding = engineData.renderGraph->register_storage_buffer(&_probeRelightOutputBuffer, "DiffuseProbeRelightOutputBuffer");

	//cluster projetion
	_clusterProjectionMatricesBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterProjectionMatrices, _precalculationLoadData->projectionMatricesSize * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
	_clusterProjectionMatricesBufferBinding = engineData.renderGraph->register_storage_buffer(&_clusterProjectionMatricesBuffer, "DiffuseClusterProjectionMatricesBuffer");

	_clusterProjectionOutputBuffer = vkutils::create_buffer(engineData.allocator, _precalculationLoadData->totalSvdCoeffCount * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_clusterProjectionOutputBufferBinding = engineData.renderGraph->register_storage_buffer(&_clusterProjectionOutputBuffer, "DiffuseClusterProjectionOutputBuffer");

	_clusterReceiverInfos = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterReceiverInfos, _config.clusterCount * sizeof(ClusterReceiverInfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_clusterReceiverInfosBinding = engineData.renderGraph->register_storage_buffer(&_clusterReceiverInfos, "DiffuseClusterReceiverInfos");

	_clusterProbes = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterProbes, (_precalculationLoadData->totalProbesPerCluster / 4 + 1) * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_clusterProbesBinding = engineData.renderGraph->register_storage_buffer(&_clusterProbes, "DiffuseClusterProbes");
	
	//receiver reconstruction
	_receiverReconstructionMatricesBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->receiverCoefficientMatrices, (_precalculationLoadData->reconstructionMatricesSize / 4 + 1) * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
	_receiverReconstructionMatricesBufferBinding = engineData.renderGraph->register_storage_buffer(&_receiverReconstructionMatricesBuffer, "DiffuseReceiverReconstructionMatricesBuffer");

	_clusterReceiverUvs = vkutils::create_upload_buffer(&engineData, _precalculationResult->clusterReceiverUvs, _precalculationLoadData->totalClusterReceiverCount * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_clusterReceiverUvsBinding = engineData.renderGraph->register_storage_buffer(&_clusterReceiverUvs, "DiffuseClusterReceiverUvs");

	_probeLocationsBuffer = vkutils::create_upload_buffer(&engineData, _precalculationResult->probes.data(), sizeof(glm::vec4) * _config.probeCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_probeLocationsBufferBinding = engineData.renderGraph->register_storage_buffer(&_probeLocationsBuffer, "DiffuseProbeLocationsBuffer");

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
		_receiverBufferBinding = engineData.renderGraph->register_storage_buffer(&_receiverBuffer, "ReceiverBuffer");
	}
}

VkSpecializationMapEntry specializationMapEntry = { 0, 0, sizeof(uint32_t) };
uint32_t maxRecursion = 1;

void DiffuseIllumination::render(VkCommandBuffer cmd, EngineData& engineData, SceneData& sceneData, Shadow& shadow, BRDF& brdfUtils, std::function<void(VkCommandBuffer cmd)>&& function, bool realtimeProbeRaycast)
{
	//GI - Probe relight
	if (!realtimeProbeRaycast) {
		VkClearValue clearValue;
		clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

		engineData.renderGraph->add_render_pass({
			.name = "DiffuseLightmapPass",
			.pipelineType = Vrg::PipelineType::RASTER_TYPE,
			.rasterPipeline = {
				.vertexShader = "../../shaders/diffuse_probes/lightmap.vert.spv",
				.fragmentShader = "../../shaders/diffuse_probes/lightmap.frag.spv",
				.size = _lightmapExtent,
				.depthState = { false, false, VK_COMPARE_OP_NEVER },
				.cullMode = Vrg::CullMode::NONE,
				.enableConservativeRasterization = true,
				.blendAttachmentStates = {
					vkinit::color_blend_attachment_state(),
				},
				.vertexBuffers = {
					sceneData.vertexBufferBinding,
					sceneData.normalBufferBinding,
					sceneData.texBufferBinding,
					sceneData.lightmapTexBufferBinding
				},
				.indexBuffer = sceneData.indexBufferBinding,
				.colorOutputs = {
					{_lightmapColorImageBinding, clearValue, false},
				},
			},
			.reads = {
				{0, sceneData.cameraBufferBinding},
				{0, shadow._shadowMapDataBinding},
				{1, sceneData.objectBufferBinding},
				{3, sceneData.materialBufferBinding},
				{4, shadow._shadowMapColorImageBinding},
				{5, _dilatedGiIndirectLightImageBinding},
			},
			.extraDescriptorSets = {
				{2, sceneData.textureDescriptor, sceneData.textureSetLayout}
			},
			.execute = function
		});

		int groupcount = ((_precalculationResult->probes.size() * _precalculationInfo->raysPerProbe) / 64) + 1;

		engineData.renderGraph->add_render_pass({
			.name = "DiffuseProbeRelight(Offline)",
			.pipelineType = Vrg::PipelineType::COMPUTE_TYPE,
			.computePipeline = {
				.shader = "../../shaders/diffuse_probes/gi_probe_projection.comp.spv",
				.dimX = (uint32_t) groupcount,
				.dimY = 1,
				.dimZ = 1
			},
			.writes = {
				{0, _probeRelightOutputBufferBinding}
			},
			.reads = {
				{0, _configBufferBinding},
				{0, _probeRaycastResultOfflineBufferBinding},
				{0, _probeBasisBufferBinding},
				{0, _lightmapColorImageBinding}
			}
		});
	}
	else {
		{
			VkSpecializationInfo specializationInfo = { 1, &specializationMapEntry, sizeof(maxRecursion), &maxRecursion };

			engineData.renderGraph->add_render_pass({
				.name = "ProbeRelightPathTracePass",
				.pipelineType = Vrg::PipelineType::RAYTRACING_TYPE,
				.raytracingPipeline = {
					.rgenShader = "../../shaders/diffuse_probes/proberaycast_realtime.rgen.spv",
					.missShader = "../../shaders/reflections/reflections_rt.rmiss.spv",
					.hitShader = "../../shaders/reflections/reflections_rt.rchit.spv",
					.recursionDepth = 2,
					.hitSpecialization = specializationInfo,
					.width = (uint32_t) _config.rayCount,
					.height = (uint32_t) _config.probeCount,
				},
				.writes = {
					{1, _probeRaycastResultOnlineBufferBinding} //gi image
				},
				.reads = {
					{1, _probeLocationsBufferBinding}, //receivers bind
					//same for all
					{2, sceneData.cameraBufferBinding},
					{2, shadow._shadowMapDataBinding},
					{3, sceneData.objectBufferBinding},
					{5, sceneData.materialBufferBinding},
					{6, shadow._shadowMapColorImageBinding},
					{7, _dilatedGiIndirectLightImageBinding},
					{8, brdfUtils.brdfLutImageBinding},
				},
				.extraDescriptorSets = {
					{0, sceneData.raytracingDescriptor, sceneData.raytracingSetLayout},
					{4, sceneData.textureDescriptor, sceneData.textureSetLayout}
				}
			});
		}

		{
			int groupcount = ((_precalculationResult->probes.size() * _precalculationInfo->raysPerProbe) / 64) + 1;

			engineData.renderGraph->add_render_pass({
				.name = "DiffuseProbeRelight(Online)",
				.pipelineType = Vrg::PipelineType::COMPUTE_TYPE,
				.computePipeline = {
					.shader = "../../shaders/diffuse_probes/gi_probe_projection_realtime.comp.spv",
					.dimX = (uint32_t)groupcount,
					.dimY = 1,
					.dimZ = 1
				},
				.writes = {
					{0, _probeRelightOutputBufferBinding}
				},
				.reads = {
					{0, _configBufferBinding},
					{0, _probeRaycastResultOnlineBufferBinding},
					{0, _probeBasisBufferBinding}
				}
			});
		}
	}

	//GI - Cluster Projection
	{
		int groupcount = ((_precalculationLoadData->aabbClusterCount * _precalculationInfo->clusterCoefficientCount) / 64) + 1;
		engineData.renderGraph->add_render_pass({
			.name = "DiffuseClusterProjection",
			.pipelineType = Vrg::PipelineType::COMPUTE_TYPE,
			.computePipeline = {
				.shader = "../../shaders/diffuse_probes/gi_cluster_projection.comp.spv",
				.dimX = (uint32_t)groupcount,
				.dimY = 1,
				.dimZ = 1
			},
			.writes = {
				{0, _clusterProjectionOutputBufferBinding}
			},
			.reads = {
				{0, _configBufferBinding},
				{0, _probeRelightOutputBufferBinding},
				{0, _clusterProjectionMatricesBufferBinding},
				{0, _clusterReceiverInfosBinding},
				{0, _clusterProbesBinding}
			}
		});
	}

	//GI - Receiver Projection
	{
		int groupcount = ((_precalculationLoadData->aabbClusterCount * _precalculationInfo->maxReceiversInCluster) / 64) + 1;
		engineData.renderGraph->add_render_pass({
			.name = "DiffuseReceiverReconstruction",
			.pipelineType = Vrg::PipelineType::COMPUTE_TYPE,
			.computePipeline = {
				.shader = "../../shaders/diffuse_probes/gi_receiver_reconstruction.comp.spv",
				.dimX = (uint32_t)groupcount,
				.dimY = 1,
				.dimZ = 1
			},
			.writes = {
				{0, _giIndirectLightImageBinding}
			},
			.reads = {
				{0, _configBufferBinding},
				{0, _clusterProjectionOutputBufferBinding},
				{0, _receiverReconstructionMatricesBufferBinding},
				{0, _clusterReceiverInfosBinding},
				{0, _clusterReceiverUvsBinding}
			}
		});
	}

	// GI LIGHTMAP DILATION RENDERING
	{
		VkClearValue clearValue;
		clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

		engineData.renderGraph->add_render_pass({
			.name = "DilationPipeline",
			.pipelineType = Vrg::PipelineType::RASTER_TYPE,
			.rasterPipeline = {
				.vertexShader = "../../shaders/fullscreen.vert.spv",
				.fragmentShader = "../../shaders/dilate.frag.spv",
				.size = _giLightmapExtent,
				.depthState = { false, false, VK_COMPARE_OP_NEVER },
				.cullMode = Vrg::CullMode::NONE,
				.blendAttachmentStates = {
					vkinit::color_blend_attachment_state(),
				},
				.colorOutputs = {
					{_dilatedGiIndirectLightImageBinding, clearValue, false},
				},

			},
			.reads = {
				{0, _giIndirectLightImageBinding} //gi image
			},
			.execute = [&](VkCommandBuffer cmd) {
				vkCmdDraw(cmd, 3, 1, 0, 0);
			}
		});
	}
}

void DiffuseIllumination::render_ground_truth(VkCommandBuffer cmd, EngineData& engineData, SceneData& sceneData, Shadow& shadow, BRDF& brdfUtils)
{
	engineData.renderGraph->add_render_pass({
		.name = "DiffuseGroundTruthPathTracePass",
		.pipelineType = Vrg::PipelineType::RAYTRACING_TYPE,
		.raytracingPipeline = {
			.rgenShader = "../../shaders/diffuse_lightmap_groundtruth/diffusegi_groundtruth.rgen.spv",
			.missShader = "../../shaders/reflections/reflections_rt.rmiss.spv",
			.hitShader = "../../shaders/reflections/reflections_rt.rchit.spv",
			.width = _gpuReceiverCount
		},
		.writes = {
			{1, _giIndirectLightImageBinding} //gi image
		},
		.reads = {
			{1, _receiverBufferBinding}, //receivers bind
			//same for all
			{2, sceneData.cameraBufferBinding},
			{2, shadow._shadowMapDataBinding},
			{3, sceneData.objectBufferBinding},
			{5, sceneData.materialBufferBinding},
			{6, shadow._shadowMapColorImageBinding},
			{7, _dilatedGiIndirectLightImageBinding},
			{8, brdfUtils.brdfLutImageBinding},
		},
		.extraDescriptorSets = {
			{0, sceneData.raytracingDescriptor, sceneData.raytracingSetLayout},
			{4, sceneData.textureDescriptor, sceneData.textureSetLayout}
		}
	});

	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

	engineData.renderGraph->add_render_pass({
		.name = "DilationPipeline",
		.pipelineType = Vrg::PipelineType::RASTER_TYPE,
		.rasterPipeline = {
			.vertexShader = "../../shaders/fullscreen.vert.spv",
			.fragmentShader = "../../shaders/dilate.frag.spv",
			.size = _giLightmapExtent,
			.depthState = { false, false, VK_COMPARE_OP_NEVER },
			.cullMode = Vrg::CullMode::NONE,
			.blendAttachmentStates = {
				vkinit::color_blend_attachment_state(),
			},
			.colorOutputs = {
				{_dilatedGiIndirectLightImageBinding, clearValue, false},
			},

		},
		.reads = {
			{0, _giIndirectLightImageBinding} //gi image
		},
		.execute = [&](VkCommandBuffer cmd) {
			vkCmdDraw(cmd, 3, 1, 0, 0);
		}
	});
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