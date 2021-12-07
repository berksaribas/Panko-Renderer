﻿#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>

#include "vk_pipeline.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <glm/gtx/transform.hpp>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"
#include <precalculation.h>
#include <optick.h>
#include <vk_utils.h>
#include <vk_extensions.h>
#include <vk_debug_renderer.h>
#include <xatlas.h>
#include <random>

VulkanDebugRenderer vkDebugRenderer;
const int MAX_TEXTURES = 75; //TODO: Replace this
constexpr bool bUseValidationLayers = true;
const VkFormat colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

Precalculation precalculation;

AllocatedBuffer configBuffer;
GIConfig config = {};

PrecalculationInfo precalculationInfo = {};
PrecalculationLoadData precalculationLoadData = {};
PrecalculationResult precalculationResult = {};

ComputeInstance probeRelight = {};
ComputeInstance clusterProjection = {};
ComputeInstance receiverReconstruction = {};

AllocatedImage giInirectLightImage;
VkImageView giInirectLightImageView;
VkDescriptorSet giInirectLightTextureDescriptor;

AllocatedImage dilatedGiInirectLightImage;
VkImageView dilatedGiInirectLightImageView;
VkDescriptorSet dilatedGiInirectLightTextureDescriptor;
VkFramebuffer dilatedGiInirectLightFramebuffer;

AllocatedBuffer probeRelightOutputBuffer;
AllocatedBuffer clusterProjectionOutputBuffer;

VkExtent2D giLightmapExtent{ 0 , 0 };

bool enableGi = false;
bool showProbes = false;
bool showProbeRays = false;
bool showReceivers = false;
bool showSpecificReceiver = true;
int specificCluster = 145;
int specificReceiver = 135;
int speicificReceiverRaySampleCount = 10;
bool showSpecificProbeRays = false;
bool probesEnabled[300];

int renderMode = 0;

void VulkanEngine::init()
{
	OPTICK_START_CAPTURE();

	_shadowMapData.positiveExponent = 40;
	_shadowMapData.negativeExponent = 5;
	_shadowMapData.LightBleedingReduction = 0.999f;
	_shadowMapData.VSMBias = 0.01;
	_camData.lightPos = { 0.020, 2, 0.140, 0.0f };
	_camData.lightColor = { 1.f, 1.f, 1.f, 1.0f };
	camera.pos = { 0, 0, 7 };
	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

	_window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_windowExtent.width,
		_windowExtent.height,
		window_flags
	);

	init_vulkan();
	init_swapchain();
	init_default_renderpass();
	init_colordepth_renderpass();
	init_color_renderpass();
	init_framebuffers();
	init_commands();
	init_sync_structures();
	init_descriptors();
	init_pipelines();

	vulkanCompute.init(_device, _allocator, _computeQueue, _computeQueueFamily);
	vulkanRaytracing.init(_device, _gpuRaytracingProperties, _allocator, _computeQueue, _computeQueueFamily);
	vkDebugRenderer.init(_device, _allocator, _renderPass, _globalSetLayout);

	init_imgui();
	init_scene();

	init_gi();

	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::cleanup()
{
	if (_isInitialized) {

		//make sure the GPU has stopped doing its things
		vkDeviceWaitIdle(_device);

		_mainDeletionQueue.flush();

		vmaDestroyAllocator(_allocator);

		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		vkDestroyDevice(_device, nullptr);
		vkDestroyInstance(_instance, nullptr);

		SDL_DestroyWindow(_window);
	}
}

float random(vec2 st)
{
	float value = sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453123;
	return value - floor(value);
}

vec3 hemiSpherePointCos(float u, float v, vec3 normal)
{
	float a = 6.2831853 * v;
	u = 2.0 * u - 1.0;
	return normalize(normal + vec3(sqrt(1.0f - u * u) * vec2(cos(a), sin(a)), u));
}

void VulkanEngine::draw()
{
	OPTICK_EVENT();
	
	vulkanCompute.rebuildPipeline(probeRelight, "../../shaders/gi_probe_projection.comp.spv");
	vulkanCompute.rebuildPipeline(clusterProjection, "../../shaders/gi_cluster_projection.comp.spv");
	vulkanCompute.rebuildPipeline(receiverReconstruction, "../../shaders/gi_receiver_reconstruction.comp.spv");
	init_pipelines(true); //recompile shaders

	constexpr glm::vec3 UP = glm::vec3(0, 1, 0);
	constexpr glm::vec3 RIGHT = glm::vec3(1, 0, 0);
	constexpr glm::vec3 FORWARD = glm::vec3(0, 0, 1);
	auto res = glm::mat4{ 1 };
	res = glm::translate(res, camera.pos);
	res = glm::rotate(res, glm::radians(camera.rotation.y), UP);
	res = glm::rotate(res, glm::radians(camera.rotation.x), RIGHT);
	res = glm::rotate(res, glm::radians(camera.rotation.z), FORWARD);
	auto view = glm::inverse(res);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), ((float)_windowExtent.width) / _windowExtent.height, 0.1f, 1000.0f);
	projection[1][1] *= -1;

	//fill a GPU camera data struct
	_camData.proj = projection;
	_camData.view = view;
	_camData.viewproj = projection * view;

	_camData.lightmapInputSize = {(float) gltf_scene.lightmap_width, (float) gltf_scene.lightmap_height};
	_camData.lightmapTargetSize = {_lightmapExtent.width, _lightmapExtent.height};

	static float timer = 0.f;
	static bool pauseTimer = false;

	if (!pauseTimer) {
		timer += 1 / 144.f;
	}

	glm::vec3 lightInvDir = glm::vec3(_camData.lightPos); 
	float radius = gltf_scene.m_dimensions.max.x > gltf_scene.m_dimensions.max.y ? gltf_scene.m_dimensions.max.x : gltf_scene.m_dimensions.max.y;
	radius = radius > gltf_scene.m_dimensions.max.z ? radius : gltf_scene.m_dimensions.max.z;
	radius *= sqrt(2);
	glm::mat4 depthViewMatrix = glm::lookAt(gltf_scene.m_dimensions.center * sceneScale + lightInvDir * radius, gltf_scene.m_dimensions.center * sceneScale, glm::vec3(0, 1, 0));
	float maxX = gltf_scene.m_dimensions.min.x * sceneScale, maxY = gltf_scene.m_dimensions.min.y * sceneScale, maxZ = gltf_scene.m_dimensions.min.z * sceneScale;
	float minX = gltf_scene.m_dimensions.max.x * sceneScale, minY = gltf_scene.m_dimensions.max.y * sceneScale, minZ = gltf_scene.m_dimensions.max.z * sceneScale;

	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
				float xCoord = x == 0 ? gltf_scene.m_dimensions.min.x* sceneScale : gltf_scene.m_dimensions.max.x* sceneScale;
				float yCoord = y == 0 ? gltf_scene.m_dimensions.min.y* sceneScale : gltf_scene.m_dimensions.max.y* sceneScale;
				float zCoord = z == 0 ? gltf_scene.m_dimensions.min.z* sceneScale : gltf_scene.m_dimensions.max.z* sceneScale;
				auto tempCoords = depthViewMatrix * glm::vec4(xCoord, yCoord, zCoord, 1.0);
				tempCoords = tempCoords / tempCoords.w;
				if (tempCoords.x < minX) {
					minX = tempCoords.x;
				}
				if (tempCoords.x > maxX) {
					maxX = tempCoords.x;
				}

				if (tempCoords.y < minY) {
					minY = tempCoords.y;
				}
				if (tempCoords.y > maxY) {
					maxY = tempCoords.y;
				}

				if (tempCoords.z < minZ) {
					minZ = tempCoords.z;
				}
				if (tempCoords.z > maxZ) {
					maxZ = tempCoords.z;
				}
			}
		}
	}

	glm::mat4 depthProjectionMatrix = glm::ortho(minX, maxX, minY, maxY, -maxZ, -minZ);

	_shadowMapData.depthMVP = depthProjectionMatrix * depthViewMatrix;

	{
		//todo: imgui stuff
		static char buffer[128];
		ImGui::Begin("Engine Config");
		sprintf_s(buffer, "Positive Exponent");
		ImGui::DragFloat(buffer, &_shadowMapData.positiveExponent);
		sprintf_s(buffer, "Negative Exponent");
		ImGui::DragFloat(buffer, &_shadowMapData.negativeExponent);
		sprintf_s(buffer, "Light Bleeding Reduction");
		ImGui::DragFloat(buffer, &_shadowMapData.LightBleedingReduction);
		sprintf_s(buffer, "VSM Bias");
		ImGui::DragFloat(buffer, &_shadowMapData.VSMBias);

		glm::vec3 mins = { minX, minY, minZ };
		glm::vec3 maxs = { maxX, maxY, maxZ };

		sprintf_s(buffer, "Ligh Direction");
		if (ImGui::DragFloat3(buffer, &_camData.lightPos.x)) {
			config.frameNumber = 0;
		}
		sprintf_s(buffer, "Factor");
		ImGui::DragFloat(buffer, &radius);


		sprintf_s(buffer, "Pause Timer");
		ImGui::Checkbox(buffer, &pauseTimer);

		ImGui::Image(shadowMapTextureDescriptor, { 128, 128 });
		ImGui::Image(_lightmapTextureDescriptor, { (float)gltf_scene.lightmap_width,  (float)gltf_scene.lightmap_height });
		ImGui::Image(giInirectLightTextureDescriptor, { (float) gltf_scene.lightmap_width,  (float)gltf_scene.lightmap_height });
		ImGui::End();

		sprintf_s(buffer, "Show Probes");
		ImGui::Checkbox(buffer, &showProbes);

		if (showProbes) {
			sprintf_s(buffer, "Show Probe Rays");
			ImGui::Checkbox(buffer, &showProbeRays);
		}

		ImGui::NewLine();

		sprintf_s(buffer, "Show Receivers");
		ImGui::Checkbox(buffer, &showReceivers);

		ImGui::NewLine();

		sprintf_s(buffer, "Show Specific Receivers");
		ImGui::Checkbox(buffer, &showSpecificReceiver);

		if (showSpecificReceiver) {
			sprintf_s(buffer, "Cluster: ");
			ImGui::SliderInt(buffer, &specificCluster, 0, precalculationLoadData.aabbClusterCount - 1);
			sprintf_s(buffer, "Receiver: ");
			ImGui::SliderInt(buffer, &specificReceiver, 0, precalculationResult.clusterReceiverInfos[specificCluster].receiverCount - 1);
			
			sprintf_s(buffer, "Ray sample count: ");
			ImGui::DragInt(buffer, &speicificReceiverRaySampleCount);

			sprintf_s(buffer, "Show Selected Probe Rays");
			ImGui::Checkbox(buffer, &showSpecificProbeRays);

			{
				int receiverCount = precalculationResult.clusterReceiverInfos[specificCluster].receiverCount;
				int receiverOffset = precalculationResult.clusterReceiverInfos[specificCluster].receiverOffset;
				for (int i = 0; i < precalculationResult.probes.size(); i++) {
					if (precalculationResult.receiverProbeWeightData[(receiverOffset + specificReceiver) * precalculationResult.probes.size() + i] > 0) {
						sprintf_s(buffer, "Probe %d: %f", i, precalculationResult.receiverProbeWeightData[(receiverOffset + specificReceiver) * precalculationResult.probes.size() + i]);
						ImGui::Checkbox(buffer, &probesEnabled[i]);
					}
				}
			}
		}
		
		ImGui::Render();
	}

	
	if (showProbes) {
		for (int i = 0; i < precalculationResult.probes.size(); i++) {
			vkDebugRenderer.draw_point(glm::vec3(precalculationResult.probes[i]) * sceneScale, { 1, 0, 0 });
			if (showProbeRays) {
				for (int j = 0; j < precalculationInfo.raysPerProbe; j += 400) {
					auto& ray = precalculationResult.probeRaycastResult[precalculationInfo.raysPerProbe * i + j];
					if (ray.objectId != -1) {
						vkDebugRenderer.draw_line(glm::vec3(precalculationResult.probes[i]) * sceneScale,
							glm::vec3(ray.worldPos) * sceneScale,
							{ 0, 0, 1 });

						vkDebugRenderer.draw_point(glm::vec3(ray.worldPos) * sceneScale, { 0, 0, 1 });
					}
					else {
						vkDebugRenderer.draw_line(glm::vec3(precalculationResult.probes[i]) * sceneScale,
							glm::vec3(precalculationResult.probes[i]) * sceneScale + glm::vec3(ray.direction) * 10.f,
							{ 0, 0, 1 });

					}
				}
			}
		}
	}
	
	if (showReceivers) {
		std::random_device dev;
		std::mt19937 rng(dev());
		rng.seed(0);
		std::uniform_real_distribution<> dist(0, 1);
		
		for (int i = 0; i < precalculationLoadData.aabbClusterCount; i += 1) {
			glm::vec3 color = { dist(rng), dist(rng) , dist(rng) };
			int receiverCount = precalculationResult.clusterReceiverInfos[i].receiverCount;
			int receiverOffset = precalculationResult.clusterReceiverInfos[i].receiverOffset;

			for (int j = receiverOffset; j < receiverOffset + receiverCount; j++) {
				vkDebugRenderer.draw_point(precalculationResult.aabbReceivers[j].position * sceneScale, color);
				//vkDebugRenderer.draw_line(precalculation._aabbClusters[i].receivers[j].position * sceneScale, (precalculation._aabbClusters[i].receivers[j].position + precalculation._aabbClusters[i].receivers[j].normal * 2.f) * sceneScale, color);
			}
		}
	}

	if (showSpecificReceiver) {
		std::random_device dev;
		std::mt19937 rng(dev());
		rng.seed(0);
		std::uniform_real_distribution<> dist(0, 1);

		int receiverCount = precalculationResult.clusterReceiverInfos[specificCluster].receiverCount;
		int receiverOffset = precalculationResult.clusterReceiverInfos[specificCluster].receiverOffset;

		auto receiverPos = precalculationResult.aabbReceivers[receiverOffset + specificReceiver].position * sceneScale;
		auto receiverNormal = precalculationResult.aabbReceivers[receiverOffset + specificReceiver].normal;
		vkDebugRenderer.draw_point(receiverPos, { 1, 0, 0 });
		
		vkDebugRenderer.draw_line(receiverPos, receiverPos + receiverNormal * 50.0f, { 0, 1, 0 });
		
		for (int abc = 0; abc < speicificReceiverRaySampleCount; abc++) {
			float _u = random(vec2(specificReceiver, abc * 2));
			float _v = random(vec2(specificReceiver, abc * 2 + 1));

			vec3 direction = hemiSpherePointCos(_u, _v, receiverNormal);

			vkDebugRenderer.draw_line(receiverPos, receiverPos + direction * 100.0f, { 0, 1, 1 });
		}


		for (int i = 0; i < precalculationResult.probes.size(); i++) {
			if (precalculationResult.receiverProbeWeightData[(receiverOffset + specificReceiver) * precalculationResult.probes.size() + i] > 0.000001) {
				if (probesEnabled[i]) {
					vkDebugRenderer.draw_point(glm::vec3(precalculationResult.probes[i]) * sceneScale, { 1, 0, 1 });

					if (showSpecificProbeRays) {
						for (int j = 0; j < precalculationInfo.raysPerProbe; j += 1) {
							auto& ray = precalculationResult.probeRaycastResult[precalculationInfo.raysPerProbe * i + j];
							if (ray.objectId != -1) {
								vkDebugRenderer.draw_line(glm::vec3(precalculationResult.probes[i]) * sceneScale,
									glm::vec3(ray.worldPos) * sceneScale,
									{ 0, 0, 1 });

								vkDebugRenderer.draw_point(glm::vec3(ray.worldPos) * sceneScale, { 0, 0, 1 });
							}
							else {
								vkDebugRenderer.draw_line(glm::vec3(precalculationResult.probes[i]) * sceneScale,
									glm::vec3(precalculationResult.probes[i]) * sceneScale + glm::vec3(ray.direction) * 1000.f,
									{ 0, 0, 1 });

							}
						}
					}
				}
			}
		}
	}

	vkutils::cpu_to_gpu(_allocator, get_current_frame().cameraBuffer, &_camData, sizeof(GPUCameraData));

	vkutils::cpu_to_gpu(_allocator, get_current_frame().shadowMapDataBuffer, &_shadowMapData, sizeof(GPUShadowMapData));

	vkutils::cpu_to_gpu(_allocator, configBuffer, &config, sizeof(GIConfig));
	config.frameNumber++;

	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);
	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

	for (int i = 0; i < gltf_scene.nodes.size(); i++)
	{
		auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
		glm::mat4 scale = glm::mat4{ 1 };
		scale = glm::scale(scale, { sceneScale, sceneScale, sceneScale });
		objectSSBO[i].model = scale * gltf_scene.nodes[i].world_matrix;
		objectSSBO[i].material_id = mesh.material_idx;
	}
	vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);

	//wait until the gpu has finished rendering the last frame. Timeout of 1 second
	VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

	//now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

	//request image from the swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._presentSemaphore, nullptr, &swapchainImageIndex));

	//naming it cmd for shorter writing
	VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
	{
		
		// SHADOW MAP RENDERING
		{
			VkClearValue clearValue;
			clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkClearValue depthClear;
			depthClear.depthStencil = { 1.0f, 0 };
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_colorDepthRenderPass, _shadowMapExtent, _shadowMapFramebuffer);

			rpInfo.clearValueCount = 2;
			VkClearValue clearValues[] = { clearValue, depthClear };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
	
			cmd_viewport_scissor(cmd, _shadowMapExtent);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipelineLayout, 0, 1, &get_current_frame().shadowMapDataDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);

			draw_objects(cmd);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipeline);
			vkCmdEndRenderPass(cmd);
		}
		
		// LIGHTMAP RENDERING
		{
			VkClearValue clearValue;
			clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_colorRenderPass, _lightmapExtent, _lightmapFramebuffer);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			cmd_viewport_scissor(cmd, _lightmapExtent);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 0, 1, &get_current_frame().globalDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 2, 1, &textureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 3, 1, &materialDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 4, 1, &shadowMapTextureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _lightmapPipelineLayout, 5, 1, &dilatedGiInirectLightTextureDescriptor, 0, nullptr);

			draw_objects(cmd);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}

		{
			// i need an image barrier here?
			//probably i dont since that's done via subpass dependencies
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 0, nullptr, 0, nullptr, 0, nullptr);
		}

		//GI - Probe relight
		{
			int groupcount = ((precalculationResult.probes.size() * precalculationInfo.raysPerProbe) / 256) + 1;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probeRelight.pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probeRelight.pipelineLayout, 0, 1, &probeRelight.descriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probeRelight.pipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probeRelight.pipelineLayout, 2, 1, &materialDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, probeRelight.pipelineLayout, 3, 1, &textureDescriptor, 0, nullptr);

			vkCmdDispatch(cmd, groupcount, 1, 1);
		}

		{
			VkBufferMemoryBarrier barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER } ;
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.offset = 0;
			barrier.size = VK_WHOLE_SIZE;
			barrier.buffer = probeRelightOutputBuffer._buffer;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 1, & barrier, 0, nullptr);
		}

		//GI - Cluster Projection
		{
			
			int groupcount = ((precalculationLoadData.aabbClusterCount * precalculationInfo.clusterCoefficientCount) / 256) + 1;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterProjection.pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterProjection.pipelineLayout, 0, 1, &clusterProjection.descriptorSet, 0, nullptr);
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
			barrier.buffer = clusterProjectionOutputBuffer._buffer;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		//GI - Receiver Projection
		{
			int groupcount = ((precalculationLoadData.aabbClusterCount * precalculationInfo.maxReceiversInCluster) / 256) + 1;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, receiverReconstruction.pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, receiverReconstruction.pipelineLayout, 0, 1, &receiverReconstruction.descriptorSet, 0, nullptr);
			vkCmdDispatch(cmd, groupcount, 1, 1);
		}

		
		//{
		//	VkImageMemoryBarrier imageMemoryBarrier = {};
		//	imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		//	// We won't be changing the layout of the image
		//	imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		//	imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		//	imageMemoryBarrier.image = giInirectLightImage._image;
		//	imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		//	imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		//	imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		//	imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//	imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//	vkCmdPipelineBarrier(
		//		cmd,
		//		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		//		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		//		0,
		//		0, nullptr,
		//		0, nullptr,
		//		1, &imageMemoryBarrier);
		//}
		

		// LIGHTMAP DILATION RENDERIN
		{
			VkClearValue clearValue;
			clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_colorRenderPass, _lightmapExtent, _dilatedLightmapFramebuffer);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			cmd_viewport_scissor(cmd, _lightmapExtent);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _dilationPipeline);
			vkCmdPushConstants(cmd, _dilationPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::ivec2), &_lightmapExtent);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _dilationPipelineLayout, 0, 1, &_lightmapTextureDescriptor, 0, nullptr);
			vkCmdDraw(cmd, 3, 1, 0, 0);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}

		// GI LIGHTMAP DILATION RENDERIN
		{
			VkClearValue clearValue;
			clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_colorRenderPass, giLightmapExtent, dilatedGiInirectLightFramebuffer);

			rpInfo.clearValueCount = 1;
			VkClearValue clearValues[] = { clearValue };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			cmd_viewport_scissor(cmd, giLightmapExtent);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _dilationPipeline);
			vkCmdPushConstants(cmd, _dilationPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::ivec2), &giLightmapExtent);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _dilationPipelineLayout, 0, 1, &giInirectLightTextureDescriptor, 0, nullptr);
			vkCmdDraw(cmd, 3, 1, 0, 0);

			//finalize the render pass
			vkCmdEndRenderPass(cmd);
		}

		// GI RENDERING
		{
			VkClearValue clearValue;
			clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkClearValue depthClear;
			depthClear.depthStencil = { 1.0f, 0 };
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_colorDepthRenderPass, _windowExtent, _giFramebuffer);

			rpInfo.clearValueCount = 2;
			VkClearValue clearValues[] = { clearValue, depthClear };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			cmd_viewport_scissor(cmd, _windowExtent);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipelineLayout, 0, 1, &get_current_frame().globalDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipelineLayout, 2, 1, &textureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipelineLayout, 3, 1, &materialDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipelineLayout, 4, 1, &_dilatedLightmapTextureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _giPipelineLayout, 5, 1, &dilatedGiInirectLightTextureDescriptor, 0, nullptr);

			draw_objects(cmd);

			vkCmdEndRenderPass(cmd);
		}
		
		//POST PROCESSING + UI
		
		{
			VkClearValue clearValue;
			clearValue.color = { { 1.0f, 1.0f, 1.0f, 1.0f } };

			VkClearValue depthClear;
			depthClear.depthStencil = { 1.0f, 0 };
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass, _windowExtent, _framebuffers[swapchainImageIndex]);

			rpInfo.clearValueCount = 2;
			VkClearValue clearValues[] = { clearValue, depthClear };
			rpInfo.pClearValues = clearValues;
			
			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
			
			cmd_viewport_scissor(cmd, _windowExtent);
			
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _gammaPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _gammaPipelineLayout, 0, 1, &_giColorTextureDescriptor, 0, nullptr);
			vkCmdDraw(cmd, 3, 1, 0, 0);
			
			vkDebugRenderer.render(cmd, get_current_frame().globalDescriptor);
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
			
			vkCmdEndRenderPass(cmd);
		}
		
	}
	
	VK_CHECK(vkEndCommandBuffer(cmd));

	//prepare the submission to the queue. 
	//we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	//we will signal the _renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &get_current_frame()._presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &get_current_frame()._renderSemaphore;

	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

	//prepare present
	// this will put the image we just rendered to into the visible window.
	// we want to wait on the _renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo = vkinit::present_info();

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	//increase the number of frames drawn
	_frameNumber++;
}

void VulkanEngine::run()
{
	OPTICK_EVENT();

	SDL_Event e;
	bool bQuit = false;
	SDL_SetRelativeMouseMode(SDL_TRUE);
	bool onGui = false;
	//main loop
	while (!bQuit)
	{
		OPTICK_FRAME("MainThread");

		const Uint8* keys = SDL_GetKeyboardState(NULL);

		while (SDL_PollEvent(&e) != 0)
		{
			ImGui_ImplSDL2_ProcessEvent(&e);

			if (e.type == SDL_QUIT)
			{
				bQuit = true;
			}
			if (e.type == SDL_MOUSEMOTION && !onGui)
			{
				SDL_MouseMotionEvent motion = e.motion;

				camera.rotation.x += -0.05f * (float)motion.yrel;
				camera.rotation.y += -0.05f * (float)motion.xrel;
			}
			if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_LCTRL) {
				onGui = !onGui;
				SDL_SetRelativeMouseMode((SDL_bool) !onGui);
			}
		}

		glm::vec3 front;
		front.x = cos(glm::radians(camera.rotation.x)) * sin(glm::radians(camera.rotation.y));
		front.y = -sin(glm::radians(camera.rotation.x));
		front.z = cos(glm::radians(camera.rotation.x)) * cos(glm::radians(camera.rotation.y));
		front = glm::normalize(-front);

		float speed = 0.1f;

		if (keys[SDL_SCANCODE_LSHIFT]) {
			speed *= 4;
		}
		if (keys[SDL_SCANCODE_W])
		{
			camera.pos += front * speed;
		}
		if (keys[SDL_SCANCODE_S])
		{
			camera.pos -= front * speed;
		}
		if (keys[SDL_SCANCODE_A]) {
			camera.pos -= glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f))) * speed;
		}
		if (keys[SDL_SCANCODE_D]) {
			camera.pos += glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f))) * speed;
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL2_NewFrame(_window);

		ImGui::NewFrame();
		draw();
	}
}

void VulkanEngine::init_vulkan()
{
	OPTICK_EVENT();

	vkb::InstanceBuilder builder;

	//make the vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.require_api_version(1, 2, 0)
		.build();

	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance 
	_instance = vkb_inst.instance;
	_debug_messenger = vkb_inst.debug_messenger;

	SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

	VkPhysicalDeviceFeatures physicalDeviceFeatures = VkPhysicalDeviceFeatures();
	//physicalDeviceFeatures.fillModeNonSolid = true;
	physicalDeviceFeatures.samplerAnisotropy = VK_TRUE;
	physicalDeviceFeatures.shaderInt64 = true;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR featureRt = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	VkPhysicalDeviceAccelerationStructureFeaturesKHR featureAccel = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	VkPhysicalDeviceVulkan12Features features12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	
	features12.bufferDeviceAddress = true;
	features12.pNext = &featureRt;
	featureRt.rayTracingPipeline = true;
	featureRt.pNext = &featureAccel;
	featureAccel.accelerationStructure = true;

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	auto physicalDeviceSelectionResult = selector
		.set_minimum_version(1, 2)
		.set_surface(_surface)
		.set_required_features(physicalDeviceFeatures)
		.add_required_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
		.add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
		.add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
		.add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
		.add_required_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
		.select();
	
	//printf(physicalDeviceSelectionResult.error().message().c_str());

	auto physicalDevice = physicalDeviceSelectionResult.value();

	//create the final vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	deviceBuilder.add_pNext(&features12);
	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;
	_chosenGPU = physicalDevice.physical_device;

	//Get extension pointers
	load_VK_EXTENSIONS(_instance, vkGetInstanceProcAddr, _device, vkGetDeviceProcAddr);

	// use vkbootstrap to get a Graphics queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	_computeQueue = vkbDevice.get_queue(vkb::QueueType::compute).value();
	_computeQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::compute).value();

	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = _chosenGPU;
	allocatorInfo.device = _device;
	allocatorInfo.instance = _instance;
	allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator(&allocatorInfo, &_allocator);

	_gpuRaytracingProperties = {};
	_gpuProperties = {};

	_gpuRaytracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	_gpuProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	_gpuProperties.pNext = &_gpuRaytracingProperties;
	vkGetPhysicalDeviceProperties2(_chosenGPU, &_gpuProperties);
	vkGetPhysicalDeviceFeatures(_chosenGPU, &_gpuFeatures);
}

void VulkanEngine::init_swapchain()
{
	OPTICK_EVENT();

	vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_device,_surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();

	_swachainImageFormat = vkbSwapchain.image_format;

	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
	});

	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};
	_depthFormat = VK_FORMAT_D32_SFLOAT;
	
	VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	//allocate and create the image
	vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

	//build an image-view for the depth image to use for rendering
	VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	//add to deletion queues
	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthImageView, nullptr);
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
	});
}

void VulkanEngine::init_default_renderpass()
{
	OPTICK_EVENT();

	//we define an attachment description for our main color image
	//the attachment is loaded as "clear" when renderpass start
	//the attachment is stored when renderpass ends
	//the attachment layout starts as "undefined", and transitions to "Present" so its possible to display it
	//we dont care about stencil, and dont use multisampling

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = _swachainImageFormat;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	// Depth attachment
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	// Subpass dependencies for layout transitions
	VkSubpassDependency dependencies[2];

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	//array of 2 attachments, one for the color, and other for depth
	VkAttachmentDescription attachments[2] = { color_attachment,depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = attachments;
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	render_pass_info.dependencyCount = 2;
	render_pass_info.pDependencies = dependencies;

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
	});
}

void VulkanEngine::init_colordepth_renderpass()
{
	OPTICK_EVENT();

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = colorFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depthAttachment{};
	depthAttachment.format = _depthFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;							// Clear depth at beginning of the render pass
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;						// We will read from depth, so it's important to store the depth attachment results
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;					// We don't care about initial layout of the attachment
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;// Attachment will be transitioned to shader read at render pass end

	VkAttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;			// Attachment will be used as depth/stencil during render pass

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;
	subpass.pDepthStencilAttachment = &depthReference;									// Reference to our depth attachment

	// Use subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkAttachmentDescription attachments[2] = { colorAttachment,depthAttachment };

	VkRenderPassCreateInfo renderPassCreateInfo = {};
	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.attachmentCount = 2;
	renderPassCreateInfo.pAttachments = attachments;
	renderPassCreateInfo.subpassCount = 1;
	renderPassCreateInfo.pSubpasses = &subpass;
	renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassCreateInfo.pDependencies = dependencies.data();

	VK_CHECK(vkCreateRenderPass(_device, &renderPassCreateInfo, nullptr, &_colorDepthRenderPass));
}

void VulkanEngine::init_color_renderpass()
{
	OPTICK_EVENT();

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = colorFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;

	// Use subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkAttachmentDescription attachments[1] = { colorAttachment };

	VkRenderPassCreateInfo renderPassCreateInfo = {};
	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.attachmentCount = 1;
	renderPassCreateInfo.pAttachments = attachments;
	renderPassCreateInfo.subpassCount = 1;
	renderPassCreateInfo.pSubpasses = &subpass;
	renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassCreateInfo.pDependencies = dependencies.data();

	VK_CHECK(vkCreateRenderPass(_device, &renderPassCreateInfo, nullptr, &_colorRenderPass));
}

void VulkanEngine::init_framebuffers()
{
	OPTICK_EVENT();

	// Create default framebuffers
	{
		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_renderPass, _windowExtent);

		const uint32_t swapchain_imagecount = _swapchainImages.size();
		_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

		for (int i = 0; i < swapchain_imagecount; i++)
		{
			VkImageView attachments[2];
			attachments[0] = _swapchainImageViews[i];
			attachments[1] = _depthImageView;

			fb_info.pAttachments = attachments;
			fb_info.attachmentCount = 2;

			VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

			_mainDeletionQueue.push_function([=]() {
				vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
				vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
				});
		}
	}

	//Create a linear sampler
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 1.0f;
	VK_CHECK(vkCreateSampler(_device, &samplerInfo, nullptr, &_linearSampler));


	// Create shadowmap framebuffer
	{
		VkExtent3D depthImageExtent3D = {
			_shadowMapExtent.width,
			_shadowMapExtent.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_shadowMapColorImage._image, &_shadowMapColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(colorFormat, _shadowMapColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_shadowMapColorImageView));
		}

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_shadowMapDepthImage._image, &_shadowMapDepthImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(_depthFormat, _shadowMapDepthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_shadowMapDepthImageView));
		}

		VkImageView attachments[2] = { _shadowMapColorImageView, _shadowMapDepthImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_colorDepthRenderPass, _shadowMapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_shadowMapFramebuffer));
	}

	// Create lightmap framebuffer and its sampler
	{
		VkExtent3D lightmapImageExtent3D = {
			_lightmapExtent.width,
			_lightmapExtent.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_lightmapColorImage._image, &_lightmapColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(colorFormat, _lightmapColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_lightmapColorImageView));
		}

		VkImageView attachments[1] = { _lightmapColorImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_colorRenderPass, _lightmapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_lightmapFramebuffer));
	}

	// Create dilated lightmap framebuffer and its sampler
	{
		VkExtent3D lightmapImageExtent3D = {
			_lightmapExtent.width,
			_lightmapExtent.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_dilatedLightmapColorImage._image, &_dilatedLightmapColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(colorFormat, _dilatedLightmapColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_dilatedLightmapColorImageView));
		}

		VkImageView attachments[1] = { _dilatedLightmapColorImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_colorRenderPass, _lightmapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_dilatedLightmapFramebuffer));
	}

	// Create GI framebuffer
	{
		VkExtent3D giImageExtent3D = {
			_windowExtent.width,
			_windowExtent.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, giImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_giColorImage._image, &_giColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(colorFormat, _giColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_giColorImageView));
		}

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, giImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_giDepthImage._image, &_giDepthImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(_depthFormat, _giDepthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_giDepthImageView));
		}

		VkImageView attachments[2] = { _giColorImageView, _giDepthImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_colorDepthRenderPass, _windowExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_giFramebuffer));
	}
}

void VulkanEngine::init_commands()
{
	OPTICK_EVENT();

	//create a command pool for commands submitted to the graphics queue.
	//we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	for (int i = 0; i < FRAME_OVERLAP; i++) {
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

		//allocate the default command buffer that we will use for rendering
		VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
		});
	}

	//create pool for upload context
	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
	});
}

void VulkanEngine::init_sync_structures()
{
	OPTICK_EVENT();

	//create syncronization structures
	//one fence to control when the gpu has finished rendering the frame,
	//and 2 semaphores to syncronize rendering with swapchain
	//we want the fence to start signalled so we can wait on it on the first frame
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	for (int i = 0; i < FRAME_OVERLAP; i++) {
		VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

		//enqueue the destruction of the fence
		_mainDeletionQueue.push_function([=]() {
			vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
			});


		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));
		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

		//enqueue the destruction of semaphores
		_mainDeletionQueue.push_function([=]() {
			vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
			vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
			});
	}

	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();
	VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._fence));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_device, _uploadContext._fence, nullptr);
	});
}

void VulkanEngine::init_descriptors()
{
	OPTICK_EVENT();

	std::vector<VkDescriptorPoolSize> sizes =
	{
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = 0;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = (uint32_t)sizes.size();
	pool_info.pPoolSizes = sizes.data();
	vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptorPool);

	VkDescriptorSetLayoutCreateInfo setinfo = {};

	//binding set for camera data + shadow map data
	VkDescriptorSetLayoutBinding bindings[2] = { 
		// camera
		vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		// shadow map data
		 vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1)
	};
	setinfo = vkinit::descriptorset_layout_create_info(bindings, 2);
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_globalSetLayout);

	//binding set for object data
	VkDescriptorSetLayoutBinding objectBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0);
	setinfo = vkinit::descriptorset_layout_create_info(&objectBind, 1);
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_objectSetLayout);

	//binding set for texture data
	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT, 0);
	textureBind.descriptorCount = MAX_TEXTURES;
	setinfo = vkinit::descriptorset_layout_create_info(&textureBind, 1);
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_textureSetLayout);

	//binding set for materials
	VkDescriptorSetLayoutBinding materialBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT, 0);
	setinfo = vkinit::descriptorset_layout_create_info(&materialBind, 1);
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_materialSetLayout);

	//binding set for shadow map data
	VkDescriptorSetLayoutBinding shadowMapDataBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	setinfo = vkinit::descriptorset_layout_create_info(&shadowMapDataBind, 1);
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_shadowMapDataSetLayout);

	//binding set fragment shader single texture
	VkDescriptorSetLayoutBinding fragmentTextureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	setinfo = vkinit::descriptorset_layout_create_info(&fragmentTextureBind, 1);
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_fragmentTextureDescriptorSetLayout);


	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		const int MAX_OBJECTS = 10000;
		_frames[i].objectBuffer = vkutils::create_buffer(_allocator, sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].cameraBuffer = vkutils::create_buffer(_allocator, sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].shadowMapDataBuffer = vkutils::create_buffer(_allocator, sizeof(GPUShadowMapData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	
		VkDescriptorSetAllocateInfo allocInfo = {};

		allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_globalSetLayout, 1);
		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].globalDescriptor);

		allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_objectSetLayout, 1);
		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].objectDescriptor);
		
		allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_shadowMapDataSetLayout, 1);
		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].shadowMapDataDescriptor);

		VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &_frames[i].cameraBuffer._descriptorBufferInfo, 0);
		VkWriteDescriptorSet shadowMapDataDefaultWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &_frames[i].shadowMapDataBuffer._descriptorBufferInfo, 1);
		
		VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].objectDescriptor, &_frames[i].objectBuffer._descriptorBufferInfo, 0);
		VkWriteDescriptorSet shadowMapDataWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].shadowMapDataDescriptor, &_frames[i].shadowMapDataBuffer._descriptorBufferInfo, 0);

		VkWriteDescriptorSet setWrites[] = { cameraWrite, shadowMapDataDefaultWrite, objectWrite, shadowMapDataWrite };

		vkUpdateDescriptorSets(_device, 4, setWrites, 0, nullptr);
	}

	//Shadow map texture descriptor
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = _linearSampler;
		imageBufferInfo.imageView = _shadowMapColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_fragmentTextureDescriptorSetLayout, 1);

		vkAllocateDescriptorSets(_device, &allocInfo, &shadowMapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowMapTextureDescriptor, &imageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
	}

	//Lightmap texture descriptor
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = _linearSampler;
		imageBufferInfo.imageView = _lightmapColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_fragmentTextureDescriptorSetLayout, 1);

		vkAllocateDescriptorSets(_device, &allocInfo, &_lightmapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _lightmapTextureDescriptor, &imageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
	}

	//Dilated lightmap texture descriptor
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = _linearSampler;
		imageBufferInfo.imageView = _dilatedLightmapColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_fragmentTextureDescriptorSetLayout, 1);

		vkAllocateDescriptorSets(_device, &allocInfo, &_dilatedLightmapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _dilatedLightmapTextureDescriptor, &imageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
	}

	//GI texture descriptor
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = _linearSampler;
		imageBufferInfo.imageView = _giColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_fragmentTextureDescriptorSetLayout, 1);

		vkAllocateDescriptorSets(_device, &allocInfo, &_giColorTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _giColorTextureDescriptor, &imageBufferInfo, 0, 1);

		vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
	}

	_mainDeletionQueue.push_function([&]() {

		vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);

		vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);

		for (int i = 0; i < FRAME_OVERLAP; i++)
		{
			vmaDestroyBuffer(_allocator, _frames[i].cameraBuffer._buffer, _frames[i].cameraBuffer._allocation);
		}
	});

}

void VulkanEngine::init_pipelines(bool rebuild) {
	OPTICK_EVENT();

	VkShaderModule lightmapVertShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/lightmap.vert.spv", &lightmapVertShader))
	{
		assert("Lightmap Vertex Shader Loading Issue");
	}

	VkShaderModule lightmapFragShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/lightmap.frag.spv", &lightmapFragShader))
	{
		assert("Lightmap Vertex Shader Loading Issue");
	}

	VkShaderModule shadowMapVertShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/evsm.vert.spv", &shadowMapVertShader))
	{
		assert("Shadow Vertex Shader Loading Issue");
	}

	VkShaderModule shadowMapFragShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/evsm.frag.spv", &shadowMapFragShader))
	{
		assert("Shadow Fragment Shader Loading Issue");
	}

	VkShaderModule giVertShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/gi.vert.spv", &giVertShader))
	{
		assert("GI Vertex Shader Loading Issue");
	}

	VkShaderModule giFragShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/gi.frag.spv", &giFragShader))
	{
		assert("GI Fragment Shader Loading Issue");
	}

	VkShaderModule fullscreenVertShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/fullscreen.vert.spv", &fullscreenVertShader))
	{
		assert("Fullscreen vertex Shader Loading Issue");
	}
	
	VkShaderModule dilationFragShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/dilate.frag.spv", &dilationFragShader))
	{
		assert("Dilation Fragment Shader Loading Issue");
	}

	VkShaderModule gammaFragShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/gamma.frag.spv", &gammaFragShader))
	{
		assert("Gamma Fragment Shader Loading Issue");
	}

	if (!rebuild) {

		//SHADOWMAP PIPELINE LAYOUT INFO
		{
			VkDescriptorSetLayout setLayouts[] = { _shadowMapDataSetLayout, _objectSetLayout };
			VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 2);
			VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_shadowMapPipelineLayout));
		}

		//LIGHTMAP PIPELINE LAYOUT INFO
		{
			VkDescriptorSetLayout setLayouts[] = { _globalSetLayout, _objectSetLayout, _textureSetLayout, _materialSetLayout, _fragmentTextureDescriptorSetLayout, _fragmentTextureDescriptorSetLayout };
			VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 6);
			VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_lightmapPipelineLayout));
		}

		//GI PIPELINE LAYOUT INFO
		{
			VkDescriptorSetLayout setLayouts[] = { _globalSetLayout, _objectSetLayout, _textureSetLayout, _materialSetLayout, _fragmentTextureDescriptorSetLayout, _fragmentTextureDescriptorSetLayout };
			VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 6);
			VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_giPipelineLayout));
		}

		//DILATION PIPELINE LAYOUT INFO
		{
			VkDescriptorSetLayout setLayouts[] = { _fragmentTextureDescriptorSetLayout };
			VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 1);

			VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_FRAGMENT_BIT , 0, sizeof(glm::ivec2) };
			pipeline_layout_info.pushConstantRangeCount = 1;
			pipeline_layout_info.pPushConstantRanges = &pushConstantRanges;

			VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_dilationPipelineLayout));
		}

		//GAMMA PIPELINE LAYOUT INFO
		{
			VkDescriptorSetLayout setLayouts[] = { _fragmentTextureDescriptorSetLayout };
			VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 1);
			VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_gammaPipelineLayout));
		}
	}
	else {
		vkDestroyPipeline(_device, _lightmapPipeline, nullptr);
		vkDestroyPipeline(_device, _shadowMapPipeline, nullptr);
		vkDestroyPipeline(_device, _giPipeline, nullptr);
		vkDestroyPipeline(_device, _dilationPipeline, nullptr);
		vkDestroyPipeline(_device, _gammaPipeline, nullptr);
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

	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &vkinit::color_blend_attachment_state());

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

	//build the mesh triangle pipeline
	_lightmapPipeline = pipelineBuilder.build_pipeline(_device, _colorRenderPass);

	/*
	* / VVVVVVVVVVVVV SHADOW MAP PIPELINE VVVVVVVVVVVVV
	*/

	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, shadowMapVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, shadowMapFragShader));
	//pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(0, nullptr);

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
	pipelineBuilder._pipelineLayout = _shadowMapPipelineLayout;

	_shadowMapPipeline = pipelineBuilder.build_pipeline(_device, _colorDepthRenderPass);

	/*
	* / VVVVVVVVVVVVV GI PIPELINE VVVVVVVVVVVVV
	*/
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, giVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, giFragShader));
	pipelineBuilder._pipelineLayout = _giPipelineLayout;
	_giPipeline = pipelineBuilder.build_pipeline(_device, _colorDepthRenderPass);

	/*
	* / VVVVVVVVVVVVV DILATION PIPELINE VVVVVVVVVVVVV
	*/
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, fullscreenVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, dilationFragShader));
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
	pipelineBuilder._pipelineLayout = _dilationPipelineLayout;

	_dilationPipeline = pipelineBuilder.build_pipeline(_device, _colorRenderPass);
	/*
	* / VVVVVVVVVVVVV GAMMA PIPELINE VVVVVVVVVVVVV
	*/
	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, fullscreenVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, gammaFragShader));
	pipelineBuilder._pipelineLayout = _gammaPipelineLayout;

	_gammaPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

	//destroy all shader modules, outside of the queue
	vkDestroyShaderModule(_device, lightmapVertShader, nullptr);
	vkDestroyShaderModule(_device, lightmapFragShader, nullptr);
	vkDestroyShaderModule(_device, shadowMapVertShader, nullptr);
	vkDestroyShaderModule(_device, shadowMapFragShader, nullptr);
	vkDestroyShaderModule(_device, giVertShader, nullptr);
	vkDestroyShaderModule(_device, giFragShader, nullptr);

	vkDestroyShaderModule(_device, fullscreenVertShader, nullptr);
	vkDestroyShaderModule(_device, dilationFragShader, nullptr);
	vkDestroyShaderModule(_device, gammaFragShader, nullptr);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, _lightmapPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _lightmapPipelineLayout, nullptr);

		vkDestroyPipeline(_device, _shadowMapPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _shadowMapPipelineLayout, nullptr);

		vkDestroyPipeline(_device, _giPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _giPipelineLayout, nullptr);

		vkDestroyPipeline(_device, _dilationPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _dilationPipelineLayout, nullptr);

		vkDestroyPipeline(_device, _gammaPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _gammaPipelineLayout, nullptr);
	});
}

void VulkanEngine::init_scene()
{
	OPTICK_EVENT();

	std::string file_name = "../../assets/cornellBox.gltf";
	//std::string file_name = "../../assets/Sponza/glTF/Sponza.gltf";
	//std::string file_name = "../../assets/VC/glTF/VC.gltf";
	
	tinygltf::Model tmodel;
	tinygltf::TinyGLTF tcontext;

	std::string warn, error;
	if (!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, file_name)) {
		assert(!"Error while loading scene");
	}
	if (!warn.empty()) {
		printf("WARNING: SCENE LOADING: %s\n", warn);
	}
	if (!error.empty()) {
		printf("WARNING: SCENE LOADING: %s\n", error);
	}

	gltf_scene.import_materials(tmodel);
	gltf_scene.import_drawable_nodes(tmodel, GltfAttributes::Normal |
		GltfAttributes::Texcoord_0);
	
	std::vector<GPUBasicMaterialData> materials;

	for (int i = 0; i < gltf_scene.materials.size(); i++) {
		GPUBasicMaterialData material = {};
		material.base_color = gltf_scene.materials[i].base_color_factor;
		material.emissive_color = gltf_scene.materials[i].emissive_factor;
		material.metallic_factor = gltf_scene.materials[i].metallic_factor;
		material.roughness_factor = gltf_scene.materials[i].roughness_factor;
		material.texture = gltf_scene.materials[i].base_color_texture;
		material.normal_texture = gltf_scene.materials[i].normal_texture;
		material.metallic_roughness_texture = gltf_scene.materials[i].metallic_rougness_texture;
		materials.push_back(material);
	}

	//TODO: Try to load lightmap uvs from disk first, if not available generate them
	/*
	* File format:
	* atlas width (4 bytes), atlas height (4 bytes)
	* all the data
	* vec2 data as byte
	*/
	printf("Generating lightmap uvs\n");
	xatlas::Atlas* atlas = xatlas::Create();
	for (int i = 0; i < gltf_scene.nodes.size(); i++) {
		auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
		xatlas::MeshDecl meshDecleration = {};
		meshDecleration.vertexPositionData = &gltf_scene.positions[mesh.vtx_offset];
		meshDecleration.vertexPositionStride = sizeof(glm::vec3);
		meshDecleration.vertexCount = mesh.vtx_count;

		meshDecleration.indexData = &gltf_scene.indices[mesh.first_idx];
		meshDecleration.indexCount = mesh.idx_count;
		meshDecleration.indexFormat = xatlas::IndexFormat::UInt32;
		xatlas::AddMesh(atlas, meshDecleration);
	}

	xatlas::ChartOptions chartOptions = xatlas::ChartOptions();
	xatlas::PackOptions packOptions = xatlas::PackOptions();
	packOptions.resolution = 256;
	xatlas::Generate(atlas, chartOptions, packOptions);

	gltf_scene.lightmap_width = atlas->width;
	gltf_scene.lightmap_height = atlas->height;

	std::vector<GltfPrimMesh> prim_meshes;
	std::vector<glm::vec3> positions;
	std::vector<uint32_t> indices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texcoords0;

	for (int i = 0; i < gltf_scene.nodes.size(); i++) {
		GltfPrimMesh mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
		uint32_t orihinal_vtx_offset = mesh.vtx_offset;
		mesh.first_idx = indices.size();
		mesh.vtx_offset = positions.size();

		mesh.idx_count = atlas->meshes[i].indexCount;
		mesh.vtx_count = atlas->meshes[i].vertexCount;

		gltf_scene.nodes[i].prim_mesh = prim_meshes.size();
		prim_meshes.push_back(mesh);

		for (int j = 0; j < atlas->meshes[i].vertexCount; j++) {
			gltf_scene.lightmapUVs.push_back({ atlas->meshes[i].vertexArray[j].uv[0], atlas->meshes[i].vertexArray[j].uv[1] });
			
			positions.push_back(gltf_scene.positions[atlas->meshes[i].vertexArray[j].xref + orihinal_vtx_offset]);
			normals.push_back(gltf_scene.normals[atlas->meshes[i].vertexArray[j].xref + orihinal_vtx_offset]);
			texcoords0.push_back(gltf_scene.texcoords0[atlas->meshes[i].vertexArray[j].xref + orihinal_vtx_offset]);
		}

		for (int j = 0; j < atlas->meshes[i].indexCount; j++) {
			indices.push_back(atlas->meshes[i].indexArray[j]);
		}
	}

	gltf_scene.prim_meshes.clear();
	gltf_scene.positions.clear();
	gltf_scene.indices.clear();
	gltf_scene.normals.clear();
	gltf_scene.texcoords0.clear();

	gltf_scene.prim_meshes = prim_meshes;
	gltf_scene.positions = positions;
	gltf_scene.indices = indices;
	gltf_scene.normals = normals;
	gltf_scene.texcoords0 = texcoords0;

	printf("Generated lightmap uvs, %d x %d\n", atlas->width, atlas->height);
	xatlas::Destroy(atlas);

	/*
	* TODO: After reading the GLTF scene, what I can do is:
	* Create the xatlas
	* Create a new vertex buffer (also normal and tex)
	* Correct the node data
	*/

	vertex_buffer = create_upload_buffer(gltf_scene.positions.data(), gltf_scene.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);

	index_buffer = create_upload_buffer(gltf_scene.indices.data(), gltf_scene.indices.size() * sizeof(uint32_t),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);

	normal_buffer = create_upload_buffer(gltf_scene.normals.data(), gltf_scene.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	tex_buffer = create_upload_buffer(gltf_scene.texcoords0.data(), gltf_scene.texcoords0.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	lightmap_tex_buffer = create_upload_buffer(gltf_scene.lightmapUVs.data(), gltf_scene.lightmapUVs.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	material_buffer = create_upload_buffer(materials.data(), materials.size() * sizeof(GPUBasicMaterialData),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// MATERIAL DESCRIPTOR
	{
		VkDescriptorSetAllocateInfo materialSetAlloc =
			vkinit::descriptorset_allocate_info(_descriptorPool, &_materialSetLayout, 1);

		vkAllocateDescriptorSets(_device, &materialSetAlloc, &materialDescriptor);

		VkWriteDescriptorSet materialWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, materialDescriptor, &material_buffer._descriptorBufferInfo, 0);
		vkUpdateDescriptorSets(_device, 1, &materialWrite, 0, nullptr);
	}
	std::vector<VkDescriptorImageInfo> image_infos;

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	samplerInfo.minLod = 0.0f; // Optional
	samplerInfo.maxLod = FLT_MAX;
	samplerInfo.mipLodBias = 0.0f; // Optional

	samplerInfo.anisotropyEnable = _gpuFeatures.samplerAnisotropy;
	samplerInfo.maxAnisotropy = _gpuFeatures.samplerAnisotropy
		? _gpuProperties.properties.limits.maxSamplerAnisotropy
		: 1.0f;

	VkSampler blockySampler;
	vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);
	std::array<uint8_t, 4> nil = { 0, 0, 0, 0 };

	if (tmodel.images.size() == 0) {
		AllocatedImage allocated_image;
		uint32_t mipLevels;
		vkutils::load_image_from_memory(*this, nil.data(), 1, 1, allocated_image, mipLevels);

		VkImageView imageView;
		VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, allocated_image._image, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		vkCreateImageView(_device, &imageinfo, nullptr, &imageView);

		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = blockySampler;
		imageBufferInfo.imageView = imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		image_infos.push_back(imageBufferInfo);
	}

	for (int i = 0; i < tmodel.images.size(); i++) {
		auto& gltf_img = tmodel.images[i];
		AllocatedImage allocated_image;
		uint32_t mipLevels;

		if (gltf_img.image.size() == 0 || gltf_img.width == -1 || gltf_img.height == -1) {
			vkutils::load_image_from_memory(*this, nil.data(), 1, 1, allocated_image, mipLevels);
		}
		else {
			vkutils::load_image_from_memory(*this, gltf_img.image.data(), gltf_img.width, gltf_img.height, allocated_image, mipLevels);
		}
	
		VkImageView imageView;
		VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, allocated_image._image, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		vkCreateImageView(_device, &imageinfo, nullptr, &imageView);

		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = blockySampler;
		imageBufferInfo.imageView = imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		image_infos.push_back(imageBufferInfo);
	}

	// TEXTURE DESCRIPTOR
	{
		VkDescriptorSetAllocateInfo allocInfo =
			vkinit::descriptorset_allocate_info(_descriptorPool, &_textureSetLayout, 1);

		vkAllocateDescriptorSets(_device, &allocInfo, &textureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureDescriptor, image_infos.data(), 0, image_infos.size());

		vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
	}

	vulkanRaytracing.convert_scene_to_vk_geometry(gltf_scene, vertex_buffer, index_buffer);
	vulkanRaytracing.build_blas(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
	vulkanRaytracing.build_tlas(gltf_scene, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, false);
}

void VulkanEngine::init_imgui()
{
	OPTICK_EVENT();

	//1: create descriptor pool for IMGUI
	// the size of the pool is very oversize, but it's copied from imgui demo itself.
	VkDescriptorPoolSize pool_sizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));


	// 2: initialize imgui library

	//this initializes the core structures of imgui
	ImGui::CreateContext();

	ImGui::StyleColorsDark();

	//this initializes imgui for SDL
	ImGui_ImplSDL2_InitForVulkan(_window);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = _instance;
	init_info.PhysicalDevice = _chosenGPU;
	init_info.Device = _device;
	init_info.Queue = _graphicsQueue;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	ImGui_ImplVulkan_Init(&init_info, _renderPass);

	//execute a gpu command to upload imgui font textures
	immediate_submit([&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
		});

	//clear font textures from cpu data
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	//add the destroy the imgui created structures
	_mainDeletionQueue.push_function([=]() {

		vkDestroyDescriptorPool(_device, imguiPool, nullptr);
		ImGui_ImplVulkan_Shutdown();
		});
}

void VulkanEngine::cmd_viewport_scissor(VkCommandBuffer cmd, VkExtent2D extent)
{
	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)extent.width;
	viewport.height = (float)extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = extent;
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void VulkanEngine::init_gi()
{
	bool loadPrecomputedData = false;

	if (!loadPrecomputedData) {
		precalculationInfo.voxelSize = 0.9;
		precalculationInfo.voxelPadding = 3;
		precalculationInfo.probeOverlaps = 10;
		precalculationInfo.raysPerProbe = 8000;
		precalculationInfo.raysPerReceiver = 8000;
		precalculationInfo.sphericalHarmonicsOrder = 7;
		precalculationInfo.clusterCoefficientCount = 32;
		precalculationInfo.maxReceiversInCluster = 1024;

		precalculation.prepare(*this, gltf_scene, precalculationInfo, precalculationLoadData, precalculationResult);
	}
	else {
		precalculation.load("../../precomputation/precalculation.cfg", precalculationInfo, precalculationLoadData, precalculationResult);
	}

	//Config buffer (GPU ONLY)
	config.probeCount = precalculationResult.probes.size();
	config.basisFunctionCount = SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder);
	config.rayCount = precalculationInfo.raysPerProbe;
	config.clusterCount = precalculationLoadData.aabbClusterCount;
	config.lightmapInputSize = glm::vec2(gltf_scene.lightmap_width, gltf_scene.lightmap_height);
	config.pcaCoefficient = precalculationInfo.clusterCoefficientCount;
	config.maxReceiversInCluster = precalculationInfo.maxReceiversInCluster;


	giLightmapExtent.width = gltf_scene.lightmap_width;
	giLightmapExtent.height = gltf_scene.lightmap_height;

	VkExtent3D lightmapImageExtent3D = {
		gltf_scene.lightmap_width,
		gltf_scene.lightmap_height,
		1
	};

	{
		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(colorFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &giInirectLightImage._image, &giInirectLightImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(colorFormat, giInirectLightImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &giInirectLightImageView));
		}

		{
			VkDescriptorImageInfo imageBufferInfo;
			imageBufferInfo.sampler = _linearSampler;
			imageBufferInfo.imageView = giInirectLightImageView;
			imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_fragmentTextureDescriptorSetLayout, 1);

			vkAllocateDescriptorSets(_device, &allocInfo, &giInirectLightTextureDescriptor);

			VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, giInirectLightTextureDescriptor, &imageBufferInfo, 0, 1);

			vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
		}
	}

	immediate_submit([&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier = {};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.image = giInirectLightImage._image;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
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
		});

	{
		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, lightmapImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &dilatedGiInirectLightImage._image, &dilatedGiInirectLightImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(colorFormat, dilatedGiInirectLightImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &dilatedGiInirectLightImageView));
		}

		VkImageView attachments[1] = { dilatedGiInirectLightImageView };
		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_colorRenderPass, giLightmapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &dilatedGiInirectLightFramebuffer));

		{
			VkDescriptorImageInfo imageBufferInfo;
			imageBufferInfo.sampler = _linearSampler;
			imageBufferInfo.imageView = dilatedGiInirectLightImageView;
			imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(_descriptorPool, &_fragmentTextureDescriptorSetLayout, 1);

			vkAllocateDescriptorSets(_device, &allocInfo, &dilatedGiInirectLightTextureDescriptor);

			VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, dilatedGiInirectLightTextureDescriptor, &imageBufferInfo, 0, 1);

			vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
		}
	}

	configBuffer = create_upload_buffer(&config, sizeof(GIConfig), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	//GPUProbeRaycastResult buffer (GPU ONLY)
	auto probeRaycastResultBuffer = create_upload_buffer(precalculationResult.probeRaycastResult, sizeof(GPUProbeRaycastResult) * config.probeCount * config.rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//ProbeBasisFunctions buffer (GPU ONLY)
	auto probeBasisBuffer = create_upload_buffer(precalculationResult.probeRaycastBasisFunctions, sizeof(glm::vec4) * (config.probeCount * config.rayCount * config.basisFunctionCount / 4 + 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_probe_projection Temp buffer (GPU ONLY)
	auto probeRelightTempBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * config.probeCount * config.rayCount * config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_probe_projection output buffer (GPU ONLY)
	probeRelightOutputBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec4) * config.probeCount * config.basisFunctionCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	
	//Create compute instances
	vulkanCompute.add_buffer_binding(probeRelight, ComputeBufferType::UNIFORM, configBuffer);
	vulkanCompute.add_buffer_binding(probeRelight, ComputeBufferType::STORAGE, probeRaycastResultBuffer);
	vulkanCompute.add_buffer_binding(probeRelight, ComputeBufferType::STORAGE, probeBasisBuffer);
	vulkanCompute.add_buffer_binding(probeRelight, ComputeBufferType::STORAGE, probeRelightTempBuffer);
	vulkanCompute.add_buffer_binding(probeRelight, ComputeBufferType::STORAGE, probeRelightOutputBuffer);
	vulkanCompute.add_texture_binding(probeRelight, ComputeBufferType::TEXTURE_SAMPLED, _linearSampler, _lightmapColorImageView);
	vulkanCompute.add_texture_binding(probeRelight, ComputeBufferType::TEXTURE_SAMPLED, _linearSampler, dilatedGiInirectLightImageView);
	
	vulkanCompute.add_descriptor_set_layout(probeRelight, _objectSetLayout);
	vulkanCompute.add_descriptor_set_layout(probeRelight, _materialSetLayout);
	vulkanCompute.add_descriptor_set_layout(probeRelight, _textureSetLayout);

	vulkanCompute.build(probeRelight, _descriptorPool, "../../shaders/gi_probe_projection.comp.spv");

	//Cluster projection matrices (GPU ONLY)
	auto clusterProjectionMatricesBuffer = create_upload_buffer(precalculationResult.clusterProjectionMatrices, (config.clusterCount * config.probeCount * config.basisFunctionCount * config.pcaCoefficient / 4 + 1) * sizeof(glm::vec4) , VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	//gi_cluster_projection output buffer (GPU ONLY)
	clusterProjectionOutputBuffer = vkutils::create_buffer(_allocator, config.clusterCount * config.pcaCoefficient * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	vulkanCompute.add_buffer_binding(clusterProjection, ComputeBufferType::UNIFORM, configBuffer);
	vulkanCompute.add_buffer_binding(clusterProjection, ComputeBufferType::STORAGE, probeRelightOutputBuffer);
	vulkanCompute.add_buffer_binding(clusterProjection, ComputeBufferType::STORAGE, clusterProjectionMatricesBuffer);
	vulkanCompute.add_buffer_binding(clusterProjection, ComputeBufferType::STORAGE, clusterProjectionOutputBuffer);
	vulkanCompute.build(clusterProjection, _descriptorPool, "../../shaders/gi_cluster_projection.comp.spv");


	auto receiverReconstructionMatricesBuffer = create_upload_buffer(precalculationResult.receiverCoefficientMatrices, (precalculationLoadData.totalClusterReceiverCount * config.pcaCoefficient / 4 + 1) * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterReceiverInfos = create_upload_buffer(precalculationResult.clusterReceiverInfos, config.clusterCount * sizeof(ClusterReceiverInfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	auto clusterReceiverUvs = create_upload_buffer(precalculationResult.clusterReceiverUvs, precalculationLoadData.totalClusterReceiverCount * sizeof(glm::ivec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);


	vulkanCompute.add_buffer_binding(receiverReconstruction, ComputeBufferType::UNIFORM, configBuffer);
	vulkanCompute.add_buffer_binding(receiverReconstruction, ComputeBufferType::STORAGE, clusterProjectionOutputBuffer);
	vulkanCompute.add_buffer_binding(receiverReconstruction, ComputeBufferType::STORAGE, receiverReconstructionMatricesBuffer);
	vulkanCompute.add_buffer_binding(receiverReconstruction, ComputeBufferType::STORAGE, clusterReceiverInfos);
	vulkanCompute.add_buffer_binding(receiverReconstruction, ComputeBufferType::STORAGE, clusterReceiverUvs);
	vulkanCompute.add_texture_binding(receiverReconstruction, ComputeBufferType::TEXTURE_STORAGE, 0, giInirectLightImageView);
	vulkanCompute.build(receiverReconstruction, _descriptorPool, "../../shaders/gi_receiver_reconstruction.comp.spv");
}

AllocatedBuffer VulkanEngine::create_upload_buffer(void* buffer_data, size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	OPTICK_EVENT();

	AllocatedBuffer stagingBuffer = vkutils::create_buffer(_allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	
	vkutils::cpu_to_gpu(_allocator, stagingBuffer, buffer_data, size);

	AllocatedBuffer new_buffer = vkutils::create_buffer(_allocator, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memoryUsage);

	immediate_submit([=](VkCommandBuffer cmd) {
		VkBufferCopy copy;
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = size;
		vkCmdCopyBuffer(cmd, stagingBuffer._buffer, new_buffer._buffer, 1, &copy);
	});

	_mainDeletionQueue.push_function([=]() {
		vmaDestroyBuffer(_allocator, new_buffer._buffer, new_buffer._allocation);
	});

	vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);

	return new_buffer;
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd)
{
	OPTICK_EVENT();

	{
		VkDeviceSize offsets[] = { 0, 0, 0, 0 };
		VkBuffer buffers[] = { vertex_buffer._buffer, normal_buffer._buffer, tex_buffer._buffer, lightmap_tex_buffer._buffer };
		vkCmdBindVertexBuffers(cmd, 0, 4, buffers, offsets);
		vkCmdBindIndexBuffer(cmd, index_buffer._buffer, 0, VK_INDEX_TYPE_UINT32);

		for (int i = 0; i < gltf_scene.nodes.size(); i++) {
			auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
			vkCmdDrawIndexed(cmd, mesh.idx_count, 1, mesh.first_idx, mesh.vtx_offset, i);
		}
	}
}

FrameData& VulkanEngine::get_current_frame()
{
	return _frames[_frameNumber % FRAME_OVERLAP];
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
	OPTICK_EVENT();

	// Calculate required alignment based on minimum device offset alignment
	size_t minUboAlignment = _gpuProperties.properties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minUboAlignment > 0) {
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}
	return alignedSize;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
	OPTICK_EVENT();

	//allocate the default command buffer that we will use for the instant commands
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);

	VkCommandBuffer cmd;
	VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &cmd));

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	//execute the function
	function(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);


	//submit command buffer to the queue and execute it.
	// _uploadFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._fence));

	vkWaitForFences(_device, 1, &_uploadContext._fence, true, 9999999999);
	vkResetFences(_device, 1, &_uploadContext._fence);

	//clear the command pool. This will free the command buffer too
	vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}
