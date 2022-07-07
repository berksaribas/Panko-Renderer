#define TINYGLTF_IMPLEMENTATION
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
#include <vk_utils.h>
#include <vk_extensions.h>
#include <xatlas.h>

#include <gi_diffuse.h>
#include <gi_glossy.h>
#include <gi_shadow.h>
#include <gi_gbuffer.h>
#include <gi_deferred.h>
#include <gi_brdf.h>
#include <gi_glossy_svgf.h>
#include <vk_timer.h>

#include <ctime>

#include <functional>
#include <vk_debug_renderer.h>
#include "VkBootstrap.h"


#define FILE_HELPER_IMPL
#include <file_helper.h>
#include <filesystem>

#include "vk_rendergraph.h"

constexpr bool bUseValidationLayers = true;
std::vector<GPUBasicMaterialData> materials;

//Precalculation
Precalculation precalculation;
PrecalculationInfo precalculationInfo = {};
PrecalculationLoadData precalculationLoadData = {};
PrecalculationResult precalculationResult = {};

//GI Models
BRDF brdfUtils;
GBuffer gbuffer;
DiffuseIllumination diffuseIllumination;
Shadow shadow;
GlossyIllumination glossyIllumination;
GlossyDenoise glossyDenoise;
Deferred deferred;

//Imgui
bool enableGi = false;
bool showProbes = false;
bool showProbeRays = false;
bool showReceivers = false;
bool showSpecificReceiver = false;
int specificCluster = 1;
int specificReceiver = 135;
int specificReceiverRaySampleCount = 10;
bool showSpecificProbeRays = false;
bool probesEnabled[300];
int renderMode = 0;

int selectedPreset = -1;
char customName[128];
bool screenshot = false;

bool useRealtimeRaycast = true;
bool enableDenoise = true;
bool enableGroundTruthDiffuse = false;

bool showFileList = false;
std::vector<std::string> load_files;
int selected_file = 0;

bool useSceneCamera = false;
CameraConfig cameraConfig;

void VulkanEngine::init()
{
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

	_engineData = {};

	init_vulkan();
	_engineData.renderGraph = new Vrg::RenderGraph(&_engineData);

	init_swapchain();
	init_commands();
	init_sync_structures();
	init_descriptor_pool();

	_vulkanRaytracing.init(_engineData, _gpuRaytracingProperties);
	_engineData.renderGraph->enable_raytracing(&_vulkanRaytracing);

	init_imgui();

	bool loadPrecomputedData = true;
	if (!loadPrecomputedData) {
		precalculationInfo.voxelSize = 0.25;
		precalculationInfo.voxelPadding = 2;
		precalculationInfo.probeOverlaps = 10;
		precalculationInfo.raysPerProbe = 1000;
		precalculationInfo.raysPerReceiver = 40000;
		precalculationInfo.sphericalHarmonicsOrder = 7;
		precalculationInfo.clusterCoefficientCount = 32;
		precalculationInfo.maxReceiversInCluster = 1024;
		precalculationInfo.lightmapResolution = 261;
		precalculationInfo.texelSize = 6;
		precalculationInfo.desiredSpacing = 2;
	}
	else {
		precalculation.load("../../precomputation/precalculation.cfg", precalculationInfo, precalculationLoadData, precalculationResult);
	}

	init_scene();

	if (!loadPrecomputedData) {
		precalculation.prepare(*this, gltf_scene, precalculationInfo, precalculationLoadData, precalculationResult);
		//precalculation.prepare(*this, gltf_scene, precalculationInfo, precalculationLoadData, precalculationResult, "../../precomputation/precalculation.Probes");
		exit(0);
	}
	
	init_query_pool();
	init_descriptors();

	brdfUtils.init_images(_engineData);
	shadow.init_buffers(_engineData);
	shadow.init_images(_engineData);
	gbuffer.init_images(_engineData, _windowExtent);
	diffuseIllumination.init(_engineData, &precalculationInfo, &precalculationLoadData, &precalculationResult, gltf_scene);
	glossyIllumination.init_images(_engineData, _windowExtent);
	glossyDenoise.init_images(_engineData, _windowExtent);
	deferred.init_images(_engineData, _windowExtent);

	shadow._shadowMapData.positiveExponent = 40;
	shadow._shadowMapData.negativeExponent = 5;
	shadow._shadowMapData.LightBleedingReduction = 0.999f;
	shadow._shadowMapData.VSMBias = 0.01;
	_camData.lightPos = { 0.020, 2, 0.140, 0.0f };
	_camData.lightColor = { 1.f, 1.f, 1.f, 1.0f };
	_camData.indirectDiffuse = true;
	_camData.indirectSpecular = true;
	_camData.useStochasticSpecular = true;
	_camData.clearColor = { 0.f, 0.f, 0.f, 0.0f };
	camera.pos = { 0, 0, 7 };
	_camData.glossyFrameCount = 0;
	_camData.glossyDenoise = 1;

	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::cleanup()
{
	if (_isInitialized) {
		//make sure the GPU has stopped doing its things
		vkDeviceWaitIdle(_engineData.device);

		//TODO
		//_engineData.renderGraph->clear();
		_mainDeletionQueue.flush();

		vmaDestroyAllocator(_engineData.allocator);

		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		vkDestroyDevice(_engineData.device, nullptr);
		vkDestroyInstance(_instance, nullptr);

		SDL_DestroyWindow(_window);
	}
}

static char buffer[256];

void VulkanEngine::prepare_gui() {
	{
		//todo: imgui stuff
		ImGui::Begin("Engine Config");
		{
			sprintf_s(buffer, "Rebuild Shaders");
			if (ImGui::Button(buffer)) {
				_engineData.renderGraph->rebuild_pipelines();
			}

			ImGui::NewLine();

			sprintf_s(buffer, "Positive Exponent");
			ImGui::DragFloat(buffer, &shadow._shadowMapData.positiveExponent);
			sprintf_s(buffer, "Negative Exponent");
			ImGui::DragFloat(buffer, &shadow._shadowMapData.negativeExponent);
			sprintf_s(buffer, "Light Bleeding Reduction");
			ImGui::DragFloat(buffer, &shadow._shadowMapData.LightBleedingReduction);
			sprintf_s(buffer, "VSM Bias");
			ImGui::DragFloat(buffer, &shadow._shadowMapData.VSMBias);

			sprintf_s(buffer, "Ligh Direction");
			ImGui::DragFloat3(buffer, &_camData.lightPos.x);

			sprintf_s(buffer, "Light Color");
			ImGui::ColorEdit3(buffer, &_camData.lightColor.x);

			ImGui::NewLine();

			sprintf_s(buffer, "Indirect Diffuse");
			ImGui::Checkbox(buffer, (bool*)&_camData.indirectDiffuse);

			if (_camData.indirectDiffuse) {
				sprintf_s(buffer, "Ground Truth");
				if (ImGui::Checkbox(buffer, &enableGroundTruthDiffuse)) {
					_frameNumber = 0;
				}
			}

			sprintf_s(buffer, "Use realtime probe raycasting");
			ImGui::Checkbox(buffer, (bool*)&useRealtimeRaycast);

			sprintf_s(buffer, "Indirect Specular");
			ImGui::Checkbox(buffer, (bool*)&_camData.indirectSpecular);

			if (_camData.indirectSpecular) {
				sprintf_s(buffer, "Use Stochastic Raytracing");
				if (ImGui::Checkbox(buffer, (bool*)&_camData.useStochasticSpecular)) {
					///_frameNumber = 0;
				}

				if (_camData.useStochasticSpecular) {
					sprintf_s(buffer, "Enable SVGF denoising");
					if (ImGui::Checkbox(buffer, &enableDenoise)) {
						_camData.glossyFrameCount = 0;
						_camData.glossyDenoise = enableDenoise;
					}



					if (enableDenoise) {
						ImGui::Text("Currently using stochastic raytracing + SVGF denoising.");
						ImGui::InputInt("Atrous iterations", &glossyDenoise.num_atrous);
					}
					else {
						ImGui::Text("Currently using stochastic raytracing");
					}
				}
				else {
					ImGui::Text("Currently using mirror raytracing + blurred mip chaining.");
				}
			}

			ImGui::NewLine();
			ImGui::Checkbox("Show Probes", &showProbes);
			if (showProbes) {
				ImGui::Checkbox("Show Probe Rays", &showProbeRays);
			}
			ImGui::Checkbox("Show Receivers", &showReceivers);
			ImGui::Checkbox("Show Specific Receivers", &showSpecificReceiver);
			if (showSpecificReceiver) {
				ImGui::SliderInt("Cluster: ", &specificCluster, 0, precalculationLoadData.aabbClusterCount - 1);
				ImGui::SliderInt("Receiver: ", &specificReceiver, 0, precalculationResult.clusterReceiverInfos[specificCluster].receiverCount - 1);

				ImGui::DragInt("Ray sample count: ", &specificReceiverRaySampleCount);

				ImGui::Checkbox("Show Selected Probe Rays", &showSpecificProbeRays);

				{
					int receiverCount = precalculationResult.clusterReceiverInfos[specificCluster].receiverCount;
					int receiverOffset = precalculationResult.clusterReceiverInfos[specificCluster].receiverOffset;
					int probeCount = precalculationResult.clusterReceiverInfos[specificCluster].probeCount;
					int probeOffset = precalculationResult.clusterReceiverInfos[specificCluster].probeOffset;

					for (int i = 0; i < probeCount; i++) {
						if (precalculationResult.receiverProbeWeightData[(receiverOffset + specificReceiver) * precalculationLoadData.maxProbesPerCluster + i] > 0) {
							int realProbeIndex = precalculationResult.clusterProbes[probeOffset + i];
							sprintf_s(buffer, "Probe %d: %f", realProbeIndex, precalculationResult.receiverProbeWeightData[(receiverOffset + specificReceiver) * precalculationLoadData.maxProbesPerCluster + i]);
							ImGui::Checkbox(buffer, &probesEnabled[realProbeIndex]);
						}
					}
				}
			}

			ImGui::NewLine();

			ImGui::InputText("Custom name", customName, sizeof(customName));
			if (ImGui::Button("Screenshot")) {
				screenshot = true;
			}

			if (ImGui::Button("Show Saved Data")) {
				std::string path = "./screenshots/";
				load_files.clear();
				for (const auto& entry : std::filesystem::directory_iterator(path)) {
					auto& path = entry.path();
					if (path.extension().generic_string().compare(".cam") == 0) {
						load_files.push_back(path.generic_string());
					}
				}
				showFileList = true;
			}

			ImGui::End();
		}

		ImGui::Begin("Camera");
		{
			ImGui::ColorEdit4("Clear Color", &_camData.clearColor.r);
			if (gltf_scene.cameras.size() > 0) {
				ImGui::Checkbox("Use scene camera", &useSceneCamera);
			}
			ImGui::NewLine();

			ImGui::SliderFloat("Camera Fov", &cameraConfig.fov, 0, 90);
			ImGui::SliderFloat("Camera Speed", &cameraConfig.speed, 0, 1);
			ImGui::SliderFloat("Camera Rotation Speed", &cameraConfig.rotationSpeed, 0, 0.5);

			ImGui::End();
		}

		ImGui::Begin("Textures");
		{
			//ImGui::Image(shadow._shadowMapTextureDescriptor, { 128, 128 });
			//ImGui::Image(diffuseIllumination._giIndirectLightTextureDescriptor, { (float)precalculationInfo.lightmapResolution,  (float)precalculationInfo.lightmapResolution });
			//ImGui::Image(diffuseIllumination._dilatedGiIndirectLightTextureDescriptor, { (float)precalculationInfo.lightmapResolution,  (float)precalculationInfo.lightmapResolution });
			//ImGui::Image(glossyIllumination._glossyReflectionsColorTextureDescriptor, { 320, 180 });
			ImGui::End();
		}

		//ImGui::Begin("Viewport", 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
		//{
		//	
		//	ImGui::Image(deferred._deferredColorTextureDescriptor, ImGui::GetWindowContentRegionMax());
		//	ImGui::End();
		//}

		if (showFileList) {
			ImGui::Begin("Files", &showFileList);
			ImGui::ListBoxHeader("");
			for (int i = 0; i < load_files.size(); i++) {
				if (ImGui::Selectable(load_files[i].c_str(), i == selected_file)) {
					selected_file = i;
				}
			}
			ImGui::ListBoxFooter();
			if (ImGui::Button("Load")) {
				ScreenshotSaveData saveData = {};
				load_binary(load_files[selected_file], &saveData, sizeof(ScreenshotSaveData));
				camera = saveData.camera;
				_camData.lightPos = saveData.lightPos;
			}
			ImGui::End();
		}

		ImGui::Begin("Materials");
		{
			bool materialsChanged = false;
			for (int i = 0; i < materials.size(); i++) {
				sprintf_s(buffer, "Material %d - %s", i, "todo");
				ImGui::LabelText(buffer, buffer);
				sprintf_s(buffer, "Base color %d", i);
				materialsChanged |= ImGui::ColorEdit4(buffer, &materials[i].base_color.r);
				sprintf_s(buffer, "Emissive color %d", i);
				materialsChanged |= ImGui::ColorEdit4(buffer, &materials[i].emissive_color.r);
				sprintf_s(buffer, "Roughness %d", i);
				materialsChanged |= ImGui::SliderFloat(buffer, &materials[i].roughness_factor, 0, 1);
				sprintf_s(buffer, "Metallic %d", i);
				materialsChanged |= ImGui::SliderFloat(buffer, &materials[i].metallic_factor, 0, 1);
			}

			if (materialsChanged) {
				vkutils::cpu_to_gpu(_engineData.allocator, _sceneData.materialBuffer, materials.data(), materials.size() * sizeof(GPUBasicMaterialData));
			}

			ImGui::End();
		}

		ImGui::Begin("Objects");
		{
			for (int i = 0; i < gltf_scene.nodes.size(); i++)
			{
				sprintf_s(buffer, "Object %d", i);
				glm::vec3 translate = { 0, 0, 0 };
				ImGui::DragFloat3(buffer, &translate.x, -1, 1);
				gltf_scene.nodes[i].world_matrix = glm::translate(translate) * gltf_scene.nodes[i].world_matrix;
			}
		}
		ImGui::End();

		ImGui::Begin("Performance");
		{
			if (_engineData.renderGraph->vkTimer.result) {
				float totalTime = 0;
				for (int i = 0; i < _engineData.renderGraph->vkTimer.count; i++) {
					float time = (_engineData.renderGraph->vkTimer.times[i * 2 + 1] - _engineData.renderGraph->vkTimer.times[i * 2]) / 1000000.0;
					totalTime += time;
					sprintf_s(buffer, "%s: %.2f ms", _engineData.renderGraph->vkTimer.names[i].c_str(), time);
					ImGui::Text(buffer);
				}
				ImGui::Separator();
				sprintf_s(buffer, "Total (GPU): %.2f ms", totalTime);
				ImGui::Text(buffer);
			}
		}
		ImGui::End();

		ImGui::Render();
	}
}

void VulkanEngine::draw()
{
	//wait until the gpu has finished rendering the last frame. Timeout of 1 second
	VK_CHECK(vkWaitForFences(_engineData.device, 1, &_renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_engineData.device, 1, &_renderFence));

	//now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
	VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

	//request image from the swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_engineData.device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapchainImageIndex));

	constexpr glm::vec3 UP = glm::vec3(0, 1, 0);
	constexpr glm::vec3 RIGHT = glm::vec3(1, 0, 0);
	constexpr glm::vec3 FORWARD = glm::vec3(0, 0, 1);
	auto res = glm::mat4{ 1 };

	float fov = glm::radians(cameraConfig.fov);

	if (useSceneCamera) {
		res = glm::scale(glm::vec3(_sceneScale)) * gltf_scene.cameras[0].world_matrix;
		camera.pos = gltf_scene.cameras[0].eye * _sceneScale;
		fov = gltf_scene.cameras[0].cam.perspective.yfov;
	}
	else {
		res = glm::translate(res, camera.pos);
		res = glm::rotate(res, glm::radians(camera.rotation.y), UP);
		res = glm::rotate(res, glm::radians(camera.rotation.x), RIGHT);
		res = glm::rotate(res, glm::radians(camera.rotation.z), FORWARD);
	}

	auto view = glm::inverse(res);
	glm::mat4 projection = glm::perspective(fov, ((float)_windowExtent.width) / _windowExtent.height, 0.1f, 1000.0f);
	projection[1][1] *= -1;

	_camData.prevViewproj = _camData.viewproj;
	_camData.viewproj = projection * view;
	_camData.viewprojInverse = glm::inverse(_camData.viewproj);
	_camData.cameraPos = glm::vec4(camera.pos.x, camera.pos.y, camera.pos.z, 1.0);

	_camData.lightmapInputSize = {(float) gltf_scene.lightmap_width, (float) gltf_scene.lightmap_height};
	_camData.lightmapTargetSize = {diffuseIllumination._lightmapExtent.width, diffuseIllumination._lightmapExtent.height};

	_camData.frameCount = _frameNumber;

	glm::vec3 lightInvDir = glm::vec3(_camData.lightPos); 
	float radius = gltf_scene.m_dimensions.max.x > gltf_scene.m_dimensions.max.y ? gltf_scene.m_dimensions.max.x : gltf_scene.m_dimensions.max.y;
	radius = radius > gltf_scene.m_dimensions.max.z ? radius : gltf_scene.m_dimensions.max.z;
	//radius *= sqrt(2);
	glm::mat4 depthViewMatrix = glm::lookAt(gltf_scene.m_dimensions.center * _sceneScale + lightInvDir * radius, gltf_scene.m_dimensions.center * _sceneScale, glm::vec3(0, 1, 0));
	float maxX = gltf_scene.m_dimensions.min.x * _sceneScale, maxY = gltf_scene.m_dimensions.min.y * _sceneScale, maxZ = gltf_scene.m_dimensions.min.z * _sceneScale;
	float minX = gltf_scene.m_dimensions.max.x * _sceneScale, minY = gltf_scene.m_dimensions.max.y * _sceneScale, minZ = gltf_scene.m_dimensions.max.z * _sceneScale;

	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
				float xCoord = x == 0 ? gltf_scene.m_dimensions.min.x* _sceneScale : gltf_scene.m_dimensions.max.x* _sceneScale;
				float yCoord = y == 0 ? gltf_scene.m_dimensions.min.y* _sceneScale : gltf_scene.m_dimensions.max.y* _sceneScale;
				float zCoord = z == 0 ? gltf_scene.m_dimensions.min.z* _sceneScale : gltf_scene.m_dimensions.max.z* _sceneScale;
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

	shadow._shadowMapData.depthMVP = depthProjectionMatrix * depthViewMatrix;

	prepare_gui();
	
	if (showProbes) {
		diffuseIllumination.debug_draw_probes(_vulkanDebugRenderer, showProbeRays, _sceneScale);
	}
	
	if (showReceivers) {
		diffuseIllumination.debug_draw_receivers(_vulkanDebugRenderer, _sceneScale);
	}

	if (showSpecificReceiver) {
		diffuseIllumination.debug_draw_specific_receiver(_vulkanDebugRenderer, specificCluster, specificReceiver, specificReceiverRaySampleCount, probesEnabled, showSpecificProbeRays, _sceneScale);
	}

	vkutils::cpu_to_gpu(_engineData.allocator, _sceneData.cameraBuffer, &_camData, sizeof(GPUCameraData));
	shadow.prepare_rendering(_engineData);

	void* objectData;
	vmaMapMemory(_engineData.allocator, _sceneData.objectBuffer._allocation, &objectData);
	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

	for (int i = 0; i < gltf_scene.nodes.size(); i++)
	{
		auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
		glm::mat4 scale = glm::mat4{ 1 };
		scale = glm::scale(scale, { _sceneScale, _sceneScale, _sceneScale });
		objectSSBO[i].model = scale * gltf_scene.nodes[i].world_matrix;
		objectSSBO[i].material_id = mesh.material_idx;
	}
	vmaUnmapMemory(_engineData.allocator, _sceneData.objectBuffer._allocation);

	_vulkanRaytracing.build_tlas(gltf_scene, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, true);

	VkCommandBuffer cmd = _mainCommandBuffer;
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	{
		shadow.render(_engineData, _sceneData, [&](VkCommandBuffer cmd) {
			draw_objects(cmd);
		});
		gbuffer.render(_engineData, _sceneData, [&](VkCommandBuffer cmd) {
			draw_objects(cmd);
		});
		if (enableGroundTruthDiffuse) {
			diffuseIllumination.render_ground_truth(cmd, _engineData, _sceneData, shadow, brdfUtils);
		}
		else {
			diffuseIllumination.render(cmd, _engineData, _sceneData, shadow, brdfUtils, [&](VkCommandBuffer cmd) {
				draw_objects(cmd);
			}, useRealtimeRaycast);
		}
		glossyIllumination.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination, brdfUtils);

		if (_camData.useStochasticSpecular) {
			glossyDenoise.render(_engineData, _sceneData, gbuffer, glossyIllumination);

			if (enableDenoise) {
				deferred.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination, glossyIllumination, brdfUtils, glossyDenoise.get_denoised_binding());
			}
			else {
				deferred.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination, glossyIllumination, brdfUtils, glossyIllumination._glossyReflectionsColorImageBinding);
			}
		}
		else {
			deferred.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination, glossyIllumination, brdfUtils, glossyIllumination._glossyReflectionsColorImageBinding);
		}

		VkClearValue clearValue;
		clearValue.color = { { 1.0f, 1.0f, 1.0f, 1.0f } };

		_engineData.renderGraph->add_render_pass(
			{
				.name = "PresentPass",
				.pipelineType = Vrg::PipelineType::RASTER_TYPE,
				.rasterPipeline = {
					.vertexShader = "../../shaders/fullscreen.vert.spv",
					.fragmentShader = "../../shaders/gamma.frag.spv",
					.size = _windowExtent,
					.depthState = { false, false, VK_COMPARE_OP_NEVER },
					.cullMode = Vrg::CullMode::NONE,
					.blendAttachmentStates = {
						vkinit::color_blend_attachment_state(),
					},
					.colorOutputs = {
						{_swapchainBindings[swapchainImageIndex], clearValue, true},
					},

				},
				.reads = {
					{0, deferred._deferredColorImageBinding}
				},
				.execute = [&](VkCommandBuffer cmd) {
					vkCmdDraw(cmd, 3, 1, 0, 0);
					//_vulkanDebugRenderer.render(cmd, _sceneDescriptors.globalDescriptor);
					ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
				}
			}
		);
	}

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
	{
		_engineData.renderGraph->execute(cmd);
	}
	
	VK_CHECK(vkEndCommandBuffer(cmd));

	//prepare the submission to the queue. 
	//we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	//we will signal the _renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &_presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &_renderSemaphore;

	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(_engineData.graphicsQueue, 1, &submit, _renderFence));

	//prepare present
	// this will put the image we just rendered to into the visible window.
	// we want to wait on the _renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo = vkinit::present_info();

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &_renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_engineData.graphicsQueue, &presentInfo));
	
	if (screenshot) {
		screenshot = false;
		std::time_t screenshot = std::time(0);
		sprintf_s(buffer, "screenshots/%d-%s.png", screenshot, customName);
		vkutils::screenshot(&_engineData, buffer, _swapchainImages[swapchainImageIndex], _windowExtent);
		ScreenshotSaveData saveData = { camera, _camData.lightPos };
		sprintf_s(buffer, "screenshots/%d-%s.cam", screenshot, customName);
		save_binary(buffer, &saveData, sizeof(ScreenshotSaveData));
		memset(customName, 0, sizeof(customName));
	}

	//increase the number of frames drawn
	_frameNumber++;
	_camData.glossyFrameCount++;

	//TODO: Enable
	_engineData.renderGraph->vkTimer.get_results(_engineData);
}

void VulkanEngine::run()
{
	SDL_Event e;
	bool bQuit = false;
	bool onGui = true;
	SDL_SetRelativeMouseMode((SDL_bool)!onGui);
	//main loop
	while (!bQuit)
	{
		bool moved = false;

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

				camera.rotation.x += -cameraConfig.rotationSpeed * (float)motion.yrel;
				camera.rotation.y += -cameraConfig.rotationSpeed * (float)motion.xrel;
				moved = true;
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

		float speed = cameraConfig.speed;

		if (!onGui) {
			if (keys[SDL_SCANCODE_LSHIFT]) {
				speed *= 4;
			}
			if (keys[SDL_SCANCODE_W])
			{
				camera.pos += front * speed;
				moved = true;
			}
			if (keys[SDL_SCANCODE_S])
			{
				camera.pos -= front * speed;
				moved = true;
			}
			if (keys[SDL_SCANCODE_A]) {
				camera.pos -= glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f))) * speed;
				moved = true;
			}
			if (keys[SDL_SCANCODE_D]) {
				camera.pos += glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f))) * speed;
				moved = true;
			}
			if (keys[SDL_SCANCODE_P]) {
				screenshot = true;
			}
		}
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL2_NewFrame(_window);

		ImGui::NewFrame();
		draw();
	}
}

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder;

	//make the vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.require_api_version(1, 3, 0)
		.build();

	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance 
	_instance = vkb_inst.instance;
	_debug_messenger = vkb_inst.debug_messenger;

	SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

	VkPhysicalDeviceFeatures physicalDeviceFeatures = VkPhysicalDeviceFeatures();
	physicalDeviceFeatures.fillModeNonSolid = true;
	physicalDeviceFeatures.samplerAnisotropy = VK_TRUE;
	physicalDeviceFeatures.shaderInt64 = true;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR featureRt = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	VkPhysicalDeviceAccelerationStructureFeaturesKHR featureAccel = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	VkPhysicalDeviceVulkan13Features features13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
	VkPhysicalDeviceVulkan12Features features12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	VkPhysicalDeviceVulkan11Features features11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };

	VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };

	features11.shaderDrawParameters = VK_TRUE;
	features11.pNext = &features12;

	features12.bufferDeviceAddress = true;
	features12.hostQueryReset = true;
	features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	features12.runtimeDescriptorArray = VK_TRUE;
	features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
	features12.pNext = &dynamic_rendering_feature;

	features13.dynamicRendering = true;
	features13.pNext = &featureRt;
	dynamic_rendering_feature.dynamicRendering = true;
	dynamic_rendering_feature.pNext = &featureRt;

	featureRt.rayTracingPipeline = true;
	featureRt.pNext = &featureAccel;

	featureAccel.accelerationStructure = true;
	featureAccel.pNext = nullptr;

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	auto physicalDeviceSelectionResult = selector
		.set_minimum_version(1, 3)
		.set_surface(_surface)
		.set_required_features(physicalDeviceFeatures)
		.add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
		.add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
		.add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
		.add_required_extension(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME)
		.select();
	
	//printf(physicalDeviceSelectionResult.error().message().c_str());

	auto physicalDevice = physicalDeviceSelectionResult.value();

	//create the final vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	deviceBuilder.add_pNext(&features11);
	vkb::Device vkbDevice = deviceBuilder.build().value();

	_engineData.device = vkbDevice.device;
	_chosenGPU = physicalDevice.physical_device;

	//Get extension pointers
	load_VK_EXTENSIONS(_instance, vkGetInstanceProcAddr, _engineData.device, vkGetDeviceProcAddr);

	// use vkbootstrap to get a Graphics queue
	_engineData.graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_engineData.graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	_engineData.computeQueue = vkbDevice.get_queue(vkb::QueueType::compute).value();
	_engineData.computeQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::compute).value();

	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = _chosenGPU;
	allocatorInfo.device = _engineData.device;
	allocatorInfo.instance = _instance;
	allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator(&allocatorInfo, &_engineData.allocator);

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
	vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_engineData.device,_surface };
	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		//.set_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
		.build()
		.value();

	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();

	for (int i = 0; i < _swapchainImages.size(); i++) {
		AllocatedImage allocatedImage;
		allocatedImage.format = vkbSwapchain.image_format;
		allocatedImage._image = _swapchainImages[i];
		_swapchainAllocatedImage.push_back(allocatedImage);
	}

	_swapchainBindings.resize(3);

	for (int i = 0; i < _swapchainImages.size(); i++) {
		_swapchainBindings[i] = _engineData.renderGraph->register_image_view(&_swapchainAllocatedImage[i], {
		.sampler = Vrg::Sampler::NEAREST,
		.baseMipLevel = 0,
		.mipLevelCount = 1
		}, "Swapchain" + std::to_string(i));
	}

	_swachainImageFormat = vkbSwapchain.image_format;

	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_engineData.device, _swapchain, nullptr);
	});

	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};
}

void VulkanEngine::init_commands()
{
	//create a command pool for commands submitted to the graphics queue.
	//we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_engineData.graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	VK_CHECK(vkCreateCommandPool(_engineData.device, &commandPoolInfo, nullptr, &commandPool));

	//allocate the default command buffer that we will use for rendering
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(commandPool, 1);

	VK_CHECK(vkAllocateCommandBuffers(_engineData.device, &cmdAllocInfo, &_mainCommandBuffer));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_engineData.device, commandPool, nullptr);
	});

	//create pool for upload context
	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_engineData.graphicsQueueFamily);
	VK_CHECK(vkCreateCommandPool(_engineData.device, &uploadCommandPoolInfo, nullptr, &_engineData.uploadContext.commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_engineData.device, _engineData.uploadContext.commandPool, nullptr);
	});
}

void VulkanEngine::init_sync_structures()
{
	//create syncronization structures
	//one fence to control when the gpu has finished rendering the frame,
	//and 2 semaphores to syncronize rendering with swapchain
	//we want the fence to start signalled so we can wait on it on the first frame
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	{
		VK_CHECK(vkCreateFence(_engineData.device, &fenceCreateInfo, nullptr, &_renderFence));

		//enqueue the destruction of the fence
		_mainDeletionQueue.push_function([=]() {
			vkDestroyFence(_engineData.device, _renderFence, nullptr);
			});


		VK_CHECK(vkCreateSemaphore(_engineData.device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
		VK_CHECK(vkCreateSemaphore(_engineData.device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));

		//enqueue the destruction of semaphores
		_mainDeletionQueue.push_function([=]() {
			vkDestroySemaphore(_engineData.device, _presentSemaphore, nullptr);
			vkDestroySemaphore(_engineData.device, _renderSemaphore, nullptr);
			});
	}

	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();
	VK_CHECK(vkCreateFence(_engineData.device, &uploadFenceCreateInfo, nullptr, &_engineData.uploadContext.fence));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_engineData.device, _engineData.uploadContext.fence, nullptr);
	});
}

void VulkanEngine::init_descriptor_pool()
{
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
	vkCreateDescriptorPool(_engineData.device, &pool_info, nullptr, &_engineData.descriptorPool);

	_mainDeletionQueue.push_function([&]() {
		vkDestroyDescriptorPool(_engineData.device, _engineData.descriptorPool, nullptr);
		});
}

void VulkanEngine::init_descriptors()
{
	const int MAX_OBJECTS = 10000;
	
	_sceneData.objectBuffer = vkutils::create_buffer(_engineData.allocator, sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_sceneData.cameraBuffer = vkutils::create_buffer(_engineData.allocator, sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_sceneData.objectBufferBinding = _engineData.renderGraph->register_storage_buffer(&_sceneData.objectBuffer, "ObjectBuffer");
	_sceneData.cameraBufferBinding = _engineData.renderGraph->register_uniform_buffer(&_sceneData.cameraBuffer, "CameraBuffer");

	VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);
	VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
	VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);

	VkDescriptorSetLayoutBinding bindings[3] = { tlasBind, sceneDescBind, meshInfoBind };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 3);
	vkCreateDescriptorSetLayout(_engineData.device, &setinfo, nullptr, &_sceneData.raytracingSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(_engineData.descriptorPool, &_sceneData.raytracingSetLayout, 1);

	vkAllocateDescriptorSets(_engineData.device, &allocateInfo, &_sceneData.raytracingDescriptor);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &_vulkanRaytracing.tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = _sceneData.raytracingDescriptor;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	writes.emplace_back(accelerationStructureWrite);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _sceneData.raytracingDescriptor, &_sceneData.sceneDescBuffer._descriptorBufferInfo, 1));
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _sceneData.raytracingDescriptor, &_sceneData.meshInfoBuffer._descriptorBufferInfo, 2));

	vkUpdateDescriptorSets(_engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

}

void VulkanEngine::init_scene()
{
	std::string file_name = "../../assets/cornellFixed.gltf";
	//std::string file_name = "../../assets/cornellsuzanne.gltf";
	//std::string file_name = "../../assets/occluderscene.gltf";
	//std::string file_name = "../../assets/reflection_new.gltf";
	//std::string file_name = "../../assets/shtest.gltf";
	//std::string file_name = "../../assets/bedroom/bedroom.gltf";
	//std::string file_name = "../../assets/livingroom/livingroom.gltf";
	//std::string file_name = "../../assets/picapica/scene.gltf";
	//std::string file_name = "D:/newsponza/combined/sponza.gltf";

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
		GltfAttributes::Texcoord_0 | GltfAttributes::Tangent);

	printf("dimensions: %f %f %f\n", gltf_scene.m_dimensions.size.x, gltf_scene.m_dimensions.size.y, gltf_scene.m_dimensions.size.z);
	
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

		if (gltf_scene.texcoords0.size() > 0) {
			meshDecleration.vertexUvData = &gltf_scene.texcoords0[mesh.vtx_offset];
			meshDecleration.vertexUvStride = sizeof(glm::vec2);
		}

		meshDecleration.indexData = &gltf_scene.indices[mesh.first_idx];
		meshDecleration.indexCount = mesh.idx_count;
		meshDecleration.indexFormat = xatlas::IndexFormat::UInt32;
		xatlas::AddMesh(atlas, meshDecleration);
	}
	
	xatlas::ChartOptions chartOptions = xatlas::ChartOptions();
	//chartOptions.fixWinding = true;
	xatlas::PackOptions packOptions = xatlas::PackOptions();
	packOptions.texelsPerUnit = precalculationInfo.texelSize;
	packOptions.bilinear = true;
	packOptions.padding = 1;
	xatlas::Generate(atlas, chartOptions, packOptions);

	gltf_scene.lightmap_width = atlas->width;
	gltf_scene.lightmap_height = atlas->height;

	std::vector<GltfPrimMesh> prim_meshes;
	std::vector<glm::vec3> positions;
	std::vector<uint32_t> indices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texcoords0;
	std::vector<glm::vec4> tangents;

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
			tangents.push_back(gltf_scene.tangents[atlas->meshes[i].vertexArray[j].xref + orihinal_vtx_offset]);
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
	gltf_scene.tangents.clear();

	gltf_scene.prim_meshes = prim_meshes;
	gltf_scene.positions = positions;
	gltf_scene.indices = indices;
	gltf_scene.normals = normals;
	gltf_scene.texcoords0 = texcoords0;
	gltf_scene.tangents = tangents;

	printf("Generated lightmap uvs, %d x %d\n", atlas->width, atlas->height);
	xatlas::Destroy(atlas);

	/*
	--
	*/
	//materials.push_back(materials[materials.size() - 1]);
	//auto new_node = gltf_scene.nodes[gltf_scene.nodes.size() - 1];
	//auto new_mesh = gltf_scene.prim_meshes[new_node.prim_mesh];
	//new_mesh.material_idx = materials.size() - 1;
	//gltf_scene.prim_meshes.push_back(new_mesh);
	//new_node.prim_mesh = gltf_scene.prim_meshes.size() - 1;
	//gltf_scene.nodes.push_back(new_node);

	/*
	* TODO: After reading the GLTF scene, what I can do is:
	* Create the xatlas
	* Create a new vertex buffer (also normal and tex)
	* Correct the node data
	*/

	_sceneData.vertexBuffer = vkutils::create_upload_buffer(&_engineData, gltf_scene.positions.data(), gltf_scene.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);

	_sceneData.indexBuffer = vkutils::create_upload_buffer(&_engineData, gltf_scene.indices.data(), gltf_scene.indices.size() * sizeof(uint32_t),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);

	_sceneData.normalBuffer = vkutils::create_upload_buffer(&_engineData, gltf_scene.normals.data(), gltf_scene.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_sceneData.tangentBuffer = vkutils::create_upload_buffer(&_engineData, gltf_scene.tangents.data(), gltf_scene.tangents.size() * sizeof(glm::vec4),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_sceneData.texBuffer = vkutils::create_upload_buffer(&_engineData, gltf_scene.texcoords0.data(), gltf_scene.texcoords0.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_sceneData.lightmapTexBuffer = vkutils::create_upload_buffer(&_engineData, gltf_scene.lightmapUVs.data(), gltf_scene.lightmapUVs.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_sceneData.materialBuffer = vkutils::create_upload_buffer(&_engineData, materials.data(), materials.size() * sizeof(GPUBasicMaterialData),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	_sceneData.vertexBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_sceneData.vertexBuffer, VK_FORMAT_R32G32B32_SFLOAT, "VertexBuffer");
	_sceneData.indexBufferBinding = _engineData.renderGraph->register_index_buffer(&_sceneData.indexBuffer, VK_FORMAT_R32_UINT, "IndexBuffer");
	_sceneData.normalBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_sceneData.normalBuffer, VK_FORMAT_R32G32B32_SFLOAT, "NormalBuffer");
	_sceneData.tangentBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_sceneData.tangentBuffer, VK_FORMAT_R32G32B32A32_SFLOAT, "TangentBuffer");
	_sceneData.texBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_sceneData.texBuffer, VK_FORMAT_R32G32_SFLOAT, "TexBuffer");
	_sceneData.lightmapTexBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_sceneData.lightmapTexBuffer, VK_FORMAT_R32G32_SFLOAT, "LightmapTexBuffer");
	_sceneData.materialBufferBinding = _engineData.renderGraph->register_storage_buffer(&_sceneData.materialBuffer, "MaterialBuffer");

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
	vkCreateSampler(_engineData.device, &samplerInfo, nullptr, &blockySampler);
	std::array<uint8_t, 4> nil = { 0, 0, 0, 0 };
	
	if (tmodel.textures.size() == 0) {
		AllocatedImage allocated_image;
		uint32_t mipLevels;
		vkutils::load_image_from_memory(&_engineData, nil.data(), 1, 1, allocated_image, mipLevels);

		VkImageView imageView;
		VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, allocated_image._image, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		vkCreateImageView(_engineData.device, &imageinfo, nullptr, &imageView);

		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = blockySampler;
		imageBufferInfo.imageView = imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		image_infos.push_back(imageBufferInfo);
	}

	for (int i = 0; i < tmodel.textures.size(); i++) {
		
		auto& gltf_img = tmodel.images[tmodel.textures[i].source];
		AllocatedImage allocated_image;
		uint32_t mipLevels;

		if (gltf_img.image.size() == 0 || gltf_img.width == -1 || gltf_img.height == -1) {
			vkutils::load_image_from_memory(&_engineData, nil.data(), 1, 1, allocated_image, mipLevels);
		}
		else {
			vkutils::load_image_from_memory(&_engineData, gltf_img.image.data(), gltf_img.width, gltf_img.height, allocated_image, mipLevels);
		}
	
		VkImageView imageView;
		VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, allocated_image._image, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		vkCreateImageView(_engineData.device, &imageinfo, nullptr, &imageView);

		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = blockySampler;
		imageBufferInfo.imageView = imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		image_infos.push_back(imageBufferInfo);
	}

	// TEXTURE DESCRIPTOR
	{
		VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT, 0);
		textureBind.descriptorCount = image_infos.size();
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(&textureBind, 1);
		vkCreateDescriptorSetLayout(_engineData.device, &setinfo, nullptr, &_sceneData.textureSetLayout);

		VkDescriptorSetAllocateInfo allocInfo =
			vkinit::descriptorset_allocate_info(_engineData.descriptorPool, &_sceneData.textureSetLayout, 1);

		vkAllocateDescriptorSets(_engineData.device, &allocInfo, &_sceneData.textureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _sceneData.textureDescriptor, image_infos.data(), 0, image_infos.size());

		vkUpdateDescriptorSets(_engineData.device, 1, &textures, 0, nullptr);
	}

	_vulkanRaytracing.convert_scene_to_vk_geometry(gltf_scene, _sceneData.vertexBuffer, _sceneData.indexBuffer);
	_vulkanRaytracing.build_blas(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
	_vulkanRaytracing.build_tlas(gltf_scene, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, false);

	GPUSceneDesc desc = {};
	VkBufferDeviceAddressInfo info = { };
	info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

	info.buffer = _sceneData.vertexBuffer._buffer;
	desc.vertexAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

	info.buffer = _sceneData.normalBuffer._buffer;
	desc.normalAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

	info.buffer = _sceneData.texBuffer._buffer;
	desc.uvAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

	info.buffer = _sceneData.indexBuffer._buffer;
	desc.indexAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

	info.buffer = _sceneData.lightmapTexBuffer._buffer;
	desc.lightmapUvAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

	GPUMeshInfo* dataMesh = new GPUMeshInfo[gltf_scene.prim_meshes.size()];
	for (int i = 0; i < gltf_scene.prim_meshes.size(); i++) {
		dataMesh[i].indexOffset = gltf_scene.prim_meshes[i].first_idx;
		dataMesh[i].vertexOffset = gltf_scene.prim_meshes[i].vtx_offset;
		dataMesh[i].materialIndex = gltf_scene.prim_meshes[i].material_idx;
	}

	_sceneData.sceneDescBuffer = vkutils::create_upload_buffer(&_engineData, &desc, sizeof(GPUSceneDesc), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_sceneData.meshInfoBuffer = vkutils::create_upload_buffer(&_engineData, dataMesh, sizeof(GPUMeshInfo) * gltf_scene.prim_meshes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	delete[] dataMesh;
}

void VulkanEngine::init_imgui()
{
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
	VK_CHECK(vkCreateDescriptorPool(_engineData.device, &pool_info, nullptr, &imguiPool));


	// 2: initialize imgui library

	//this initializes the core structures of imgui
	ImGui::CreateContext();

	ImGui::StyleColorsDark();

	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	//this initializes imgui for SDL
	ImGui_ImplSDL2_InitForVulkan(_window);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = _instance;
	init_info.PhysicalDevice = _chosenGPU;
	init_info.Device = _engineData.device;
	init_info.Queue = _engineData.graphicsQueue;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.UseDynamicRendering = true;
	init_info.ColorAttachmentFormat = _swachainImageFormat;

	ImGui_ImplVulkan_Init(&init_info, nullptr);

	//execute a gpu command to upload imgui font textures
	vkutils::immediate_submit(&_engineData, [&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
	});

	//clear font textures from cpu data
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	//add the destroy the imgui created structures
	_mainDeletionQueue.push_function([=]() {

		vkDestroyDescriptorPool(_engineData.device, imguiPool, nullptr);
		ImGui_ImplVulkan_Shutdown();
		});
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd)
{
	for (int i = 0; i < gltf_scene.nodes.size(); i++) {
		if (true || gltf_scene.materials[gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh].material_idx].alpha_mode == 0) {
			auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
			vkCmdDrawIndexed(cmd, mesh.idx_count, 1, mesh.first_idx, mesh.vtx_offset, i);
		}
	}
}

void VulkanEngine::init_query_pool()
{
	VkQueryPoolCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	createInfo.pNext = nullptr;
	createInfo.flags = 0;

	createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	createInfo.queryCount = 512; // REVIEW

	VK_CHECK(vkCreateQueryPool(_engineData.device, &createInfo, nullptr, &_engineData.queryPool));
}