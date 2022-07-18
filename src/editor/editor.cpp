#include "editor.h"
#include <vk_utils.h>
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"
#include "vk_rendergraph.h"
#include <filesystem>
#include <gi_shadow.h>
#include <gi_glossy_svgf.h>
#include <glm\gtx\transform.hpp>
#include <gltf_scene.hpp>
#include <file_helper.h>

void Editor::initialize(EngineData& engineData, SDL_Window* window, VkFormat targetFormat)
{
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

	VK_CHECK(vkCreateDescriptorPool(engineData.device, &pool_info, nullptr, &_imguiPool));

	// 2: initialize imgui library

	//this initializes the core structures of imgui
	ImGui::CreateContext();

	ImGui::StyleColorsDark();

	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	//this initializes imgui for SDL
	_window = window;
	ImGui_ImplSDL2_InitForVulkan(window);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = engineData.instance;
	init_info.PhysicalDevice = engineData.physicalDevice;
	init_info.Device = engineData.device;
	init_info.Queue = engineData.graphicsQueue;
	init_info.DescriptorPool = _imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.UseDynamicRendering = true;
	init_info.ColorAttachmentFormat = targetFormat;

	ImGui_ImplVulkan_Init(&init_info, nullptr);

	//execute a gpu command to upload imgui font textures
	vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
	});

	//clear font textures from cpu data
	ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void Editor::retrieve_input(SDL_Event* event)
{
	ImGui_ImplSDL2_ProcessEvent(event);
}

static char buffer[256];

void Editor::prepare(EngineData& engineData)
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL2_NewFrame(_window);
	ImGui::NewFrame();

	//bool demo = true;
	//ImGui::ShowDemoWindow(&demo);
}

void Editor::prepare_camera_settings(EngineData& engineData, GPUCameraData& camData, CameraConfig& camConfig, bool sceneCameraAvailable)
{
	ImGui::Begin("Camera");
	{
		ImGui::ColorEdit4("Clear Color", &camData.clearColor.r);
		//if (gltf_scene.cameras.size() > 0) {
		if (sceneCameraAvailable) {
			ImGui::Checkbox("Use scene camera", &editorSettings.useSceneCamera);
		}
		ImGui::NewLine();

		ImGui::SliderFloat("Camera Fov", &camConfig.fov, 0, 90);
		ImGui::SliderFloat("Camera Speed", &camConfig.speed, 0, 1);
		ImGui::SliderFloat("Camera Rotation Speed", &camConfig.rotationSpeed, 0, 0.5);

		ImGui::End();
	}
}

void Editor::prepare_debug_settings(EngineData& engineData)
{
	ImGui::Begin("Debug");
	{
		static int selected = -1;
		for (int i = 0; i < engineData.renderGraph->bindings.max_size(); i++) {
			auto bindable = engineData.renderGraph->bindings[i];
			if (bindable.isValid()) {
				auto binding = engineData.renderGraph->bindings.get(bindable);
				if (binding->type == Vrg::BindType::IMAGE_VIEW) {
					if (ImGui::Selectable(binding->name.c_str(), selected == i)) {
						selected = i;
						editorSettings.selectedRenderBinding = bindable;
					}
				}
			}
		}

		ImGui::End();
	}
}

void Editor::prepare_performance_settings(EngineData& engineData)
{
	ImGuiIO& io = ImGui::GetIO();

	ImGui::Begin("Performance");
	{
		if (engineData.renderGraph->vkTimer.result) {
			float totalTime = 0;
			for (int i = 0; i < engineData.renderGraph->vkTimer.count; i++) {
				float time = (engineData.renderGraph->vkTimer.times[i * 2 + 1] - engineData.renderGraph->vkTimer.times[i * 2]) / 1000000.0;
				totalTime += time;
				sprintf_s(buffer, "%s: %.2f ms", engineData.renderGraph->vkTimer.names[i].c_str(), time);
				ImGui::Text(buffer);
			}
			ImGui::Separator();
			ImGui::Text("Total (GPU): %.2f ms", totalTime);
			ImGui::Text("Total: %.2f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		}
	}
	ImGui::End();
}

void Editor::prepare_material_settings(EngineData& engineData, SceneData& sceneData, GPUBasicMaterialData* materials, int count)
{
	ImGui::Begin("Materials");
	{
		bool materialsChanged = false;
		for (int i = 0; i < count; i++) {
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
			vkutils::cpu_to_gpu(engineData.allocator, sceneData.materialBuffer, materials, count * sizeof(GPUBasicMaterialData));
		}

		ImGui::End();
	}
}

void Editor::prepare_object_settings(EngineData& engineData, GltfNode* nodes, int count)
{
	ImGui::Begin("Objects");
	{
		for (int i = 0; i < count; i++)
		{
			sprintf_s(buffer, "Object %d", i);
			glm::vec3 translate = { 0, 0, 0 };
			ImGui::DragFloat3(buffer, &translate.x, -1, 1);
			nodes[i].world_matrix = glm::translate(translate) * nodes[i].world_matrix;
		}
	}
	ImGui::End();
}

void Editor::prepare_renderer_settings(EngineData& engineData, GPUCameraData& camData, Shadow& shadow, GlossyDenoise& glossyDenoise, int& frameNumber)
{
	ImGui::Begin("Renderer");
	{
		sprintf_s(buffer, "Rebuild Shaders");
		if (ImGui::Button(buffer)) {
			engineData.renderGraph->rebuild_pipelines();
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
		ImGui::DragFloat3(buffer, &camData.lightPos.x);

		sprintf_s(buffer, "Light Color");
		ImGui::ColorEdit3(buffer, &camData.lightColor.x);

		ImGui::NewLine();

		sprintf_s(buffer, "Indirect Diffuse");
		ImGui::Checkbox(buffer, (bool*)&camData.indirectDiffuse);

		if (camData.indirectDiffuse) {
			sprintf_s(buffer, "Ground Truth");
			if (ImGui::Checkbox(buffer, &editorSettings.enableGroundTruthDiffuse)) {
				frameNumber = 0;
			}
		}

		sprintf_s(buffer, "Use realtime probe raycasting");
		ImGui::Checkbox(buffer, (bool*)&editorSettings.useRealtimeRaycast);

		sprintf_s(buffer, "Indirect Specular");
		ImGui::Checkbox(buffer, (bool*)&camData.indirectSpecular);

		if (camData.indirectSpecular) {
			sprintf_s(buffer, "Use Stochastic Raytracing");
			if (ImGui::Checkbox(buffer, (bool*)&camData.useStochasticSpecular)) {
				///_frameNumber = 0;
			}

			if (camData.useStochasticSpecular) {
				sprintf_s(buffer, "Enable SVGF denoising");
				if (ImGui::Checkbox(buffer, &editorSettings.enableDenoise)) {
					camData.glossyFrameCount = 0;
					camData.glossyDenoise = editorSettings.enableDenoise;
				}

				if (editorSettings.enableDenoise) {
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
		ImGui::Checkbox("Show Probes", &editorSettings.showProbes);
		if (editorSettings.showProbes) {
			ImGui::Checkbox("Show Probe Rays", &editorSettings.showProbeRays);
		}
		ImGui::Checkbox("Show Receivers", &editorSettings.showReceivers);
		ImGui::Checkbox("Show Specific Receivers", &editorSettings.showSpecificReceiver);

		/*
		if (editorSettings.showSpecificReceiver) {
			ImGui::SliderInt("Cluster: ", &editorSettings.specificCluster, 0, precalculationLoadData.aabbClusterCount - 1);
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
		*/

		ImGui::NewLine();

		ImGui::InputText("Custom name", editorSettings.customName, sizeof(editorSettings.customName));
		if (ImGui::Button("Screenshot")) {
			editorSettings.screenshot = true;
		}

		if (ImGui::Button("Show Saved Data")) {
			std::string path = "./screenshots/";
			editorSettings.load_files.clear();
			for (const auto& entry : std::filesystem::directory_iterator(path)) {
				auto& path = entry.path();
				if (path.extension().generic_string().compare(".cam") == 0) {
					editorSettings.load_files.push_back(path.generic_string());
				}
			}
			editorSettings.showFileList = true;
		}
		
		ImGui::End();
	}
}

void Editor::prepare_screenshot_loader_settings(GPUCameraData& camData, Camera& camera)
{
	if (editorSettings.showFileList) {
		ImGui::Begin("Files", &editorSettings.showFileList);
		ImGui::ListBoxHeader("");
		for (int i = 0; i < editorSettings.load_files.size(); i++) {
			if (ImGui::Selectable(editorSettings.load_files[i].c_str(), i == editorSettings.selected_file)) {
				editorSettings.selected_file = i;
			}
		}
		ImGui::ListBoxFooter();
		if (ImGui::Button("Load")) {
			ScreenshotSaveData saveData = {};
			load_binary(editorSettings.load_files[editorSettings.selected_file], &saveData, sizeof(ScreenshotSaveData));
			camera = saveData.camera;
			camData.lightPos = saveData.lightPos;
		}
		ImGui::End();
	}
}

void Editor::render(VkCommandBuffer cmd)
{
	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void Editor::destroy(EngineData& engineData)
{
	vkDestroyDescriptorPool(engineData.device, _imguiPool, nullptr);
	ImGui_ImplVulkan_Shutdown();
}
