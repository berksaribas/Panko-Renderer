// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>
#include <deque>
#include <functional>
#include "vk_mesh.h"
#include <glm/glm.hpp>
#include <unordered_map>
#include <gltf_scene.hpp>
#include <vk_compute.h>
#include "vk_mem_alloc.h"
#include "VkBootstrap.h"
#include <vk_raytracing.h>
#define RAYTRACING
#include "../shaders/common.glsl"
#undef RAYTRACING
#include <vk_debug_renderer.h>

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call the function
		}

		deletors.clear();
	}
};

struct Camera {
	glm::vec3 pos;
	glm::vec3 rotation;
};

class VulkanEngine {
public:
	EngineData _engineData;
	VulkanCompute _vulkanCompute;
	VulkanRaytracing _vulkanRaytracing;
	VulkanDebugRenderer _vulkanDebugRenderer;

	bool _isInitialized{ false };
	int _frameNumber{ 0 };

	VkExtent2D _windowExtent{ 1600 , 900 };

	struct SDL_Window* _window{ nullptr };

	VkPhysicalDeviceFeatures _gpuFeatures;
	VkPhysicalDeviceProperties2 _gpuProperties;
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR _gpuRaytracingProperties;

	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkInstance _instance;

	VkSurfaceKHR _surface;
	VkSwapchainKHR _swapchain;
	VkFormat _swachainImageFormat;

	std::vector<VkFramebuffer> _framebuffers;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	DeletionQueue _mainDeletionQueue;

	GltfScene gltf_scene;
	Camera camera = { glm::vec3(0, 0, 28.5), glm::vec3(0, 0, 0) };

	AllocatedBuffer vertex_buffer, index_buffer, normal_buffer, tex_buffer, material_buffer, lightmap_tex_buffer;
	AllocatedBuffer sceneDescBuffer, meshInfoBuffer;

	/* DEFAULT RENDERING VARIABLES */

	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkRenderPass _renderPass;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	AllocatedBuffer _cameraBuffer, _objectBuffer;
	SceneDescriptors _sceneDescriptors;

	GPUCameraData _camData = {};
	float _sceneScale = 0.3f;


	/* Post processing pipelines */
	VkPipeline _dilationPipeline;
	VkPipelineLayout _dilationPipelineLayout;

	VkPipeline _gammaPipeline;
	VkPipelineLayout _gammaPipelineLayout;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	//our draw function
	void draw_objects(VkCommandBuffer cmd);
private:

	void init_vulkan();

	void init_swapchain();

	void init_default_renderpass();

	void init_colordepth_renderpass();

	void init_color_renderpass();

	void init_framebuffers();

	void init_commands();

	void init_sync_structures();

	void init_descriptor_pool();

	void init_descriptors();

	void init_pipelines(bool rebuild = false);

	void init_scene();

	void init_imgui();
};