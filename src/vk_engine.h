#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/glm.hpp>
#include <gltf_scene.hpp>
#include <vk_compute.h>
#include <vk_raytracing.h>
#define RAYTRACING
#include "../shaders/common.glsl"
#include <deque>
#include "vk_debug_renderer.h"
#undef RAYTRACING

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

class VulkanEngine {
public:
	EngineData _engineData;
	SceneData _sceneData;

	VulkanCompute _vulkanCompute;
	VulkanRaytracing _vulkanRaytracing;
	VulkanDebugRenderer _vulkanDebugRenderer;

	DeletionQueue _mainDeletionQueue;

	bool _isInitialized{ false };
	int _frameNumber{ 0 };

	VkExtent2D _renderResolution{ 256 , 144 };
	VkExtent2D _displayResolution{ 1280 , 720 };

	struct SDL_Window* _window{ nullptr };

	VkPhysicalDeviceFeatures _gpuFeatures;
	VkPhysicalDeviceProperties2 _gpuProperties;
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR _gpuRaytracingProperties;

	VkDebugUtilsMessengerEXT _debug_messenger;

	VkSurfaceKHR _surface;
	VkSwapchainKHR _swapchain;
	VkFormat _swachainImageFormat;

	std::vector<VkFramebuffer> _framebuffers;
	std::vector<VkImage> _swapchainImages;
	std::vector<AllocatedImage> _swapchainAllocatedImage;
	std::vector<Handle<Vrg::Bindable>> _swapchainBindings;

	GltfScene gltf_scene;

	/* DEFAULT RENDERING VARIABLES */

	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool commandPool;
	VkCommandBuffer _mainCommandBuffer;

	GPUCameraData _camData = {};

	float _sceneScale = 0.3f;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw(double deltaTime);

	//run main loop
	void run();

	//our draw function
	void draw_objects(VkCommandBuffer cmd);
private:

	void init_vulkan();

	void init_swapchain();

	void init_commands();

	void init_sync_structures();

	void init_descriptor_pool();

	void init_descriptors();

	void init_scene();

	void init_query_pool();

	void prepare_gui();
};