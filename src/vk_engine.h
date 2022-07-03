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

struct CameraConfig {
	float fov = 45;
	float speed = 0.1f;
	float rotationSpeed = 0.05f;
};

struct Camera {
	glm::vec3 pos;
	glm::vec3 rotation;
};

struct ScreenshotSaveData {
	Camera camera;
	glm::vec4 lightPos;
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

	VkExtent2D _windowExtent{ 1920 , 1080 };

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
	std::vector<AllocatedImage> _swapchainAllocatedImage;
	std::vector<Vrg::Bindable*> _swapchainBindings;

	GltfScene gltf_scene;
	Camera camera = { glm::vec3(0, 0, 28.5), glm::vec3(0, 0, 0) };

	/* DEFAULT RENDERING VARIABLES */

	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool commandPool;
	VkCommandBuffer _mainCommandBuffer;

	GPUCameraData _camData = {};

	float _sceneScale = 0.3f;

	VkRenderPass _renderPass;

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

	void init_commands();

	void init_sync_structures();

	void init_descriptor_pool();

	void init_descriptors();

	void init_scene();

	void init_imgui();

	void init_query_pool();

	void init_default_renderpass();

	void prepare_gui();
};