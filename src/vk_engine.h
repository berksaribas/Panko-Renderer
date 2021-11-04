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

#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			printf("Detected Vulkan error: %s\n", err); \
			abort();                                                \
		}                                                           \
	} while (0)

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

struct GPUCameraData {
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewproj;

	glm::vec4 lightPos;
	glm::vec4 lightColor;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
	int material_id;
	float pad0, pad1, pad2;
};

struct GPUShadowMapData {
	glm::mat4 depthMVP;
	float positiveExponent; //40
	float negativeExponent; //5
	float LightBleedingReduction; //0
	float VSMBias; //0.01
};

struct FrameData {
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	AllocatedBuffer cameraBuffer;
	VkDescriptorSet globalDescriptor;

	AllocatedBuffer objectBuffer;
	VkDescriptorSet objectDescriptor;

	AllocatedBuffer shadowMapDataBuffer;
	VkDescriptorSet shadowMapDataDescriptor;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber{ 0 };

	VkExtent2D _windowExtent{ 1600 , 900 };

	struct SDL_Window* _window{ nullptr };

	VkPhysicalDeviceFeatures _gpuFeatures;
	VkPhysicalDeviceProperties _gpuProperties;

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkQueue _computeQueue;
	uint32_t _computeQueueFamily;

	VkSurfaceKHR _surface;
	VkSwapchainKHR _swapchain;
	VkFormat _swachainImageFormat;

	std::vector<VkFramebuffer> _framebuffers;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	VulkanCompute vulkanCompute;

	//the format for the depth image
	VkFormat _depthFormat;

	DeletionQueue _mainDeletionQueue;
	
	VmaAllocator _allocator;
	VkDescriptorPool _descriptorPool;

	CommandContext _uploadContext;

	GltfScene gltf_scene;
	Camera camera = { glm::vec3(0, 0, 28.5), glm::vec3(0, 0, 0) };

	AllocatedBuffer vertex_buffer, index_buffer, normal_buffer;

	/* DEFAULT RENDERING VARIABLES */
	VkRenderPass _renderPass;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	VkPipeline _meshPipeline;
	VkPipelineLayout _meshPipelineLayout;

	FrameData _frames[FRAME_OVERLAP];
	AllocatedBuffer tex_buffer, material_buffer;

	VkDescriptorSet textureDescriptor;
	VkDescriptorSet materialDescriptor;

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _materialSetLayout;
	VkDescriptorSetLayout _textureSetLayout;

	GPUCameraData _camData;
	float sceneScale = 0.3f;

	/* SHADOW MAP VARIABLES */
	VkExtent2D _shadowMapExtent{ 4096 , 4096 };

	VkRenderPass _shadowMapRenderPass;

	AllocatedImage _shadowMapDepthImage;
	VkImageView _shadowMapDepthImageView;

	AllocatedImage _shadowMapColorImage;
	VkImageView _shadowMapColorImageView;

	VkSampler _shadowMapSampler;
	VkFramebuffer _shadowMapFramebuffer;

	VkDescriptorSet shadowMapTextureDescriptor;

	VkPipeline _shadowMapPipeline;
	VkPipelineLayout _shadowMapPipelineLayout;

	VkDescriptorSetLayout _shadowMapDataSetLayout;
	VkDescriptorSetLayout _shadowMapTextureSetLayout;

	GPUShadowMapData _shadowMapData;

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

	FrameData& get_current_frame();

	size_t pad_uniform_buffer_size(size_t originalSize);

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	void cpu_to_gpu(AllocatedBuffer& allocatedBuffer, void* data, size_t size);
private:

	void init_vulkan();

	void init_swapchain();

	void init_default_renderpass();

	void init_shadowmap_renderpass();

	void init_framebuffers();

	void init_commands();

	void init_sync_structures();

	void init_descriptors();

	void init_pipelines();

	AllocatedBuffer create_upload_buffer(void* buffer_data, size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	void init_scene();

	void init_imgui();
};