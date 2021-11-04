#define TINYGLTF_IMPLEMENTATION
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
#include "vk_texture.h"
#include "vk_shader.h"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"
#include <precalculation.h>
#include <optick.h>

const int MAX_TEXTURES = 75; //TODO: Replace this
constexpr bool bUseValidationLayers = true;

const VkFormat depthMapColorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

void VulkanEngine::init()
{
	OPTICK_START_CAPTURE();

	_shadowMapData.positiveExponent = 40;
	_shadowMapData.negativeExponent = 5;
	_shadowMapData.LightBleedingReduction = 0.0f;
	_shadowMapData.VSMBias = 0.01;
	_camData.lightPos = { 0.1, 1, 0.1, 0.0f };
	_camData.lightColor = { 1.f, 1.f, 1.f, 1.0f };

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
	init_shadowmap_renderpass();
	init_framebuffers();
	init_commands();
	init_sync_structures();
	init_descriptors();
	init_pipelines();

	vulkanCompute.init(_device, _allocator, _computeQueue, _computeQueueFamily);

	init_imgui();
	init_scene();

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

void VulkanEngine::draw()
{
	OPTICK_EVENT();

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
	_camData.projection = projection;
	_camData.view = view;
	_camData.viewproj = projection * view;

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
		ImGui::DragFloat3(buffer, &_camData.lightPos.x);
		sprintf_s(buffer, "Factor");
		ImGui::DragFloat(buffer, &radius);


		sprintf_s(buffer, "Pause Timer");
		ImGui::Checkbox(buffer, &pauseTimer);

		ImGui::Image(shadowMapTextureDescriptor, { 128, 128 });
		ImGui::End();
		
		ImGui::Render();
	}

	cpu_to_gpu(get_current_frame().cameraBuffer, &_camData, sizeof(GPUCameraData));

	cpu_to_gpu(get_current_frame().shadowMapDataBuffer, &_shadowMapData, sizeof(GPUShadowMapData));

	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);
	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

	for (int i = 0; i < gltf_scene.nodes.size(); i++)
	{
		auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
		glm::mat4 scale = glm::mat4{ 1 };
		scale = glm::scale(scale, { sceneScale, sceneScale, sceneScale });
		objectSSBO[i].modelMatrix = scale * gltf_scene.nodes[i].world_matrix;
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
			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_shadowMapRenderPass, _shadowMapExtent, _shadowMapFramebuffer);

			rpInfo.clearValueCount = 2;
			VkClearValue clearValues[] = { clearValue, depthClear };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipelineLayout, 0, 1, &get_current_frame().shadowMapDataDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _shadowMapPipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);

			draw_objects(cmd);

			vkCmdEndRenderPass(cmd);
		}

		// DEFAULT SCENE RENDERING
		{
			VkClearValue clearValue;
			clearValue.color = { { 1.0f, 1.0f, 1.0f, 1.0f } };

			VkClearValue depthClear;
			depthClear.depthStencil.depth = 1.f;

			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass, _windowExtent, _framebuffers[swapchainImageIndex]);

			rpInfo.clearValueCount = 2;
			VkClearValue clearValues[] = { clearValue, depthClear };
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 0, 1, &get_current_frame().globalDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 2, 1, &textureDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 3, 1, &materialDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 4, 1, &shadowMapTextureDescriptor, 0, nullptr);

			draw_objects(cmd);
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
			//finalize the render pass
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

		float speed = 0.5f;

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
	physicalDeviceFeatures.samplerAnisotropy = VK_TRUE;

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 2)
		.set_surface(_surface)
		.set_required_features(physicalDeviceFeatures)
		.add_required_extension("VK_KHR_shader_non_semantic_info")
		.select()
		.value();

	//create the final vulkan device

	vkb::DeviceBuilder deviceBuilder{ physicalDevice };

	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;
	_chosenGPU = physicalDevice.physical_device;

	// use vkbootstrap to get a Graphics queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	_computeQueue = vkbDevice.get_queue(vkb::QueueType::compute).value();
	_computeQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::compute).value();

	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = _chosenGPU;
	allocatorInfo.device = _device;
	allocatorInfo.instance = _instance;
	vmaCreateAllocator(&allocatorInfo, &_allocator);

	vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);
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

void VulkanEngine::init_shadowmap_renderpass()
{
	OPTICK_EVENT();

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = depthMapColorFormat;
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
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
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

	VK_CHECK(vkCreateRenderPass(_device, &renderPassCreateInfo, nullptr, &_shadowMapRenderPass));
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

	// Create shadowmap framebuffer and sampler
	{
		VkExtent3D depthImageExtent3D = {
			_shadowMapExtent.width,
			_shadowMapExtent.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(depthMapColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_shadowMapColorImage._image, &_shadowMapColorImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(depthMapColorFormat, _shadowMapColorImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
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

		VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		VK_CHECK(vkCreateSampler(_device, &samplerInfo, nullptr, &_shadowMapSampler));

		VkImageView attachments[2] = { _shadowMapColorImageView, _shadowMapDepthImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_shadowMapRenderPass, _shadowMapExtent);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_shadowMapFramebuffer));
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
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = 0;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = (uint32_t)sizes.size();
	pool_info.pPoolSizes = sizes.data();

	vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptorPool);

	//Main set info
	VkDescriptorSetLayoutCreateInfo setinfo = {};
	setinfo.flags = 0;
	setinfo.pNext = nullptr;
	setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setinfo.bindingCount = 1;

	//binding for camera data + shadow map data
	VkDescriptorSetLayoutBinding cameraBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	VkDescriptorSetLayoutBinding shadowMapDataDefaultBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	VkDescriptorSetLayoutBinding bindings[2] = { cameraBind, shadowMapDataDefaultBind };
	setinfo.pBindings = bindings;
	setinfo.bindingCount = 2;
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_globalSetLayout);

	//binding for object data
	VkDescriptorSetLayoutBinding objectBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	setinfo.pBindings = &objectBind;
	setinfo.bindingCount = 1;
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_objectSetLayout);

	//binding for texture data
	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	textureBind.descriptorCount = MAX_TEXTURES;
	setinfo.pBindings = &textureBind;
	setinfo.bindingCount = 1;
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_textureSetLayout);

	//binding for materials
	VkDescriptorSetLayoutBinding materialBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	setinfo.pBindings = &materialBind;
	setinfo.bindingCount = 1;
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_materialSetLayout);

	//binding for shadow map data
	VkDescriptorSetLayoutBinding shadowMapDataBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	setinfo.pBindings = &shadowMapDataBind;
	setinfo.bindingCount = 1;
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_shadowMapDataSetLayout);

	VkDescriptorSetLayoutBinding shadowMapTextureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	setinfo.pBindings = &shadowMapTextureBind;
	setinfo.bindingCount = 1;
	vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_shadowMapTextureSetLayout);


	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		const int MAX_OBJECTS = 10000;
		_frames[i].objectBuffer = vkinit::create_buffer(_allocator, sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].cameraBuffer = vkinit::create_buffer(_allocator, sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].shadowMapDataBuffer = vkinit::create_buffer(_allocator, sizeof(GPUShadowMapData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.pNext = nullptr;
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = _descriptorPool;
		allocInfo.descriptorSetCount = 1;

		allocInfo.pSetLayouts = &_globalSetLayout;
		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].globalDescriptor);

		allocInfo.pSetLayouts = &_objectSetLayout;
		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].objectDescriptor);

		allocInfo.pSetLayouts = &_shadowMapDataSetLayout;
		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].shadowMapDataDescriptor);

		VkDescriptorBufferInfo cameraBufferInfo;
		cameraBufferInfo.buffer = _frames[i].cameraBuffer._buffer;
		cameraBufferInfo.offset = 0;
		cameraBufferInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo objectBufferInfo;
		objectBufferInfo.buffer = _frames[i].objectBuffer._buffer;
		objectBufferInfo.offset = 0;
		objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

		VkDescriptorBufferInfo shadowMapDataBufferInfo;
		shadowMapDataBufferInfo.buffer = _frames[i].shadowMapDataBuffer._buffer;
		shadowMapDataBufferInfo.offset = 0;
		shadowMapDataBufferInfo.range = sizeof(GPUShadowMapData);

		VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &cameraBufferInfo, 0);
		VkWriteDescriptorSet shadowMapDataDefaultWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &shadowMapDataBufferInfo, 1);
		
		VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].objectDescriptor, &objectBufferInfo, 0);
		VkWriteDescriptorSet shadowMapDataWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].shadowMapDataDescriptor, &shadowMapDataBufferInfo, 0);

		VkWriteDescriptorSet setWrites[] = { cameraWrite, shadowMapDataDefaultWrite, objectWrite, shadowMapDataWrite };

		vkUpdateDescriptorSets(_device, 4, setWrites, 0, nullptr);
	}

	//Shadow map texture descriptor
	{
		VkDescriptorImageInfo imageBufferInfo;
		imageBufferInfo.sampler = _shadowMapSampler;
		imageBufferInfo.imageView = _shadowMapColorImageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.pNext = nullptr;
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = _descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &_shadowMapTextureSetLayout;

		vkAllocateDescriptorSets(_device, &allocInfo, &shadowMapTextureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowMapTextureDescriptor, &imageBufferInfo, 0, 1);

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

void VulkanEngine::init_pipelines() {
	OPTICK_EVENT();

	VkShaderModule meshVertShader;
	if (!vkutil::load_shader_module(_device, "../../shaders/default.vert.spv", &meshVertShader))
	{
		assert("Default Vertex Shader Loading Issue");
	}

	VkShaderModule texturedMeshShader;
	if (!vkutil::load_shader_module(_device, "../../shaders/default.frag.spv", &texturedMeshShader))
	{
		assert("Default Vertex Shader Loading Issue");
	}

	VkShaderModule shadowMapVertShader;
	if (!vkutil::load_shader_module(_device, "../../shaders/evsm.vert.spv", &shadowMapVertShader))
	{
		assert("Shadow Vertex Shader Loading Issue");
	}

	VkShaderModule shadowMapFragShader;
	if (!vkutil::load_shader_module(_device, "../../shaders/evsm.frag.spv", &shadowMapFragShader))
	{
		assert("Shadow Fragment Shader Loading Issue");
	}

	//MESH PIPELINE LAYOUT INFO
	{
		VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
		VkDescriptorSetLayout setLayouts[] = { _globalSetLayout, _objectSetLayout, _textureSetLayout, _materialSetLayout, _shadowMapTextureSetLayout };
		pipeline_layout_info.setLayoutCount = 5;
		pipeline_layout_info.pSetLayouts = setLayouts;
		VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));
	}

	//SHADOWMAP PIPELINE LAYOUT INFO
	{
		VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
		VkDescriptorSetLayout setLayouts[] = { _shadowMapDataSetLayout, _objectSetLayout };
		pipeline_layout_info.setLayoutCount = 2;
		pipeline_layout_info.pSetLayouts = setLayouts;
		VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_shadowMapPipelineLayout));
	}

	//build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
	PipelineBuilder pipelineBuilder;

	//vertex input controls how to read vertices from vertex buffers. We aren't using it yet
	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

	//input assembly is the configuration for drawing triangle lists, strips, or individual points.
	//we are just going to draw triangle list
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	//build viewport and scissor from the swapchain extents
	pipelineBuilder._viewport.x = 0.0f;
	pipelineBuilder._viewport.y = 0.0f;
	pipelineBuilder._viewport.width = (float)_windowExtent.width;
	pipelineBuilder._viewport.height = (float)_windowExtent.height;
	pipelineBuilder._viewport.minDepth = 0.0f;
	pipelineBuilder._viewport.maxDepth = 1.0f;

	pipelineBuilder._scissor.offset = { 0, 0 };
	pipelineBuilder._scissor.extent = _windowExtent;

	//configure the rasterizer to draw filled triangles
	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

	//we don't use multisampling, so just run the default one
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

	//a single blend attachment with no blending and writing to RGBA

	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &vkinit::color_blend_attachment_state());

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	//build the mesh pipeline
	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	//add the other shaders
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	//make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, texturedMeshShader));

	pipelineBuilder._pipelineLayout = _meshPipelineLayout;

	//build the mesh triangle pipeline
	_meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

	/*
	* / VVVVVVVVVVVVV SHADOW MAP PIPELINE VVVVVVVVVVVVV
	*/

	pipelineBuilder._shaderStages.clear();
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, shadowMapVertShader));
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, shadowMapFragShader));
	//pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(0, nullptr);

	//build viewport and scissor from the swapchain extents
	pipelineBuilder._viewport.x = 0.0f;
	pipelineBuilder._viewport.y = 0.0f;
	pipelineBuilder._viewport.width = (float)_shadowMapExtent.width;
	pipelineBuilder._viewport.height = (float)_shadowMapExtent.height;
	pipelineBuilder._viewport.minDepth = 0.0f;
	pipelineBuilder._viewport.maxDepth = 1.0f;

	pipelineBuilder._scissor.offset = { 0, 0 };
	pipelineBuilder._scissor.extent = _shadowMapExtent;

	//pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;
	pipelineBuilder._pipelineLayout = _shadowMapPipelineLayout;

	_shadowMapPipeline = pipelineBuilder.build_pipeline(_device, _shadowMapRenderPass);
	
	//destroy all shader modules, outside of the queue
	vkDestroyShaderModule(_device, meshVertShader, nullptr);
	vkDestroyShaderModule(_device, texturedMeshShader, nullptr);
	vkDestroyShaderModule(_device, shadowMapVertShader, nullptr);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, _meshPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);

		vkDestroyPipeline(_device, _shadowMapPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _shadowMapPipelineLayout, nullptr);
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
	
	std::vector<BasicMaterial> materials;

	for (int i = 0; i < gltf_scene.materials.size(); i++) {
		BasicMaterial material = {};
		material.base_color = gltf_scene.materials[i].base_color_factor;
		material.emissive_color = gltf_scene.materials[i].emissive_factor;
		material.metallic_factor = gltf_scene.materials[i].metallic_factor;
		material.roughness_factor = gltf_scene.materials[i].roughness_factor;
		material.texture = gltf_scene.materials[i].base_color_texture;
		material.normal_texture = gltf_scene.materials[i].normal_texture;
		material.metallic_roughness_texture = gltf_scene.materials[i].metallic_rougness_texture;
		materials.push_back(material);
	}

	vertex_buffer = create_upload_buffer(gltf_scene.positions.data(), gltf_scene.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	index_buffer = create_upload_buffer(gltf_scene.indices.data(), gltf_scene.indices.size() * sizeof(uint32_t),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	normal_buffer = create_upload_buffer(gltf_scene.normals.data(), gltf_scene.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	tex_buffer = create_upload_buffer(gltf_scene.texcoords0.data(), gltf_scene.texcoords0.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);	

	material_buffer = create_upload_buffer(materials.data(), materials.size() * sizeof(BasicMaterial),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// MATERIAL DESCRIPTOR
	{
		VkDescriptorSetAllocateInfo materialSetAlloc = {};
		materialSetAlloc.pNext = nullptr;
		materialSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		materialSetAlloc.descriptorPool = _descriptorPool;
		materialSetAlloc.descriptorSetCount = 1;
		materialSetAlloc.pSetLayouts = &_materialSetLayout;

		vkAllocateDescriptorSets(_device, &materialSetAlloc, &materialDescriptor);

		VkDescriptorBufferInfo materialBufferInfo;
		materialBufferInfo.buffer = material_buffer._buffer;
		materialBufferInfo.offset = 0;
		materialBufferInfo.range = sizeof(BasicMaterial) * materials.size();

		VkWriteDescriptorSet materialWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, materialDescriptor, &materialBufferInfo, 0);
		vkUpdateDescriptorSets(_device, 1, &materialWrite, 0, nullptr);
	}
	std::vector<VkDescriptorImageInfo> image_infos;

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	samplerInfo.minLod = 0.0f; // Optional
	samplerInfo.maxLod = FLT_MAX;
	samplerInfo.mipLodBias = 0.0f; // Optional

	samplerInfo.anisotropyEnable = _gpuFeatures.samplerAnisotropy;
	samplerInfo.maxAnisotropy = _gpuFeatures.samplerAnisotropy
		? _gpuProperties.limits.maxSamplerAnisotropy
		: 1.0f;

	VkSampler blockySampler;
	vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);
	std::array<uint8_t, 4> nil = { 0, 0, 0, 0 };

	if (tmodel.images.size() == 0) {
		AllocatedImage allocated_image;
		uint32_t mipLevels;
		vkutil::load_image_from_memory(*this, nil.data(), 1, 1, allocated_image, mipLevels);

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
			vkutil::load_image_from_memory(*this, nil.data(), 1, 1, allocated_image, mipLevels);
		}
		else {
			vkutil::load_image_from_memory(*this, gltf_img.image.data(), gltf_img.width, gltf_img.height, allocated_image, mipLevels);
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
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.pNext = nullptr;
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = _descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &_textureSetLayout;

		vkAllocateDescriptorSets(_device, &allocInfo, &textureDescriptor);

		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureDescriptor, image_infos.data(), 0, image_infos.size());

		vkUpdateDescriptorSets(_device, 1, &textures, 0, nullptr);
	}

	Precalculation precalculation;
	int dimX, dimY, dimZ;
	uint8_t* voxelSpace = precalculation.voxelize(gltf_scene, 0.9f, 10, true);
	Receiver* receivers = precalculation.generate_receivers(128);
	std::vector<glm::vec4> probes = precalculation.place_probes(*this, 10); //N_OVERLAPS
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

AllocatedBuffer VulkanEngine::create_upload_buffer(void* buffer_data, size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	OPTICK_EVENT();

	AllocatedBuffer stagingBuffer = vkinit::create_buffer(_allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	
	cpu_to_gpu(stagingBuffer, buffer_data, size);

	AllocatedBuffer new_buffer = vkinit::create_buffer(_allocator, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memoryUsage);

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
		VkDeviceSize offsets[] = { 0, 0, 0 };
		VkBuffer buffers[] = { vertex_buffer._buffer, normal_buffer._buffer, tex_buffer._buffer };
		vkCmdBindVertexBuffers(cmd, 0, 3, buffers, offsets);
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
	size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
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

void VulkanEngine::cpu_to_gpu(AllocatedBuffer& allocatedBuffer, void* data, size_t size)
{
	void* gpuData;
	vmaMapMemory(_allocator, allocatedBuffer._allocation, &gpuData);
	memcpy(gpuData, data, size);
	vmaUnmapMemory(_allocator, allocatedBuffer._allocation);
}
