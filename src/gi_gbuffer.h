#pragma once

#include "vk_types.h"
#include <functional>

class GBuffer {
public:
	void init_render_pass(EngineData& engineData);
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void init_descriptors(EngineData& engineData);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, bool rebuild = false);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, std::function<void(VkCommandBuffer cmd)>&& function);
	void cleanup(EngineData& engineData);

	VkDescriptorSet _gbufferDescriptorSet;
	VkDescriptorSetLayout _gbufferDescriptorSetLayout;
private:
	VkRenderPass _gbufferRenderPass;
	VkFramebuffer _gbufferFrameBuffer;

	VkExtent2D _imageSize;

	AllocatedImage _gbufferPosMaterialImage;
	AllocatedImage _gbufferNormalImage;
	AllocatedImage _gbufferUvImage;
	AllocatedImage _gbufferDepthImage;

	VkImageView _gbufferPosMaterialImageView;
	VkImageView _gbufferNormalImageView;
	VkImageView _gbufferUvImageView;
	VkImageView _gbufferDepthImageView;

	VkPipeline _gbufferPipeline;
	VkPipelineLayout _gbufferPipelineLayout;
};