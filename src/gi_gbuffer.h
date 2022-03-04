#pragma once

#include "vk_types.h"
#include <functional>

struct GbufferData {
	AllocatedImage _gbufferAlbedoMetallicImage;
	AllocatedImage _gbufferNormalMotionImage;
	AllocatedImage _gbufferRoughnessDepthCurvatureMaterialImage;
	AllocatedImage _gbufferUVImage;
	AllocatedImage _gbufferDepthImage;

	VkImageView _gbufferAlbedoMetallicImageView;
	VkImageView _gbufferNormalMotionImageView;
	VkImageView _gbufferRoughnessDepthCurvatureMaterialImageView;
	VkImageView _gbufferUVImageView;
	VkImageView _gbufferDepthImageView;

	VkDescriptorSet _gbufferDescriptorSet;
	VkFramebuffer _gbufferFrameBuffer;
};

class GBuffer {
public:
	void init_render_pass(EngineData& engineData);
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void init_descriptors(EngineData& engineData);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, bool rebuild = false);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, std::function<void(VkCommandBuffer cmd)>&& function);
	void cleanup(EngineData& engineData);

	VkDescriptorSet getGbufferCurrentDescriptorSet();
	VkDescriptorSet getGbufferPreviousFrameDescriptorSet();
	VkDescriptorSetLayout _gbufferDescriptorSetLayout;
private:
	int _currFrame = 0;
	VkRenderPass _gbufferRenderPass;

	VkExtent2D _imageSize;

	VkPipeline _gbufferPipeline;
	VkPipelineLayout _gbufferPipelineLayout;
	GbufferData _gbufferdata[2];
};