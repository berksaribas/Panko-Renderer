#pragma once
#include "vk_types.h"
#include <gi_gbuffer.h>
#include <gi_glossy.h>
#include <vk_compute.h>

struct TemporalData {
	AllocatedImage colorImage;
	AllocatedImage momentsImage;

	VkImageView colorImageView;
	VkImageView momentsImageView;

	VkDescriptorSet temporalSampleDescriptor;
	VkDescriptorSet temporalStorageDescriptor;
};

struct AtrousData {
	AllocatedImage pingImage;
	VkImageView pingImageView;

	VkDescriptorSet atrousSampleDescriptor;
	VkDescriptorSet atrousStorageDescriptor;
};

class GlossyDenoise {
public:
	void init(VulkanCompute* vulkanCompute);
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, bool rebuild = false);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, GlossyIllumination& glossyIllumination);
	VkDescriptorSet getDenoisedDescriptor();
	void cleanup();
	int num_atrous = 4;
private:
	VulkanCompute* _vulkanCompute;

	int _currFrame = 0;
	VkExtent2D _imageSize;

	TemporalData _temporalData[2];
	VkDescriptorSetLayout _temporalStorageDescriptorSetLayout;
	VkDescriptorSetLayout _temporalSampleDescriptorSetLayout;

	AtrousData _atrousData[2];
	VkDescriptorSetLayout _atrousPingStorageDescriptorSetLayout;

	ComputeInstance _temporalFilter = {};
	ComputeInstance _atrousFilter = {};
};
