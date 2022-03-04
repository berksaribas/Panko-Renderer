#pragma once
#include "vk_types.h"

class BRDF {
public:
	void init_images(EngineData& engineData);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors);
	void cleanup(EngineData& engineData);
	VkDescriptorSet _brdfLutTextureDescriptor;

	VkDescriptorSetLayout _blueNoiseDescriptorSetLayout;
	VkDescriptorSet _blueNoiseDescriptor;
private:
	AllocatedImage _brdfLutImage;
	VkImageView _brdfLutImageView;

	AllocatedImage _scramblingRanking1sppImage;
	VkImageView _scramblingRanking1sppImageView;

	AllocatedImage _sobolImage;
	VkImageView _sobolImageView;
};