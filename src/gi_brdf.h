#pragma once
#include "vk_types.h"

class BRDF {
public:
	void init_images(EngineData& engineData);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors);
	void cleanup(EngineData& engineData);
	VkDescriptorSet _brdfLutTextureDescriptor;
private:
	AllocatedImage _brdfLutImage;
	VkImageView _brdfLutImageView;
};