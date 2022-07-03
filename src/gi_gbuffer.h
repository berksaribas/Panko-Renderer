#pragma once

#include "vk_types.h"
#include <functional>

struct GbufferData {
	AllocatedImage gbufferAlbedoMetallicImage;
	AllocatedImage gbufferNormalMotionImage;
	AllocatedImage gbufferRoughnessDepthCurvatureMaterialImage;
	AllocatedImage gbufferUVImage;
	AllocatedImage gbufferDepthImage;

	Vrg::Bindable* albedoMetallicBinding;
	Vrg::Bindable* normalMotionBinding;
	Vrg::Bindable* roughnessDepthCurvatureMaterialBinding;
	Vrg::Bindable* uvBinding;
	Vrg::Bindable* depthBinding;
};

class GBuffer {
public:
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void render(EngineData& engineData, SceneData& sceneData, std::function<void(VkCommandBuffer cmd)>&& function);

	VkDescriptorSetLayout _gbufferDescriptorSetLayout;
	GbufferData* get_current_frame_data();
	GbufferData* get_previous_frame_data();
private:
	int _currFrame = 0;

	VkExtent2D _imageSize;
	GbufferData _gbufferdata[2];
};