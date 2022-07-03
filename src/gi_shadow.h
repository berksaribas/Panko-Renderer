#pragma once

#include <vk_types.h>
#include <functional>
#include "../shaders/common.glsl"

class Shadow {
public:
	void init_images(EngineData& engineData);
	void init_buffers(EngineData& engineData);
	void prepare_rendering(EngineData& engineData);
	void render(EngineData& engineData, SceneData& sceneData, std::function<void(VkCommandBuffer cmd)>&& function);

	AllocatedBuffer _shadowMapDataBuffer;
	GPUShadowMapData _shadowMapData = {};
	Vrg::Bindable* _shadowMapDataBinding;
	Vrg::Bindable* _shadowMapDepthImageBinding;
	Vrg::Bindable* _shadowMapColorImageBinding;
private:
	VkExtent2D _shadowMapExtent{ 4096 , 4096 };
	AllocatedImage _shadowMapDepthImage;
	AllocatedImage _shadowMapColorImage;
};