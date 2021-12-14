#pragma once

#include "vk_types.h"
#include <gi_gbuffer.h>
#include <gi_shadow.h>
#include <gi_diffuse.h>

class Deferred {
public:
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, bool rebuild = false);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination);
	void cleanup();
	VkDescriptorSet _deferredColorTextureDescriptor;
private:
	VkPipeline _deferredPipeline;
	VkPipelineLayout _deferredPipelineLayout;

	AllocatedImage _deferredColorImage;
	VkImageView _deferredColorImageView;
	VkFramebuffer _deferredFramebuffer;

	VkExtent2D _imageSize;
};