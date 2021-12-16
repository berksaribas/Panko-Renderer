#pragma once
#include "vk_types.h"
#include <gi_gbuffer.h>
#include <gi_shadow.h>
#include <gi_diffuse.h>

class GlossyIllumination {
public:
	void init(VulkanRaytracing& vulkanRaytracing);
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, bool rebuild = false);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination);
	void cleanup();

	VkDescriptorSet _glossyReflectionsColorTextureDescriptor;
private:
	VulkanRaytracing* _vulkanRaytracing;

	VkExtent2D _imageSize;

	AllocatedImage _glossyReflectionsColorImage;
	VkImageView _glossyReflectionsColorImageView;

	VkPipeline _glossyReflectionPipeline;
	VkPipelineLayout _glossyReflectionPipelineLayout;

	//Raytracing
	RaytracingPipeline rtPipeline;
	VkDescriptorSetLayout rtDescriptorSetLayout;
	VkDescriptorSet rtDescriptorSet;
};