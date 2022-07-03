/*
#pragma once
#include "vk_types.h"
#include <gi_gbuffer.h>
#include <gi_shadow.h>
#include <gi_diffuse.h>
#include <gi_brdf.h>

class GlossyIllumination {
public:
	void init(VulkanRaytracing& vulkanRaytracing);
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, BRDF& brdfUtils, bool rebuild = false);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination, BRDF& brdfUtils);
	void cleanup();

	VkDescriptorSet _glossyReflectionsColorTextureDescriptor;
	VkDescriptorSet _normalMipmapTextureDescriptor;
private:
	void init_blur_pipeline(EngineData& engineData, SceneDescriptors& sceneDescriptors);
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

	//Blur
	VkPipelineLayout _blurPipelineLayout;
	VkPipeline _blurPipeline;


	std::vector<VkImageView> _colorMipmapImageViews;
	std::vector<VkFramebuffer> _colorMipmapFramebuffers;

	AllocatedImage _tempImage;
	std::vector<VkImageView> _tempMipmapImageViews;
	std::vector<VkFramebuffer> _tempMipmapFramebuffers;
	VkDescriptorSet _tempMipmapTextureDescriptor;

	AllocatedImage _normalImage;
	std::vector<VkImageView> _normalMipmapImageViews;
	std::vector<VkFramebuffer> _normalMipmapFramebuffers;
	VkImageView _normalImageView;

};
*/