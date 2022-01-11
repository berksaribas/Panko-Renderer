#pragma once

#include <vk_compute.h>
#include <vk_raytracing.h>
#include <precalculation_types.h>
#include <vk_debug_renderer.h>

class DiffuseIllumination {
public:
	void init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, VulkanCompute* vulkanCompute, VulkanRaytracing* vulkanRaytracing, GltfScene& scene, SceneDescriptors& sceneDescriptors, VkImageView lightmapImageView);
	void render(VkCommandBuffer cmd, SceneDescriptors& sceneDescriptors);
	void rebuild_shaders();

	void debug_draw_probes(VulkanDebugRenderer& debugRenderer, bool showProbeRays, float sceneScale);
	void debug_draw_receivers(VulkanDebugRenderer& debugRenderer, float sceneScale);
	void debug_draw_specific_receiver(VulkanDebugRenderer& debugRenderer, int specificCluster, int specificReceiver, int specificReceiverRaySampleCount, bool* enabledProbes, bool showSpecificProbeRays, float sceneScale);

	void cleanup();
	VkDescriptorSet _giIndirectLightTextureDescriptor;
private:
	VkDevice _device;
	VmaAllocator _allocator;
	VkDescriptorPool _descriptorPool;
	VulkanCompute* _vulkanCompute;
	VulkanRaytracing* _vulkanRaytracing;
	VkRenderPass _colorRenderPass;
	PrecalculationInfo* _precalculationInfo;
	PrecalculationLoadData* _precalculationLoadData;
	PrecalculationResult* _precalculationResult;

	AllocatedBuffer _configBuffer;
	GIConfig _config = {};

	ComputeInstance _probeRelight = {};
	ComputeInstance _clusterProjection = {};
	ComputeInstance _receiverReconstruction = {};

	AllocatedImage _giIndirectLightImage;
	VkImageView _giIndirectLightImageView;

	AllocatedBuffer _probeRelightOutputBuffer;
	AllocatedBuffer _clusterProjectionOutputBuffer;

	VkExtent2D _giLightmapExtent{ 0 , 0 };
};