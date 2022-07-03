#pragma once

#include <vk_compute.h>
#include <vk_raytracing.h>
#include <precalculation_types.h>
#include <vk_debug_renderer.h>
#include <gi_shadow.h>
#include <gi_brdf.h>

class DiffuseIllumination {
public:
	void init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, VulkanCompute* vulkanCompute, VulkanRaytracing* vulkanRaytracing, GltfScene& scene, SceneDescriptors& sceneDescriptors);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, Shadow& shadow, BRDF& brdfUtils, std::function<void(VkCommandBuffer cmd)>&& function, bool realtimeProbeRaycast, VkPipeline dilationPipeline, VkPipelineLayout dilationPipelineLayout);
	void render_ground_truth(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, Shadow& shadow, BRDF& brdfUtils, VkPipeline dilationPipeline, VkPipelineLayout dilationPipelineLayout);
	
	void build_lightmap_pipeline(EngineData& engineData);
	void build_proberaycast_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer);
	void build_realtime_proberaycast_pipeline(EngineData& engineData, SceneDescriptors& sceneDescriptors);

	void build_radiance_coefficients_descriptor(EngineData& engineData);

	void build_groundtruth_gi_raycast_descriptors(EngineData& engineData, GltfScene& scene, SceneDescriptors& sceneDescriptors, AllocatedBuffer sceneDescBuffer, AllocatedBuffer meshInfoBuffer);
	void build_groundtruth_gi_raycast_pipeline(EngineData& engineData, SceneDescriptors& sceneDescriptors);

	void rebuild_shaders(EngineData& engineData, SceneDescriptors& sceneDescriptors);

	void debug_draw_probes(VulkanDebugRenderer& debugRenderer, bool showProbeRays, float sceneScale);
	void debug_draw_receivers(VulkanDebugRenderer& debugRenderer, float sceneScale);
	void debug_draw_specific_receiver(VulkanDebugRenderer& debugRenderer, int specificCluster, int specificReceiver, int specificReceiverRaySampleCount, bool* enabledProbes, bool showSpecificProbeRays, float sceneScale);

	void cleanup(EngineData& engineData);
	VkDescriptorSet _giIndirectLightTextureDescriptor;
	VkDescriptorSet _dilatedGiIndirectLightTextureDescriptor;
	VkExtent2D _lightmapExtent{ 2048 , 2048 };

	//Test: new sh
	VkDescriptorSetLayout _radianceCoefficientsDescriptorSetLayout;
	VkDescriptorSet _radianceCoefficientsDescriptorSet;
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
	ComputeInstance _probeRelightRealtime = {};
	ComputeInstance _clusterProjection = {};
	ComputeInstance _receiverReconstruction = {};

	AllocatedImage _giIndirectLightImage;
	VkImageView _giIndirectLightImageView;

	AllocatedBuffer _probeRelightOutputBuffer;
	AllocatedBuffer _clusterProjectionOutputBuffer;

	VkExtent2D _giLightmapExtent{ 0 , 0 };

	AllocatedImage _lightmapColorImage;
	VkImageView _lightmapColorImageView;
	VkFramebuffer _lightmapFramebuffer;
	VkDescriptorSet _lightmapTextureDescriptor;

	VkPipeline _lightmapPipeline;
	VkPipelineLayout _lightmapPipelineLayout;

	AllocatedImage _dilatedGiIndirectLightImage;
	VkImageView _dilatedGiIndirectLightImageView;
	VkFramebuffer _dilatedGiIndirectLightFramebuffer;

	//Raytracing for probes
	RaytracingPipeline _probeRTPipeline;
	VkDescriptorSetLayout _probeRTDescriptorSetLayout;
	VkDescriptorSet _probeRTDescriptorSet;
	AllocatedBuffer _probeLocationsBuffer;
	AllocatedBuffer _probeRaycastResultBuffer;

	//Raytracing for ground truth
	RaytracingPipeline _gtDiffuseRTPipeline;
	VkDescriptorSetLayout _gtDiffuseRTDescriptorSetLayout;
	VkDescriptorSet _gtDiffuseRTDescriptorSet;
	AllocatedBuffer _receiverBuffer;
	int _gpuReceiverCount;

	std::vector<GPUReceiverDataUV> receiverDataVector;

};