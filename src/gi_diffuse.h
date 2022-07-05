#pragma once

#include <vk_compute.h>
#include <vk_raytracing.h>
#include <precalculation_types.h>
#include <vk_debug_renderer.h>
#include <gi_shadow.h>
#include <gi_brdf.h>
#include <vk_types.h>

class DiffuseIllumination {
public:
	void init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, GltfScene& scene);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneData& sceneData, Shadow& shadow, BRDF& brdfUtils, std::function<void(VkCommandBuffer cmd)>&& function, bool realtimeProbeRaycast);
	void render_ground_truth(VkCommandBuffer cmd, EngineData& engineData, SceneData& sceneData, Shadow& shadow, BRDF& brdfUtils);

	void debug_draw_probes(VulkanDebugRenderer& debugRenderer, bool showProbeRays, float sceneScale);
	void debug_draw_receivers(VulkanDebugRenderer& debugRenderer, float sceneScale);
	void debug_draw_specific_receiver(VulkanDebugRenderer& debugRenderer, int specificCluster, int specificReceiver, int specificReceiverRaySampleCount, bool* enabledProbes, bool showSpecificProbeRays, float sceneScale);

	AllocatedBuffer _configBuffer;
	Vrg::Bindable* _configBufferBinding;
	AllocatedBuffer _probeRaycastResultOfflineBuffer;
	Vrg::Bindable* _probeRaycastResultOfflineBufferBinding;
	AllocatedBuffer _probeRaycastResultOnlineBuffer;
	Vrg::Bindable* _probeRaycastResultOnlineBufferBinding;
	AllocatedBuffer _probeBasisBuffer;
	Vrg::Bindable* _probeBasisBufferBinding;
	AllocatedBuffer _probeRelightOutputBuffer;
	Vrg::Bindable* _probeRelightOutputBufferBinding;

	AllocatedBuffer _clusterProjectionMatricesBuffer;
	Vrg::Bindable* _clusterProjectionMatricesBufferBinding;
	AllocatedBuffer _clusterProjectionOutputBuffer;
	Vrg::Bindable* _clusterProjectionOutputBufferBinding;
	AllocatedBuffer _clusterReceiverInfos;
	Vrg::Bindable* _clusterReceiverInfosBinding;
	AllocatedBuffer _clusterProbes;
	Vrg::Bindable* _clusterProbesBinding;

	AllocatedBuffer _receiverReconstructionMatricesBuffer;
	Vrg::Bindable* _receiverReconstructionMatricesBufferBinding;
	AllocatedBuffer _clusterReceiverUvs;
	Vrg::Bindable* _clusterReceiverUvsBinding;

	AllocatedBuffer _probeLocationsBuffer;
	Vrg::Bindable* _probeLocationsBufferBinding;

	AllocatedBuffer _receiverBuffer;
	Vrg::Bindable* _receiverBufferBinding;

	AllocatedImage _giIndirectLightImage;
	Vrg::Bindable* _giIndirectLightImageBinding;
	AllocatedImage _lightmapColorImage;
	Vrg::Bindable* _lightmapColorImageBinding;
	AllocatedImage _dilatedGiIndirectLightImage;
	Vrg::Bindable* _dilatedGiIndirectLightImageBinding;

	VkExtent2D _lightmapExtent{ 2048 , 2048 };
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

	GIConfig _config = {};
	VkExtent2D _giLightmapExtent{ 0 , 0 };

	uint32_t _gpuReceiverCount;

	std::vector<GPUReceiverDataUV> receiverDataVector;
};