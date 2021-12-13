#include <vk_compute.h>
#include <vk_raytracing.h>
#include <precalculation_types.h>
#include <vk_debug_renderer.h>

class DiffuseIllumination {
public:
	void init(EngineData& engineData, PrecalculationInfo* precalculationInfo, PrecalculationLoadData* precalculationLoadData, PrecalculationResult* precalculationResult, VulkanCompute* vulkanCompute, VulkanRaytracing* vulkanRaytracing, GltfScene& scene, SceneDescriptors& sceneDescriptors, VkImageView lightmapImageView);
	void render(VkCommandBuffer cmd, VkPipeline dilationPipeline, VkPipelineLayout dilationPipelineLayout, SceneDescriptors& sceneDescriptors);
	void rebuild_shaders();

	void debug_draw_probes(VulkanDebugRenderer& debugRenderer, bool showProbeRays, float sceneScale);
	void debug_draw_receivers(VulkanDebugRenderer& debugRenderer, float sceneScale);
	void debug_draw_specific_receiver(VulkanDebugRenderer& debugRenderer, int specificCluster, int specificReceiver, int specificReceiverRaySampleCount, bool* enabledProbes, bool showSpecificProbeRays, float sceneScale);

	void cleanup();
	VkDescriptorSet dilatedGiInirectLightTextureDescriptor;
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

	AllocatedBuffer configBuffer;
	GIConfig config = {};

	ComputeInstance probeRelight = {};
	ComputeInstance clusterProjection = {};
	ComputeInstance receiverReconstruction = {};

	AllocatedImage giInirectLightImage;
	VkImageView giInirectLightImageView;
	VkDescriptorSet giInirectLightTextureDescriptor;

	AllocatedImage dilatedGiInirectLightImage;
	VkImageView dilatedGiInirectLightImageView;
	VkFramebuffer dilatedGiInirectLightFramebuffer;

	AllocatedBuffer probeRelightOutputBuffer;
	AllocatedBuffer clusterProjectionOutputBuffer;

	VkExtent2D giLightmapExtent{ 0 , 0 };
};