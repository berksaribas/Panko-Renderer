#include <vk_types.h>
#include <functional>
#include "../shaders/common.glsl"

class Shadow {
public:
	void init_images(EngineData& engineData);
	void init_buffers(EngineData& engineData);
	void init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors);
	void init_pipelines(EngineData& engineData, SceneDescriptors& sceneDescriptors, bool rebuild = false);
	void prepare_rendering(EngineData& engineData);
	void render(VkCommandBuffer cmd, EngineData& engineData, SceneDescriptors& sceneDescriptors, std::function<void(VkCommandBuffer cmd)>&& function);
	void cleanup(EngineData& engineData);

	AllocatedBuffer _shadowMapDataBuffer;
	GPUShadowMapData _shadowMapData = {};
	VkDescriptorSet _shadowMapTextureDescriptor;
private:
	VkDescriptorSet _shadowMapDataDescriptor;

	VkExtent2D _shadowMapExtent{ 4096 , 4096 };

	AllocatedImage _shadowMapDepthImage;
	VkImageView _shadowMapDepthImageView;

	AllocatedImage _shadowMapColorImage;
	VkImageView _shadowMapColorImageView;

	VkFramebuffer _shadowMapFramebuffer;

	VkPipeline _shadowMapPipeline;
	VkPipelineLayout _shadowMapPipelineLayout;

	VkDescriptorSetLayout _shadowMapDataSetLayout;
};