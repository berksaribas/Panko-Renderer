#include "gi_shadow.h"
#include <vk_initializers.h>
#include <vk_utils.h>
#include <vk_pipeline.h>
#include <vk_rendergraph.h>
#include "../shaders/common.glsl"

void Shadow::init_images(EngineData& engineData)
{
	VkExtent3D depthImageExtent3D = {
		_shadowMapExtent.width,
		_shadowMapExtent.height,
		1
	};

	_shadowMapColorImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);

	_shadowMapDepthImage = vkutils::create_image(&engineData, DEPTH_32_FORMAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent3D);

	_shadowMapColorImageBinding = engineData.renderGraph->register_image_view(&_shadowMapColorImage, {
		.sampler = Vrg::Sampler::NEAREST,
		.baseMipLevel = 0,
		.mipLevelCount = 1
	}, "ShadowMapColorImage");

	_shadowMapDepthImageBinding = engineData.renderGraph->register_image_view(&_shadowMapDepthImage, {
		.sampler = Vrg::Sampler::NEAREST,
		.baseMipLevel = 0,
		.mipLevelCount = 1
	}, "ShadowMapDepthImage");
}

void Shadow::init_buffers(EngineData& engineData)
{
	_shadowMapDataBuffer = vkutils::create_buffer(engineData.allocator, sizeof(GPUShadowMapData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	vkutils::cpu_to_gpu(engineData.allocator, _shadowMapDataBuffer, &_shadowMapData, sizeof(GPUShadowMapData));
	_shadowMapDataBinding = engineData.renderGraph->register_uniform_buffer(&_shadowMapDataBuffer, "ShadowMapData");
}

void Shadow::prepare_rendering(EngineData& engineData)
{
	vkutils::cpu_to_gpu(engineData.allocator, _shadowMapDataBuffer, &_shadowMapData, sizeof(GPUShadowMapData));
}

void Shadow::render(EngineData& engineData, SceneData& sceneData, std::function<void(VkCommandBuffer cmd)>&& function)
{
	VkClearValue zeroColor = {
		.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
	};

	VkClearValue depthColor = {
		.depthStencil = { 1.0f, 0 }
	};

	engineData.renderGraph->add_render_pass(
	{
		.name = "ShadowPass",
		.pipelineType = Vrg::PipelineType::RASTER_TYPE,
		.rasterPipeline = {
			.vertexShader = "../../shaders/evsm.vert.spv",
			.fragmentShader = "../../shaders/evsm.frag.spv",
			.size = _shadowMapExtent,
			.blendAttachmentStates = {
				vkinit::color_blend_attachment_state(),
			},
			.vertexBuffers = {
				sceneData.vertexBufferBinding,
				sceneData.normalBufferBinding,
				sceneData.texBufferBinding,
				sceneData.lightmapTexBufferBinding,
				sceneData.tangentBufferBinding
			},
			.indexBuffer = sceneData.indexBufferBinding,
			.colorOutputs = {
				{_shadowMapColorImageBinding, zeroColor}
			},
			.depthOutput = {_shadowMapDepthImageBinding, depthColor}
		},
		.reads = {
			{0, _shadowMapDataBinding},
			{1, sceneData.objectBufferBinding},
		},
		.execute = function
	});
}
