#include <gi_deferred.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include <vk_pipeline.h>
#include <vk_rendergraph.h>
#include <gi_gbuffer.h>
#include <gi_shadow.h>
#include <gi_diffuse.h>
#include <gi_glossy.h>
#include <gi_brdf.h>
#include <gi_glossy_svgf.h>

void Deferred::init_images(EngineData& engineData, VkExtent2D imageSize)
{
	_imageSize = imageSize;

	VkExtent3D extent3D = {
		_imageSize.width,
		_imageSize.height,
		1
	};

	_deferredColorImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

	_deferredColorImageBinding = engineData.renderGraph->register_image_view(&_deferredColorImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = 1
	}, "DeferredColorImage");
}

void Deferred::render(EngineData& engineData, SceneData& sceneData, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination, GlossyIllumination& glossyIllumination, BRDF& brdfUtils, Handle<Vrg::Bindable> glossyBinding)
{
	auto gbufferData = gbuffer.get_current_frame_data();

	VkClearValue clearValue;
	clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

	engineData.renderGraph->add_render_pass({
		.name = "DeferredPass",
		.pipelineType = Vrg::PipelineType::RASTER_TYPE,
		.rasterPipeline = {
			.vertexShader = "../../shaders/fullscreen.vert.spv",
			.fragmentShader = "../../shaders/deferred.frag.spv",
			.size = _imageSize,
			.depthState = { false, false, VK_COMPARE_OP_NEVER },
			.cullMode = Vrg::CullMode::NONE,
			.blendAttachmentStates = {
				vkinit::color_blend_attachment_state(),
			},
			.colorOutputs = {
				{_deferredColorImageBinding, clearValue, false},
			},

		},
		.reads = {
			{0, sceneData.cameraBufferBinding},
			{0, shadow._shadowMapDataBinding},
			{1, gbufferData->albedoMetallicBinding},
			{1, gbufferData->normalBinding},
			{1, gbufferData->motionBinding},
			{1, gbufferData->roughnessDepthCurvatureMaterialBinding},
			{1, gbufferData->uvBinding},
			{1, gbufferData->depthBinding},
			{3, sceneData.materialBufferBinding},
			{4, shadow._shadowMapColorImageBinding},
			{5, diffuseIllumination._dilatedGiIndirectLightImageBinding},
			{6, glossyBinding},
			{7, brdfUtils.brdfLutImageBinding},
			{8, glossyIllumination._glossyReflectionsGbufferImageBinding},
		},
		.extraDescriptorSets = {
			{2, sceneData.textureDescriptor, sceneData.textureSetLayout}
		},
		.execute = [&](VkCommandBuffer cmd) {
			vkCmdDraw(cmd, 3, 1, 0, 0);
		}
});
}
