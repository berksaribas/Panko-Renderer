#include <gi_glossy.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include "../shaders/common.glsl"
#include <vk_pipeline.h>
#include <vk_rendergraph.h>
#include <gi_gbuffer.h>
#include <gi_shadow.h>
#include <gi_diffuse.h>
#include <gi_brdf.h>

uint32_t mipmapLevels;

void GlossyIllumination::init_images(EngineData& engineData, VkExtent2D imageSize)
{
	_imageSize = imageSize;

	VkExtent3D extent3D = {
			_imageSize.width,
			_imageSize.height,
			1
	};

	mipmapLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(_imageSize.width, _imageSize.height)))) + 1;

	_glossyReflectionsColorImage = vkutils::create_image(&engineData, COLOR_16_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent3D, mipmapLevels);

	_glossyReflectionsGbufferImage = vkutils::create_image(&engineData, COLOR_16_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent3D, mipmapLevels);
	
	_glossyReflectionsColorImageBinding = engineData.renderGraph->register_image_view(&_glossyReflectionsColorImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = mipmapLevels
	}, "GlossyReflectionColorImageBase");

	_glossyReflectionsGbufferImageBinding = engineData.renderGraph->register_image_view(&_glossyReflectionsGbufferImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = mipmapLevels
	}, "GlossyReflectionGbufferImageBase");

	for (uint32_t i = 0; i < mipmapLevels; i++) {
		break;//todo remove this
		_glossyReflectionsColorImageBindings.push_back(engineData.renderGraph->register_image_view(&_glossyReflectionsColorImage, {
			.sampler = Vrg::Sampler::LINEAR,
			.baseMipLevel = i,
			.mipLevelCount = 1
		}, "GlossyReflectionColorImage " + std::to_string(i)));

		_glossyReflectionsGbufferImageBindings.push_back(engineData.renderGraph->register_image_view(&_glossyReflectionsGbufferImage, {
			.sampler = Vrg::Sampler::LINEAR,
			.baseMipLevel = i,
			.mipLevelCount = 1
		}, "GlossyReflectionGbufferImage " + std::to_string(i)));
	}

}

void GlossyIllumination::render(EngineData& engineData, SceneData& sceneData, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination, BRDF& brdfUtils)
{
	auto gbufferData = gbuffer.get_current_frame_data();
	engineData.renderGraph->add_render_pass({
		.name = "GlossyPathTracePass",
		.pipelineType = Vrg::PipelineType::RAYTRACING_TYPE,
		.raytracingPipeline = {
			.rgenShader = "../../shaders/reflections/reflections_rt.rgen.spv",
			.missShader = "../../shaders/reflections/reflections_rt.rmiss.spv",
			.hitShader = "../../shaders/reflections/reflections_rt.rchit.spv",
			.width = (uint32_t)_imageSize.width,
			.height = (uint32_t)_imageSize.height,
		},
		.writes = {
			{1, _glossyReflectionsColorImageBinding},
			{1, _glossyReflectionsGbufferImageBinding},
		},
		.reads = {
			{1, gbufferData->albedoMetallicBinding},
			{1, gbufferData->normalBinding},
			{1, gbufferData->motionBinding},
			{1, gbufferData->roughnessDepthCurvatureMaterialBinding},
			{1, gbufferData->uvBinding},
			{1, gbufferData->depthBinding},
			{1, brdfUtils.sobolImageBinding},
			{1, brdfUtils.scramblingRanking1sppImageBinding},
			//same for all
			{2, sceneData.cameraBufferBinding},
			{2, shadow._shadowMapDataBinding},
			{3, sceneData.objectBufferBinding},
			{5, sceneData.materialBufferBinding},
			{6, shadow._shadowMapColorImageBinding},
			{7, diffuseIllumination._dilatedGiIndirectLightImageBinding},
			{8, brdfUtils.brdfLutImageBinding},
		},
		.extraDescriptorSets = {
			{0, sceneData.raytracingDescriptor, sceneData.raytracingSetLayout},
			{4, sceneData.textureDescriptor, sceneData.textureSetLayout}
		}
	});

	/*
	//Do gaussian blur on each mip chain. Each mip will sample from the previous mip. I'll need a temp image as well
	for (int i = 1; i < mipmapLevels; i++) {
		VkExtent2D previousMipSize = { _imageSize.width / (int)pow(2, i - 1), _imageSize.height / (int)pow(2, i - 1) };
		VkExtent2D mipSize = { _imageSize.width / (int)pow(2, i), _imageSize.height / (int)pow(2, i) };
		VkClearValue clearValue;
		clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		{
			glm::vec3 pushData = { 2, 0, i - 1 };
			engineData.renderGraph->add_render_pass({
				.name = "GlossyBlurPass",
				.pipelineType = Vrg::PipelineType::RASTER_TYPE,
				.rasterPipeline = {
					.vertexShader = "../../shaders/fullscreen.vert.spv",
					.fragmentShader = "../../shaders/gaussblur.frag.spv",
					.size = mipSize,
					.depthState = { false, false, VK_COMPARE_OP_NEVER },
					.cullMode = Vrg::CullMode::NONE,
					.blendAttachmentStates = {
						vkinit::color_blend_attachment_state(),
					},
					.colorOutputs = {
						{_glossyReflectionsGbufferImageBindings[i], clearValue, false},
					},

				},
				.reads = {
					{0, _glossyReflectionsGbufferImageBinding},
					{0, _glossyReflectionsGbufferImageBinding},
				},
				.constants = {
					{&pushData, sizeof(glm::vec3)}
				},
				.execute = [&](VkCommandBuffer cmd) {
					vkCmdDraw(cmd, 3, 1, 0, 0);
				}
			});
		}		
	}
	//return;
	for (int i = 1; i < mipmapLevels; i++) {
		VkExtent2D previousMipSize = { _imageSize.width / (int)pow(2, i - 1), _imageSize.height / (int)pow(2, i - 1) };
		VkExtent2D mipSize = { _imageSize.width / (int)pow(2, i), _imageSize.height / (int)pow(2, i) };
		VkClearValue clearValue;
		clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		{
			glm::vec3 pushData = { 0, 1, i - 1 };
			engineData.renderGraph->add_render_pass({
				.name = "GlossyBlurPass",
				.pipelineType = Vrg::PipelineType::RASTER_TYPE,
				.rasterPipeline = {
					.vertexShader = "../../shaders/fullscreen.vert.spv",
					.fragmentShader = "../../shaders/gaussblur.frag.spv",
					.size = mipSize,
					.depthState = { false, false, VK_COMPARE_OP_NEVER },
					.cullMode = Vrg::CullMode::NONE,
					.blendAttachmentStates = {
						vkinit::color_blend_attachment_state(),
					},
					.colorOutputs = {
						{_glossyReflectionsColorImageBindings[i], clearValue, false},
					},

				},
				.reads = {
					{0, _glossyReflectionsColorImageBinding},
					{0, _glossyReflectionsGbufferImageBinding},
				},
				.constants = {
					{&pushData, sizeof(glm::vec3)}
				},
				.execute = [&](VkCommandBuffer cmd) {
					vkCmdDraw(cmd, 3, 1, 0, 0);
				}
			});
		}
	}
	*/
}
