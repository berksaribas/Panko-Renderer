#include "gi_glossy_svgf.h"
#include <vk_initializers.h>
#include <vk_utils.h>
#include <vk_rendergraph.h>

void GlossyDenoise::init_images(EngineData& engineData, VkExtent2D imageSize)
{
	_imageSize = imageSize;

	VkExtent3D extent3D = {
			_imageSize.width,
			_imageSize.height,
			1
	};

	for (int i = 0; i < 2; i++) {
		_temporalData[i].colorImage = vkutils::create_image(&engineData, COLOR_16_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

		_temporalData[i].momentsImage = vkutils::create_image(&engineData, COLOR_16_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

		_temporalData[i].colorImageBinding = engineData.renderGraph->register_image_view(&_temporalData[i].colorImage, {
			.sampler = Vrg::Sampler::NEAREST,
			.baseMipLevel = 0,
			.mipLevelCount = 1
		}, "SVGFTemporalColorImage " + std::to_string(i));

		_temporalData[i].momentsImageBinding = engineData.renderGraph->register_image_view(&_temporalData[i].momentsImage, {
			.sampler = Vrg::Sampler::NEAREST,
			.baseMipLevel = 0,
			.mipLevelCount = 1
		}, "SVGFTemporalMomentsImage " + std::to_string(i));
	}

	//atrous
	for (int i = 0; i < 2; i++) {
		_atrousData[i].pingImage = vkutils::create_image(&engineData, COLOR_16_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

		_atrousData[i].pingImageBinding = engineData.renderGraph->register_image_view(&_atrousData[i].pingImage, {
			.sampler = Vrg::Sampler::NEAREST,
			.baseMipLevel = 0,
			.mipLevelCount = 1
		}, "SVGFAtrousImage " + std::to_string(i));
	}
}

void GlossyDenoise::render(EngineData& engineData, SceneData& sceneData, GBuffer& gbuffer, GlossyIllumination& glossyIllumination)
{
	_currFrame++;

	auto gbufferCurrent = gbuffer.get_current_frame_data();
	auto gbufferPrevious = gbuffer.get_previous_frame_data();

	engineData.renderGraph->add_render_pass({
		.name = "SVGFTemporalPass",
		.pipelineType = Vrg::PipelineType::COMPUTE_TYPE,
		.computePipeline = {
			.shader = "../../shaders/svgf_temporal.comp.spv",
			.dimX = static_cast<uint32_t>(ceil(float(_imageSize.width) / float(8))),
			.dimY = static_cast<uint32_t>(ceil(float(_imageSize.height) / float(8))),
			.dimZ = 1
		},
		.writes = {
			{5, _temporalData[(_currFrame) % 2].colorImageBinding},
			{5, _temporalData[(_currFrame) % 2].momentsImageBinding}
		},
		.reads = {
			//0
			{0, sceneData.cameraBufferBinding},
			//1
			{1, gbufferCurrent->albedoMetallicBinding},
			{1, gbufferCurrent->normalMotionBinding},
			{1, gbufferCurrent->roughnessDepthCurvatureMaterialBinding},
			{1, gbufferCurrent->uvBinding},
			{1, gbufferCurrent->depthBinding},
			//2
			{2, gbufferPrevious->albedoMetallicBinding},
			{2, gbufferPrevious->normalMotionBinding},
			{2, gbufferPrevious->roughnessDepthCurvatureMaterialBinding},
			{2, gbufferPrevious->uvBinding},
			{2, gbufferPrevious->depthBinding},
			//3
			{3, glossyIllumination._glossyReflectionsColorImageBinding},
			//4
			{4, _temporalData[(_currFrame - 1) % 2].colorImageBinding},
			{4, _temporalData[(_currFrame - 1) % 2].momentsImageBinding}
		}
	});


	for (int i = 0; i < num_atrous; i++)
	{
		Vrg::Bindable* current = _atrousData[(i) % 2].pingImageBinding;
		if (i == 0) {
			current = _temporalData[(_currFrame) % 2].colorImageBinding;
		}

		int stepsize = 1u << i;
		engineData.renderGraph->add_render_pass({
			.name = "SVGFAtrousPass",
			.pipelineType = Vrg::PipelineType::COMPUTE_TYPE,
			.computePipeline = {
				.shader = "../../shaders/svgf_atrous.comp.spv",
				.dimX = static_cast<uint32_t>(ceil(float(_imageSize.width) / float(8))),
				.dimY = static_cast<uint32_t>(ceil(float(_imageSize.height) / float(8))),
				.dimZ = 1
			},
			.writes = {
				{3, _atrousData[(i + 1) % 2].pingImageBinding},
			},
			.reads = {
				//0
				{0, sceneData.cameraBufferBinding},
				//1
				{1, gbufferCurrent->albedoMetallicBinding},
				{1, gbufferCurrent->normalMotionBinding},
				{1, gbufferCurrent->roughnessDepthCurvatureMaterialBinding},
				{1, gbufferCurrent->uvBinding},
				{1, gbufferCurrent->depthBinding},
				//3
				{2, current},
			},
			.constants = {
				{&stepsize, sizeof(int)}
			}
		});
	}
}

Vrg::Bindable* GlossyDenoise::get_denoised_binding()
{
	return _atrousData[num_atrous%2].pingImageBinding;
}
