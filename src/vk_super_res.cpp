#include "vk_super_res.h"
#include "ffx_fsr2.h"
#include <vk\ffx_fsr2_vk.h>
#include <cstdlib>
#include "vk_rendergraph.h"
#include "editor/editor.h"
#include "vk_utils.h"

FfxFsr2Context context;
FfxFsr2ContextDescription contextDescription;
FfxFsr2DispatchDescription dispatchDescription;

void SuperResolution::initialize(EngineData& engineData, VkExtent2D renderSize, VkExtent2D displaySize)
{
	if (isInitialized) {
		destroy();
		isInitialized = false;

		engineData.renderGraph->destroy_resource(outputImage);
	}

	const size_t scratchBufferSize = ffxFsr2GetScratchMemorySizeVK(engineData.physicalDevice);
	void* scratchBuffer = malloc(scratchBufferSize);
	FfxErrorCode errorCode = ffxFsr2GetInterfaceVK(&contextDescription.callbacks, scratchBuffer, scratchBufferSize, engineData.physicalDevice, vkGetDeviceProcAddr);
	FFX_ASSERT(errorCode == FFX_OK);

	
	contextDescription.flags = FFX_FSR2_ENABLE_AUTO_EXPOSURE | FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;
	contextDescription.maxRenderSize = { renderSize.width, renderSize.height };
	contextDescription.displaySize = { displaySize.width, displaySize.height };
	contextDescription.device = ffxGetDeviceVK(engineData.device);


	int32_t result = ffxFsr2ContextCreate(&context, &contextDescription);
	FFX_ASSERT(result == FFX_OK);

	isInitialized = true;

	outputImage = vkutils::create_image(&engineData, COLOR_32_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, {displaySize.width, displaySize.height, 1});

	outputBindable = engineData.renderGraph->register_image_view(&outputImage, {
		.sampler = Vrg::Sampler::LINEAR,
		.baseMipLevel = 0,
		.mipLevelCount = 1
		}, "SuperResolutionOutputImage");
}

void SuperResolution::get_data(SuperResolutionData* data)
{
	uint32_t resolution_render_x = contextDescription.maxRenderSize.width;
	uint32_t resolution_output_x = contextDescription.displaySize.width;

	// Generate jitter sample
	static uint32_t index = 0; index++;
	const int32_t jitter_phase_count = ffxFsr2GetJitterPhaseCount(resolution_render_x, resolution_output_x);
	ffxFsr2GetJitterOffset(&dispatchDescription.jitterOffset.x, &dispatchDescription.jitterOffset.y, index, jitter_phase_count);

	// Out
	data->jitterX = dispatchDescription.jitterOffset.x;
	data->jitterY = dispatchDescription.jitterOffset.y;
	//TODO: calculate mip bias
	data->mipBias = log2(resolution_render_x / resolution_output_x) - 1.0;
}

void SuperResolution::dispatch(EngineData& engineData, VkCommandBuffer cmd, Handle<Vrg::Bindable> colorInput, Handle<Vrg::Bindable> depthInput, Handle<Vrg::Bindable> motionInput, CameraConfig* cameraConfig, float deltaTime)
{
	auto renderResolution = contextDescription.maxRenderSize;
	auto displayResolution = contextDescription.displaySize;

	auto colorBinding = engineData.renderGraph->bindings.get(colorInput);
	auto depthBinding = engineData.renderGraph->bindings.get(depthInput);
	auto motionBinding = engineData.renderGraph->bindings.get(motionInput);
	auto outputBinding = engineData.renderGraph->bindings.get(outputBindable);

	Vrg::ImageView imView = { Vrg::Sampler::NEAREST, 0, 1 };

	VkImageView colorImageView = engineData.renderGraph->get_image_view(colorBinding->image->_image, imView, colorBinding->image->format);
	VkImageView depthImageView = engineData.renderGraph->get_image_view(depthBinding->image->_image, imView, depthBinding->image->format);
	VkImageView motionImageView = engineData.renderGraph->get_image_view(motionBinding->image->_image, imView, motionBinding->image->format);
	VkImageView outputImageView = engineData.renderGraph->get_image_view(outputBinding->image->_image, imView, outputBinding->image->format);

	engineData.renderGraph->insert_barrier(cmd, colorInput, Vrg::PipelineType::COMPUTE_TYPE, false);
	engineData.renderGraph->insert_barrier(cmd, depthInput, Vrg::PipelineType::COMPUTE_TYPE, false);
	engineData.renderGraph->insert_barrier(cmd, motionInput, Vrg::PipelineType::COMPUTE_TYPE, false);
	engineData.renderGraph->insert_barrier(cmd, outputBindable, Vrg::PipelineType::COMPUTE_TYPE, false);

	dispatchDescription.commandList = ffxGetCommandListVK(cmd);
	dispatchDescription.color = ffxGetTextureResourceVK(&context, colorBinding->image->_image, colorImageView, renderResolution.width, renderResolution.height, colorBinding->image->format);
	dispatchDescription.depth = ffxGetTextureResourceVK(&context, depthBinding->image->_image, depthImageView, renderResolution.width, renderResolution.height, depthBinding->image->format);
	dispatchDescription.motionVectors = ffxGetTextureResourceVK(&context, motionBinding->image->_image, motionImageView, renderResolution.width, renderResolution.height, motionBinding->image->format);
	dispatchDescription.output = ffxGetTextureResourceVK(&context, outputBinding->image->_image, outputImageView, renderResolution.width, renderResolution.height, outputBinding->image->format);
	dispatchDescription.motionVectorScale = { (float)renderResolution.width, (float)renderResolution.height };
	dispatchDescription.renderSize = renderResolution;
	dispatchDescription.enableSharpening = false;
	dispatchDescription.sharpness = false;
	dispatchDescription.frameTimeDelta = deltaTime;
	dispatchDescription.preExposure = 1.0f;
	dispatchDescription.reset = false;
	dispatchDescription.cameraNear = cameraConfig->nearPlane;
	dispatchDescription.cameraFar = cameraConfig->farPlane;
	dispatchDescription.cameraFovAngleVertical = cameraConfig->fov;

	int32_t result = ffxFsr2ContextDispatch(&context, &dispatchDescription);
	FFX_ASSERT(result == FFX_OK);

	engineData.renderGraph->inform_current_image_layout(outputBinding->image->_image, 0, VK_IMAGE_LAYOUT_GENERAL);
}

void SuperResolution::destroy()
{
	ffxFsr2ContextDestroy(&context);
}
