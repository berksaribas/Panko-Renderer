#pragma once
#include "vk_types.h"
#include <gi_gbuffer.h>
#include <gi_glossy.h>

struct TemporalData {
	AllocatedImage colorImage;
	AllocatedImage momentsImage;
	Handle<Vrg::Bindable> colorImageBinding;
	Handle<Vrg::Bindable> momentsImageBinding;
};

struct AtrousData {
	AllocatedImage pingImage;
	Handle<Vrg::Bindable> pingImageBinding;
};

class GlossyDenoise {
public:
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void render(EngineData& engineData, SceneData& sceneData, GBuffer& gbuffer, GlossyIllumination& glossyIllumination);
	Handle<Vrg::Bindable> get_denoised_binding();
	int num_atrous = 4;
private:
	int _currFrame = 0;
	VkExtent2D _imageSize;

	TemporalData _temporalData[2];
	AtrousData _atrousData[2];
};