#pragma once
#include "vk_types.h"

class BRDF {
public:
	void init_images(EngineData& engineData);
	Vrg::Bindable* brdfLutImageBinding;
	Vrg::Bindable* scramblingRanking1sppImageBinding;
	Vrg::Bindable* sobolImageBinding;
private:
	AllocatedImage _brdfLutImage;
	AllocatedImage _scramblingRanking1sppImage;
	AllocatedImage _sobolImage;
};