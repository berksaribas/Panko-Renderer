#pragma once
#include "vk_types.h"

class BRDF {
public:
	void init_images(EngineData& engineData);
	Handle<Vrg::Bindable> brdfLutImageBinding;
	Handle<Vrg::Bindable> scramblingRanking1sppImageBinding;
	Handle<Vrg::Bindable> sobolImageBinding;
private:
	AllocatedImage _brdfLutImage;
	AllocatedImage _scramblingRanking1sppImage;
	AllocatedImage _sobolImage;
};