#pragma once
#include "vk_types.h"
#include <vector>

class GBuffer;
class Shadow;
class DiffuseIllumination;
class BRDF;

class GlossyIllumination {
public:
	void init_images(EngineData& engineData, VkExtent2D imageSize);
	void render(EngineData& engineData, SceneData& sceneData, GBuffer& gbuffer, Shadow& shadow, DiffuseIllumination& diffuseIllumination, BRDF& brdfUtils);

	Vrg::Bindable* _glossyReflectionsColorImageBinding;
	Vrg::Bindable* _glossyReflectionsGbufferImageBinding;
private:
	VkExtent2D _imageSize;

	AllocatedImage _glossyReflectionsColorImage;
	AllocatedImage _glossyReflectionsGbufferImage;

	std::vector<Vrg::Bindable*> _glossyReflectionsColorImageBindings;
	std::vector<Vrg::Bindable*> _ColorbindingForColor;
	std::vector<Vrg::Bindable*> _gBufferbindingForColor;

	std::vector<Vrg::Bindable*> _glossyReflectionsGbufferImageBindings;
	std::vector<Vrg::Bindable*> _gBufferbindingForGbuffer;
};