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

	Handle<Vrg::Bindable> _glossyReflectionsColorImageBinding;
	Handle<Vrg::Bindable> _glossyReflectionsGbufferImageBinding;
private:
	VkExtent2D _imageSize;

	AllocatedImage _glossyReflectionsColorImage;
	AllocatedImage _glossyReflectionsGbufferImage;

	std::vector<Handle<Vrg::Bindable>> _glossyReflectionsColorImageBindings;
	std::vector<Handle<Vrg::Bindable>> _ColorbindingForColor;
	std::vector<Handle<Vrg::Bindable>> _gBufferbindingForColor;

	std::vector<Handle<Vrg::Bindable>> _glossyReflectionsGbufferImageBindings;
	std::vector<Handle<Vrg::Bindable>> _gBufferbindingForGbuffer;
};