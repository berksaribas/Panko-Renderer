#pragma once

#include "vk_types.h"

class GBuffer;
class Shadow;
class DiffuseIllumination;
class GlossyIllumination;
class SceneData;
class BRDF;

class Deferred
{
public:
    void init_images(EngineData& engineData, VkExtent2D imageSize);
    void render(EngineData& engineData, SceneData& sceneData, GBuffer& gbuffer, Shadow& shadow,
                DiffuseIllumination& diffuseIllumination,
                GlossyIllumination& glossyIllumination, BRDF& brdfUtils,
                Handle<Vrg::Bindable> glossyBinding);

    AllocatedImage _deferredColorImage;
    Handle<Vrg::Bindable> _deferredColorImageBinding;

private:
    VkExtent2D _imageSize;
};