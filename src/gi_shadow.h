#pragma once

#include "../shaders/common.glsl"
#include <functional>
#include <vk_types.h>

class Shadow
{
public:
    void init_images(EngineData& engineData);
    void init_buffers(EngineData& engineData);
    void prepare_rendering(EngineData& engineData);
    void render(EngineData& engineData, SceneData& sceneData,
                std::function<void(VkCommandBuffer cmd)>&& function);

    AllocatedBuffer _shadowMapDataBuffer;
    GPUShadowMapData _shadowMapData = {};
    Handle<Vrg::Bindable> _shadowMapDataBinding;
    Handle<Vrg::Bindable> _shadowMapDepthImageBinding;
    Handle<Vrg::Bindable> _shadowMapColorImageBinding;

private:
    VkExtent2D _shadowMapExtent{4096, 4096};
    AllocatedImage _shadowMapDepthImage;
    AllocatedImage _shadowMapColorImage;
};