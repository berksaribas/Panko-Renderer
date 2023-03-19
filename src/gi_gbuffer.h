#pragma once

#include "vk_types.h"
#include <functional>

struct GbufferData
{
    AllocatedImage gbufferAlbedoMetallicImage;
    AllocatedImage gbufferNormalImage;
    AllocatedImage gbufferMotionImage;
    AllocatedImage gbufferRoughnessDepthCurvatureMaterialImage;
    AllocatedImage gbufferUVImage;
    AllocatedImage gbufferDepthImage;

    Handle<Vrg::Bindable> albedoMetallicBinding;
    Handle<Vrg::Bindable> normalBinding;
    Handle<Vrg::Bindable> motionBinding;
    Handle<Vrg::Bindable> roughnessDepthCurvatureMaterialBinding;
    Handle<Vrg::Bindable> uvBinding;
    Handle<Vrg::Bindable> depthBinding;
};

class GBuffer
{
public:
    void init_images(EngineData& engineData, VkExtent2D imageSize);
    void render(EngineData& engineData, SceneData& sceneData,
                std::function<void(VkCommandBuffer cmd)>&& function);

    VkDescriptorSetLayout _gbufferDescriptorSetLayout;
    GbufferData* get_current_frame_data();
    GbufferData* get_previous_frame_data();

private:
    int _currFrame = 0;

    VkExtent2D _imageSize;
    GbufferData _gbufferdata[2];
};