#include "vk_rendergraph.h"
#include <gi_gbuffer.h>
#include <vk_initializers.h>
#include <vk_pipeline.h>
#include <vk_utils.h>

void GBuffer::init_images(EngineData& engineData, VkExtent2D imageSize)
{
    _imageSize = imageSize;
    VkExtent3D extent3D = {_imageSize.width, _imageSize.height, 1};

    for (int i = 0; i < 2; i++)
    {
        _gbufferdata[i].gbufferAlbedoMetallicImage = vkutils::create_image(
            &engineData, COLOR_8_FORMAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

        _gbufferdata[i].gbufferNormalImage = vkutils::create_image(
            &engineData, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

        _gbufferdata[i].gbufferMotionImage = vkutils::create_image(
            &engineData, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

        _gbufferdata[i].gbufferRoughnessDepthCurvatureMaterialImage = vkutils::create_image(
            &engineData, COLOR_16_FORMAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

        _gbufferdata[i].gbufferUVImage = vkutils::create_image(
            &engineData, COLOR_16_FORMAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent3D);

        _gbufferdata[i].gbufferDepthImage = vkutils::create_image(
            &engineData, DEPTH_32_FORMAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            extent3D);

        auto index = std::to_string(i);

        _gbufferdata[i].albedoMetallicBinding = engineData.renderGraph->register_image_view(
            &_gbufferdata[i].gbufferAlbedoMetallicImage,
            {.sampler = Vrg::Sampler::NEAREST, .baseMipLevel = 0, .mipLevelCount = 1},
            "GbufferAlbedoMetallicImage" + index);
        _gbufferdata[i].normalBinding = engineData.renderGraph->register_image_view(
            &_gbufferdata[i].gbufferNormalImage,
            {.sampler = Vrg::Sampler::NEAREST, .baseMipLevel = 0, .mipLevelCount = 1},
            "GbufferNormalImage" + index);
        _gbufferdata[i].motionBinding = engineData.renderGraph->register_image_view(
            &_gbufferdata[i].gbufferMotionImage,
            {.sampler = Vrg::Sampler::NEAREST, .baseMipLevel = 0, .mipLevelCount = 1},
            "GbufferMotionImage" + index);
        _gbufferdata[i].roughnessDepthCurvatureMaterialBinding =
            engineData.renderGraph->register_image_view(
                &_gbufferdata[i].gbufferRoughnessDepthCurvatureMaterialImage,
                {.sampler = Vrg::Sampler::NEAREST, .baseMipLevel = 0, .mipLevelCount = 1},
                "GbufferRoughnessDepthCurvatureMaterialImage" + index);
        _gbufferdata[i].uvBinding = engineData.renderGraph->register_image_view(
            &_gbufferdata[i].gbufferUVImage,
            {.sampler = Vrg::Sampler::NEAREST, .baseMipLevel = 0, .mipLevelCount = 1},
            "GbufferUVImage" + index);
        _gbufferdata[i].depthBinding = engineData.renderGraph->register_image_view(
            &_gbufferdata[i].gbufferDepthImage,
            {.sampler = Vrg::Sampler::NEAREST, .baseMipLevel = 0, .mipLevelCount = 1},
            "GbufferDepthImage" + index);
    }
}

void GBuffer::render(EngineData& engineData, SceneData& sceneData,
                     std::function<void(VkCommandBuffer cmd)>&& function)
{
    _currFrame++;

    GbufferData* current_data = get_current_frame_data();

    VkClearValue zeroColor = {.color = {{0.0f, 0.0f, 0.0f, 0.0f}}};
    VkClearValue materialGbufferColor = {.color = {{0.0f, 0.0f, 0.0f, -1.0f}}};
    VkClearValue depthColor = {.depthStencil = {1.0f, 0}};

    engineData.renderGraph->add_render_pass(
        {.name = "GBufferPass",
         .pipelineType = Vrg::PipelineType::RASTER_TYPE,
         .rasterPipeline = {.vertexShader = "../shaders/gbuffer.vert",
                            .fragmentShader = "../shaders/gbuffer.frag",
                            .size = _imageSize,
                            .blendAttachmentStates =
                                {
                                    vkinit::color_blend_attachment_state(),
                                    vkinit::color_blend_attachment_state(),
                                    vkinit::color_blend_attachment_state(),
                                    vkinit::color_blend_attachment_state(),
                                    vkinit::color_blend_attachment_state(),
                                },
                            .vertexBuffers = {sceneData.vertexBufferBinding,
                                              sceneData.normalBufferBinding,
                                              sceneData.texBufferBinding,
                                              sceneData.lightmapTexBufferBinding,
                                              sceneData.tangentBufferBinding},
                            .indexBuffer = sceneData.indexBufferBinding,
                            .colorOutputs =
                                {
                                    {current_data->albedoMetallicBinding, zeroColor},
                                    {current_data->normalBinding, zeroColor},
                                    {current_data->motionBinding, zeroColor},
                                    {current_data->roughnessDepthCurvatureMaterialBinding,
                                     materialGbufferColor},
                                    {current_data->uvBinding, zeroColor},
                                },
                            .depthOutput = {current_data->depthBinding, depthColor}},
         .reads =
             {
                 {0, sceneData.cameraBufferBinding},
                 {1, sceneData.objectBufferBinding},
                 {3, sceneData.materialBufferBinding},
             },
         .extraDescriptorSets = {{2, sceneData.textureDescriptor, sceneData.textureSetLayout}},
         .execute = function});
}

GbufferData* GBuffer::get_current_frame_data()
{
    return &_gbufferdata[_currFrame % 2];
}

GbufferData* GBuffer::get_previous_frame_data()
{
    return &_gbufferdata[(_currFrame - 1) % 2];
}
