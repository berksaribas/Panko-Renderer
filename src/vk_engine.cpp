#define TINYGLTF_IMPLEMENTATION
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include "vk_pipeline.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <glm/gtx/transform.hpp>

#include <precalculation.h>
#include <vk_extensions.h>
#include <vk_utils.h>
#include <xatlas.h>

#include <gi_brdf.h>
#include <gi_deferred.h>
#include <gi_diffuse.h>
#include <gi_gbuffer.h>
#include <gi_glossy.h>
#include <gi_glossy_svgf.h>
#include <gi_shadow.h>
#include <vk_timer.h>

#include <ctime>

#include "VkBootstrap.h"
#include <functional>
#include <vk_debug_renderer.h>

#define FILE_HELPER_IMPL
#include <file_helper.h>

#include "editor\editor.h"
#include "vk_rendergraph.h"
#include "vk_super_res.h"

constexpr bool bUseValidationLayers = true;
std::vector<GPUBasicMaterialData> materials;

// Precalculation
Precalculation precalculation;
PrecalculationInfo precalculationInfo = {};
PrecalculationLoadData precalculationLoadData = {};
PrecalculationResult precalculationResult = {};

// GI Models
BRDF brdfUtils;
GBuffer gbuffer;
DiffuseIllumination diffuseIllumination;
Shadow shadow;
GlossyIllumination glossyIllumination;
GlossyDenoise glossyDenoise;
Deferred deferred;

Editor editor;
CameraConfig cameraConfig;
Camera camera = {glm::vec3(0, 0, 28.5), glm::vec3(0, 0, 0)};

vkb::Swapchain vkbSwapchain;
bool swapchainInitialized = false;
bool swapchainNeedsRecreation = false;

SuperResolution superResolution;

void VulkanEngine::init()
{
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window =
        SDL_CreateWindow("Panko", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                         _displayResolution.width, _displayResolution.height, window_flags);

    _engineData = {};
    _shaderManager.initialize();

    init_vulkan();
    _engineData.renderGraph = new Vrg::RenderGraph(&_engineData, &_shaderManager);

    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptor_pool();

    _vulkanCompute.init(_engineData);
    _vulkanRaytracing.init(_engineData, _gpuRaytracingProperties);
    _engineData.renderGraph->enable_raytracing(&_vulkanRaytracing);

    editor.initialize(_engineData, _window, _swachainImageFormat);

    bool loadPrecomputedData = false;
    if (!loadPrecomputedData)
    {
        precalculationInfo.voxelSize = 0.25;
        precalculationInfo.voxelPadding = 2;
        precalculationInfo.probeOverlaps = 10;
        precalculationInfo.raysPerProbe = 1000;
        precalculationInfo.raysPerReceiver = 40000;
        precalculationInfo.sphericalHarmonicsOrder = 7;
        precalculationInfo.clusterCoefficientCount = 32;
        precalculationInfo.maxReceiversInCluster = 1024;
        precalculationInfo.lightmapResolution = 261;
        precalculationInfo.texelSize = 6;
        precalculationInfo.desiredSpacing = 2;
    }
    else
    {
        precalculation.load("../precomputation/precalculation.cfg", precalculationInfo,
                            precalculationLoadData, precalculationResult);
    }

    init_scene();

    if (!loadPrecomputedData)
    {
        precalculation.prepare(*this, gltf_scene, precalculationInfo, precalculationLoadData,
                               precalculationResult);
        // precalculation.prepare(*this, gltf_scene, precalculationInfo,
        // precalculationLoadData, precalculationResult,
        // "../precomputation/precalculation.Probes");
        exit(0);
    }

    init_query_pool();
    init_descriptors();

    brdfUtils.init_images(_engineData);
    shadow.init_buffers(_engineData);
    shadow.init_images(_engineData);
    gbuffer.init_images(_engineData, _renderResolution);
    diffuseIllumination.init(_engineData, &precalculationInfo, &precalculationLoadData,
                             &precalculationResult, gltf_scene);
    glossyIllumination.init_images(_engineData, _renderResolution);
    glossyDenoise.init_images(_engineData, _renderResolution);
    deferred.init_images(_engineData, _renderResolution);
    _vulkanDebugRenderer.init(_engineData);

    shadow._shadowMapData.positiveExponent = 40;
    shadow._shadowMapData.negativeExponent = 5;
    shadow._shadowMapData.LightBleedingReduction = 0.999f;
    shadow._shadowMapData.VSMBias = 0.01;
    _camData.lightPos = {0.020, 2, 0.140, 0.0f};
    _camData.lightColor = {1.f, 1.f, 1.f, 1.0f};
    _camData.indirectDiffuse = true;
    _camData.indirectSpecular = true;
    _camData.useStochasticSpecular = true;
    _camData.clearColor = {0.f, 0.f, 0.f, 0.0f};
    camera.pos = {0, 0, 7};
    _camData.glossyFrameCount = 0;
    _camData.glossyDenoise = 1;

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {
        // make sure the GPU has stopped doing its things
        vkDeviceWaitIdle(_engineData.device);

        // TODO
        //_engineData.renderGraph->clear();
        _mainDeletionQueue.flush();

        vmaDestroyAllocator(_engineData.allocator);

        vkDestroySurfaceKHR(_engineData.instance, _surface, nullptr);

        vkDestroyDevice(_engineData.device, nullptr);
        vkDestroyInstance(_engineData.instance, nullptr);

        SDL_DestroyWindow(_window);
    }
}

static char buffer[256];

void VulkanEngine::draw(double deltaTime)
{
    // wait until the gpu has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(_engineData.device, 1, &_renderFence, true, 1000000000));

    // request image from the swapchain
    uint32_t swapchainImageIndex;
    VkResult swapchainResult =
        vkAcquireNextImageKHR(_engineData.device, _swapchain, 1000000000, _presentSemaphore,
                              nullptr, &swapchainImageIndex);

    if (swapchainResult == VK_ERROR_OUT_OF_DATE_KHR || swapchainResult == VK_SUBOPTIMAL_KHR ||
        swapchainNeedsRecreation)
    {
        init_swapchain();
        return;
    }
    else if (swapchainResult != VK_SUCCESS)
    {
        abort();
    }

    VK_CHECK(vkResetFences(_engineData.device, 1, &_renderFence));

    // TODO: Enable
    _engineData.renderGraph->vkTimer.get_results(_engineData);

    // now that we are sure that the commands finished executing, we can safely reset the
    // command buffer to begin recording again.
    VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

    constexpr glm::vec3 UP = glm::vec3(0, 1, 0);
    constexpr glm::vec3 RIGHT = glm::vec3(1, 0, 0);
    constexpr glm::vec3 FORWARD = glm::vec3(0, 0, 1);
    auto res = glm::mat4{1};

    float fov = glm::radians(cameraConfig.fov);

    if (editor.editorSettings.useSceneCamera)
    {
        res = glm::scale(glm::vec3(_sceneScale)) * gltf_scene.cameras[0].world_matrix;
        camera.pos = gltf_scene.cameras[0].eye * _sceneScale;
        fov = gltf_scene.cameras[0].cam.perspective.yfov;
    }
    else
    {
        res = glm::translate(res, camera.pos);
        res = glm::rotate(res, glm::radians(camera.rotation.y), UP);
        res = glm::rotate(res, glm::radians(camera.rotation.x), RIGHT);
        res = glm::rotate(res, glm::radians(camera.rotation.z), FORWARD);
    }

    auto view = glm::inverse(res);
    glm::mat4 projection =
        glm::perspective(fov, ((float)_renderResolution.width) / _renderResolution.height,
                         cameraConfig.nearPlane, cameraConfig.farPlane);
    SuperResolutionData superResData;
    superResolution.get_data(&superResData);

    const float jitterX = 2.0f * superResData.jitterX / (float)_renderResolution.width;
    const float jitterY = 2.0f * superResData.jitterY / (float)_renderResolution.height;
    glm::mat4 jitterTranslationMatrix =
        glm::translate(glm::mat4(1), glm::vec3(jitterX, jitterY, 0));
    projection = jitterTranslationMatrix * projection;

    projection[1][1] *= -1;

    _camData.prevJitter = _camData.jitter;
    _camData.jitter = {jitterX, jitterY};

    _camData.prevViewproj = _camData.viewproj;
    _camData.viewproj = projection * view;
    _camData.viewprojInverse = glm::inverse(_camData.viewproj);
    _camData.cameraPos = glm::vec4(camera.pos.x, camera.pos.y, camera.pos.z, 1.0);

    _camData.lightmapInputSize = {(float)gltf_scene.lightmap_width,
                                  (float)gltf_scene.lightmap_height};
    _camData.lightmapTargetSize = {diffuseIllumination._lightmapExtent.width,
                                   diffuseIllumination._lightmapExtent.height};

    _camData.frameCount = _frameNumber;

    glm::vec3 lightInvDir = glm::vec3(_camData.lightPos);
    float radius = gltf_scene.m_dimensions.max.x > gltf_scene.m_dimensions.max.y
                       ? gltf_scene.m_dimensions.max.x
                       : gltf_scene.m_dimensions.max.y;
    radius = radius > gltf_scene.m_dimensions.max.z ? radius : gltf_scene.m_dimensions.max.z;
    // radius *= sqrt(2);
    glm::mat4 depthViewMatrix =
        glm::lookAt(gltf_scene.m_dimensions.center * _sceneScale + lightInvDir * radius,
                    gltf_scene.m_dimensions.center * _sceneScale, glm::vec3(0, 1, 0));
    float maxX = gltf_scene.m_dimensions.min.x * _sceneScale,
          maxY = gltf_scene.m_dimensions.min.y * _sceneScale,
          maxZ = gltf_scene.m_dimensions.min.z * _sceneScale;
    float minX = gltf_scene.m_dimensions.max.x * _sceneScale,
          minY = gltf_scene.m_dimensions.max.y * _sceneScale,
          minZ = gltf_scene.m_dimensions.max.z * _sceneScale;

    for (int x = 0; x < 2; x++)
    {
        for (int y = 0; y < 2; y++)
        {
            for (int z = 0; z < 2; z++)
            {
                float xCoord = x == 0 ? gltf_scene.m_dimensions.min.x * _sceneScale
                                      : gltf_scene.m_dimensions.max.x * _sceneScale;
                float yCoord = y == 0 ? gltf_scene.m_dimensions.min.y * _sceneScale
                                      : gltf_scene.m_dimensions.max.y * _sceneScale;
                float zCoord = z == 0 ? gltf_scene.m_dimensions.min.z * _sceneScale
                                      : gltf_scene.m_dimensions.max.z * _sceneScale;
                auto tempCoords = depthViewMatrix * glm::vec4(xCoord, yCoord, zCoord, 1.0);
                tempCoords = tempCoords / tempCoords.w;
                if (tempCoords.x < minX)
                {
                    minX = tempCoords.x;
                }
                if (tempCoords.x > maxX)
                {
                    maxX = tempCoords.x;
                }

                if (tempCoords.y < minY)
                {
                    minY = tempCoords.y;
                }
                if (tempCoords.y > maxY)
                {
                    maxY = tempCoords.y;
                }

                if (tempCoords.z < minZ)
                {
                    minZ = tempCoords.z;
                }
                if (tempCoords.z > maxZ)
                {
                    maxZ = tempCoords.z;
                }
            }
        }
    }

    glm::mat4 depthProjectionMatrix = glm::ortho(minX, maxX, minY, maxY, -maxZ, -minZ);

    shadow._shadowMapData.depthMVP = depthProjectionMatrix * depthViewMatrix;

    editor.prepare(_engineData);
    editor.prepare_debug_settings(_engineData);
    editor.prepare_camera_settings(_engineData, _camData, cameraConfig,
                                   gltf_scene.cameras.size() > 0);
    editor.prepare_performance_settings(_engineData);
    editor.prepare_material_settings(_engineData, _sceneData, materials.data(),
                                     materials.size());
    editor.prepare_object_settings(_engineData, gltf_scene.nodes.data(),
                                   gltf_scene.nodes.size());
    editor.prepare_renderer_settings(_engineData, _camData, shadow, glossyDenoise,
                                     _frameNumber);

    if (editor.editorSettings.showProbes)
    {
        diffuseIllumination.debug_draw_probes(
            _vulkanDebugRenderer, editor.editorSettings.showProbeRays, _sceneScale);
    }

    if (editor.editorSettings.showReceivers)
    {
        diffuseIllumination.debug_draw_receivers(_vulkanDebugRenderer, _sceneScale);
    }

    if (editor.editorSettings.showSpecificReceiver)
    {
        diffuseIllumination.debug_draw_specific_receiver(
            _vulkanDebugRenderer, editor.editorSettings.specificCluster,
            editor.editorSettings.specificReceiver,
            editor.editorSettings.specificReceiverRaySampleCount,
            editor.editorSettings.probesEnabled, editor.editorSettings.showSpecificProbeRays,
            _sceneScale);
    }

    vkutils::cpu_to_gpu(_engineData.allocator, _sceneData.cameraBuffer, &_camData,
                        sizeof(GPUCameraData));
    shadow.prepare_rendering(_engineData);

    void* objectData;
    vmaMapMemory(_engineData.allocator, _sceneData.objectBuffer._allocation, &objectData);
    GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

    for (int i = 0; i < gltf_scene.nodes.size(); i++)
    {
        auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
        glm::mat4 scale = glm::mat4{1};
        scale = glm::scale(scale, {_sceneScale, _sceneScale, _sceneScale});
        objectSSBO[i].model = scale * gltf_scene.nodes[i].world_matrix;
        objectSSBO[i].material_id = mesh.material_idx;
    }
    vmaUnmapMemory(_engineData.allocator, _sceneData.objectBuffer._allocation);

    _vulkanRaytracing.build_tlas(gltf_scene,
                                 VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                     VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                                 true);

    VkCommandBuffer cmd = _mainCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    {
        shadow.render(_engineData, _sceneData,
                      [&](VkCommandBuffer cmd) { draw_objects(cmd); });
        gbuffer.render(_engineData, _sceneData,
                       [&](VkCommandBuffer cmd) { draw_objects(cmd); });
        if (editor.editorSettings.enableGroundTruthDiffuse)
        {
            diffuseIllumination.render_ground_truth(cmd, _engineData, _sceneData, shadow,
                                                    brdfUtils);
        }
        else
        {
            diffuseIllumination.render(
                cmd, _engineData, _sceneData, shadow, brdfUtils,
                [&](VkCommandBuffer cmd) { draw_objects(cmd); },
                editor.editorSettings.useRealtimeRaycast,
                editor.editorSettings.numberOfBasisFunctions);
        }
        glossyIllumination.render(_engineData, _sceneData, gbuffer, shadow,
                                  diffuseIllumination, brdfUtils);

        if (_camData.useStochasticSpecular)
        {
            glossyDenoise.render(_engineData, _sceneData, gbuffer, glossyIllumination);

            if (editor.editorSettings.enableDenoise)
            {
                deferred.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination,
                                glossyIllumination, brdfUtils,
                                glossyDenoise.get_denoised_binding());
            }
            else
            {
                deferred.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination,
                                glossyIllumination, brdfUtils,
                                glossyIllumination._glossyReflectionsColorImageBinding);
            }
        }
        else
        {
            deferred.render(_engineData, _sceneData, gbuffer, shadow, diffuseIllumination,
                            glossyIllumination, brdfUtils,
                            glossyIllumination._glossyReflectionsColorImageBinding);
        }

        auto gbufferData = gbuffer.get_current_frame_data();

        _engineData.renderGraph->add_render_pass(
            {.name = "SuperResolutionPass",
             .pipelineType = Vrg::PipelineType::CUSTOM,
             .execute = [&](VkCommandBuffer cmd) {
                 superResolution.dispatch(
                     _engineData, cmd, deferred._deferredColorImageBinding,
                     gbufferData->depthBinding, gbufferData->motionBinding, &cameraConfig,
                     deltaTime);
             }});

        _vulkanDebugRenderer.render(_engineData, _sceneData, _displayResolution,
                                    _swapchainBindings[swapchainImageIndex]);

        VkClearValue clearValue;
        clearValue.color = {{1.0f, 1.0f, 1.0f, 1.0f}};

        _engineData.renderGraph->add_render_pass(
            {.name = "PresentPass",
             .pipelineType = Vrg::PipelineType::RASTER_TYPE,
             .rasterPipeline =
                 {
                     .vertexShader = "../shaders/fullscreen.vert",
                     .fragmentShader = "../shaders/gamma.frag",
                     .size = _displayResolution,
                     .depthState = {false, false, VK_COMPARE_OP_NEVER},
                     .cullMode = Vrg::CullMode::NONE,
                     .blendAttachmentStates =
                         {
                             vkinit::color_blend_attachment_state(),
                         },
                     .colorOutputs =
                         {
                             {_swapchainBindings[swapchainImageIndex], clearValue, true},
                         },

                 },
             .reads = {{0, editor.editorSettings.selectedRenderBinding.isValid()
                               ? editor.editorSettings.selectedRenderBinding
                               : superResolution.outputBindable}},
             .execute = [&](VkCommandBuffer cmd) {
                 vkCmdDraw(cmd, 3, 1, 0, 0);
                 _vulkanDebugRenderer.custom_execute(cmd, _engineData);
                 editor.render(cmd);
             }});
    }

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    {
        _engineData.renderGraph->execute(cmd);
    }

    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare the submission to the queue.
    // we want to wait on the _presentSemaphore, as that semaphore is signaled when the
    // swapchain is ready we will signal the _renderSemaphore, to signal that rendering has
    // finished

    VkSubmitInfo submit = vkinit::submit_info(&cmd);
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    submit.pWaitDstStageMask = &waitStage;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &_presentSemaphore;

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &_renderSemaphore;

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit(_engineData.graphicsQueue, 1, &submit, _renderFence));

    // prepare present
    //  this will put the image we just rendered to into the visible window.
    //  we want to wait on the _renderSemaphore for that,
    //  as its necessary that drawing commands have finished before the image is displayed to
    //  the user
    VkPresentInfoKHR presentInfo = vkinit::present_info();

    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &_renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    swapchainResult = vkQueuePresentKHR(_engineData.graphicsQueue, &presentInfo);

    if (swapchainResult == VK_ERROR_OUT_OF_DATE_KHR || swapchainResult == VK_SUBOPTIMAL_KHR)
    {
        init_swapchain();
        return;
    }
    else if (swapchainResult != VK_SUCCESS)
    {
        abort();
    }

    if (editor.editorSettings.screenshot)
    {
        editor.editorSettings.screenshot = false;
        std::time_t screenshot = std::time(0);
        sprintf_s(buffer, "screenshots/%d-%s.png", screenshot,
                  editor.editorSettings.customName);
        vkutils::screenshot(&_engineData, buffer, _swapchainImages[swapchainImageIndex],
                            _displayResolution);
        ScreenshotSaveData saveData = {camera, _camData.lightPos};
        sprintf_s(buffer, "screenshots/%d-%s.cam", screenshot,
                  editor.editorSettings.customName);
        save_binary(buffer, &saveData, sizeof(ScreenshotSaveData));
        memset(editor.editorSettings.customName, 0, sizeof(editor.editorSettings.customName));
    }

    // increase the number of frames drawn
    _frameNumber++;
    _camData.glossyFrameCount++;
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;
    bool onGui = true;
    SDL_SetRelativeMouseMode((SDL_bool)!onGui);

    Uint64 NOW = SDL_GetPerformanceCounter();
    Uint64 LAST = 0;
    double deltaTime = 0;

    bool isFullScreen = false;
    VkExtent2D currentWindowSize = _displayResolution;

    // main loop
    while (!bQuit)
    {
        bool moved = false;

        const Uint8* keys = SDL_GetKeyboardState(NULL);

        while (SDL_PollEvent(&e) != 0)
        {
            editor.retrieve_input(&e);

            if (e.type == SDL_QUIT)
            {
                bQuit = true;
            }
            if (e.type == SDL_MOUSEMOTION && !onGui)
            {
                SDL_MouseMotionEvent motion = e.motion;

                camera.rotation.x += -cameraConfig.rotationSpeed * (float)motion.yrel;
                camera.rotation.y += -cameraConfig.rotationSpeed * (float)motion.xrel;
                moved = true;
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_LCTRL)
            {
                onGui = !onGui;
                SDL_SetRelativeMouseMode((SDL_bool)!onGui);
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F11)
            {
                if (!isFullScreen)
                {
                    currentWindowSize = _displayResolution;
                    SDL_SetWindowFullscreen(_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
                }
                else
                {
                    SDL_SetWindowFullscreen(_window, 0);
                    SDL_SetWindowSize(_window, currentWindowSize.width,
                                      currentWindowSize.height);
                }
                isFullScreen = !isFullScreen;
            }

            if (e.type == SDL_WINDOWEVENT)
            {
                if (e.window.event == SDL_WINDOWEVENT_RESIZED)
                {
                    swapchainNeedsRecreation = true;
                }
            }
        }

        glm::vec3 front;
        front.x = cos(glm::radians(camera.rotation.x)) * sin(glm::radians(camera.rotation.y));
        front.y = -sin(glm::radians(camera.rotation.x));
        front.z = cos(glm::radians(camera.rotation.x)) * cos(glm::radians(camera.rotation.y));
        front = glm::normalize(-front);

        float speed = cameraConfig.speed;

        if (!onGui)
        {
            if (keys[SDL_SCANCODE_LSHIFT])
            {
                speed *= 4;
            }
            if (keys[SDL_SCANCODE_W])
            {
                camera.pos += front * speed;
                moved = true;
            }
            if (keys[SDL_SCANCODE_S])
            {
                camera.pos -= front * speed;
                moved = true;
            }
            if (keys[SDL_SCANCODE_A])
            {
                camera.pos -=
                    glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f))) * speed;
                moved = true;
            }
            if (keys[SDL_SCANCODE_D])
            {
                camera.pos +=
                    glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f))) * speed;
                moved = true;
            }
            if (keys[SDL_SCANCODE_P])
            {
                editor.editorSettings.screenshot = true;
            }
        }

        LAST = NOW;
        NOW = SDL_GetPerformanceCounter();

        deltaTime = (double)((NOW - LAST) * 1000 / (double)SDL_GetPerformanceFrequency());

        draw(deltaTime);
    }
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    // make the vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Panko Renderer")
                        .request_validation_layers(bUseValidationLayers)
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0)
                        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    // grab the instance
    _engineData.instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _engineData.instance, &_surface);

    VkPhysicalDeviceFeatures physicalDeviceFeatures = VkPhysicalDeviceFeatures();
    physicalDeviceFeatures.fillModeNonSolid = VK_TRUE;
    physicalDeviceFeatures.samplerAnisotropy = VK_TRUE;
    physicalDeviceFeatures.shaderInt64 = VK_TRUE;
    physicalDeviceFeatures.shaderInt16 = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR featureRt = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR featureAccel = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features features13 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features features12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features features11 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};

    features11.shaderDrawParameters = VK_TRUE;
    features11.pNext = &features12;

    features12.bufferDeviceAddress = VK_TRUE;
    features12.hostQueryReset = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    features12.pNext = &dynamic_rendering_feature;
    features12.shaderFloat16 = VK_TRUE;

    features13.dynamicRendering = VK_TRUE;
    features13.pNext = &featureRt;
    dynamic_rendering_feature.dynamicRendering = VK_TRUE;
    dynamic_rendering_feature.pNext = &featureRt;

    featureRt.rayTracingPipeline = VK_TRUE;
    featureRt.pNext = &featureAccel;

    featureAccel.accelerationStructure = VK_TRUE;
    featureAccel.pNext = nullptr;

    // use vkbootstrap to select a gpu.
    // We want a gpu that can write to the SDL surface and supports vulkan 1.2
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    auto physicalDeviceSelectionResult =
        selector.set_minimum_version(1, 3)
            .set_surface(_surface)
            .set_required_features(physicalDeviceFeatures)
            .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
            .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
            .add_required_extension(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME)
            .select();

    // printf(physicalDeviceSelectionResult.error().message().c_str());

    auto physicalDevice = physicalDeviceSelectionResult.value();

    // create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    deviceBuilder.add_pNext(&features11);
    vkb::Device vkbDevice = deviceBuilder.build().value();

    _engineData.device = vkbDevice.device;
    _engineData.physicalDevice = physicalDevice.physical_device;

    // Get extension pointers
    load_VK_EXTENSIONS(_engineData.instance, vkGetInstanceProcAddr, _engineData.device,
                       vkGetDeviceProcAddr);

    // use vkbootstrap to get a Graphics queue
    _engineData.graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _engineData.graphicsQueueFamily =
        vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    _engineData.computeQueue = vkbDevice.get_queue(vkb::QueueType::compute).value();
    _engineData.computeQueueFamily =
        vkbDevice.get_queue_index(vkb::QueueType::compute).value();

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _engineData.physicalDevice;
    allocatorInfo.device = _engineData.device;
    allocatorInfo.instance = _engineData.instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_engineData.allocator);

    _gpuRaytracingProperties = {};
    _gpuProperties = {};

    _gpuRaytracingProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    _gpuProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    _gpuProperties.pNext = &_gpuRaytracingProperties;
    vkGetPhysicalDeviceProperties2(_engineData.physicalDevice, &_gpuProperties);
    vkGetPhysicalDeviceFeatures(_engineData.physicalDevice, &_gpuFeatures);
}

void VulkanEngine::init_swapchain()
{
    vkDeviceWaitIdle(_engineData.device);

    vkb::SwapchainBuilder swapchainBuilder{_engineData.physicalDevice, _engineData.device,
                                           _surface};

    if (swapchainInitialized)
    {
        vkb::destroy_swapchain(vkbSwapchain);

        for (int i = 0; i < _swapchainAllocatedImage.size(); i++)
        {
            _engineData.renderGraph->destroy_resource(_swapchainAllocatedImage[i]);
        }
    }

    vkbSwapchain = swapchainBuilder
                       .use_default_format_selection()
                       // use vsync present mode
                       .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                       //.set_desired_extent(_windowExtent.width, _windowExtent.height)
                       //.set_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                       .build()
                       .value();

    _displayResolution = vkbSwapchain.extent;

    // store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainAllocatedImage.clear();

    for (int i = 0; i < _swapchainImages.size(); i++)
    {
        AllocatedImage allocatedImage;
        allocatedImage.format = vkbSwapchain.image_format;
        allocatedImage._image = _swapchainImages[i];
        allocatedImage._allocation = nullptr;
        _swapchainAllocatedImage.push_back(allocatedImage);
    }

    _swapchainBindings.resize(3);

    for (int i = 0; i < _swapchainImages.size(); i++)
    {
        _swapchainBindings[i] = _engineData.renderGraph->register_image_view(
            &_swapchainAllocatedImage[i],
            {.sampler = Vrg::Sampler::LINEAR, .baseMipLevel = 0, .mipLevelCount = 1},
            "Swapchain" + std::to_string(i));
    }

    _swachainImageFormat = vkbSwapchain.image_format;

    superResolution.initialize(_engineData, _renderResolution, _displayResolution);

    _mainDeletionQueue.push_function(
        [=]() { vkDestroySwapchainKHR(_engineData.device, _swapchain, nullptr); });

    swapchainInitialized = true;
    swapchainNeedsRecreation = false;
}

void VulkanEngine::init_commands()
{
    // create a command pool for commands submitted to the graphics queue.
    // we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
        _engineData.graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    VK_CHECK(vkCreateCommandPool(_engineData.device, &commandPoolInfo, nullptr, &commandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_engineData.device, &cmdAllocInfo, &_mainCommandBuffer));

    _mainDeletionQueue.push_function(
        [=]() { vkDestroyCommandPool(_engineData.device, commandPool, nullptr); });

    // create pool for upload context
    VkCommandPoolCreateInfo uploadCommandPoolInfo =
        vkinit::command_pool_create_info(_engineData.graphicsQueueFamily);
    VK_CHECK(vkCreateCommandPool(_engineData.device, &uploadCommandPoolInfo, nullptr,
                                 &_engineData.uploadContext.commandPool));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_engineData.device, _engineData.uploadContext.commandPool,
                             nullptr);
    });
}

void VulkanEngine::init_sync_structures()
{
    // create syncronization structures
    // one fence to control when the gpu has finished rendering the frame,
    // and 2 semaphores to syncronize rendering with swapchain
    // we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo =
        vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    {
        VK_CHECK(vkCreateFence(_engineData.device, &fenceCreateInfo, nullptr, &_renderFence));

        // enqueue the destruction of the fence
        _mainDeletionQueue.push_function(
            [=]() { vkDestroyFence(_engineData.device, _renderFence, nullptr); });

        VK_CHECK(vkCreateSemaphore(_engineData.device, &semaphoreCreateInfo, nullptr,
                                   &_presentSemaphore));
        VK_CHECK(vkCreateSemaphore(_engineData.device, &semaphoreCreateInfo, nullptr,
                                   &_renderSemaphore));

        // enqueue the destruction of semaphores
        _mainDeletionQueue.push_function([=]() {
            vkDestroySemaphore(_engineData.device, _presentSemaphore, nullptr);
            vkDestroySemaphore(_engineData.device, _renderSemaphore, nullptr);
        });
    }

    VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();
    VK_CHECK(vkCreateFence(_engineData.device, &uploadFenceCreateInfo, nullptr,
                           &_engineData.uploadContext.fence));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_engineData.device, _engineData.uploadContext.fence, nullptr);
    });
}

void VulkanEngine::init_descriptor_pool()
{
    std::vector<VkDescriptorPoolSize> sizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000}};

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)sizes.size();
    pool_info.pPoolSizes = sizes.data();
    vkCreateDescriptorPool(_engineData.device, &pool_info, nullptr,
                           &_engineData.descriptorPool);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyDescriptorPool(_engineData.device, _engineData.descriptorPool, nullptr);
    });
}

void VulkanEngine::init_descriptors()
{
    const int MAX_OBJECTS = 10000;

    _sceneData.objectBuffer = vkutils::create_buffer(
        _engineData.allocator, sizeof(GPUObjectData) * MAX_OBJECTS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    _sceneData.cameraBuffer = vkutils::create_buffer(
        _engineData.allocator, sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    _sceneData.objectBufferBinding = _engineData.renderGraph->register_storage_buffer(
        &_sceneData.objectBuffer, "ObjectBuffer");
    _sceneData.cameraBufferBinding = _engineData.renderGraph->register_uniform_buffer(
        &_sceneData.cameraBuffer, "CameraBuffer");

    VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);
    VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
    VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);

    VkDescriptorSetLayoutBinding bindings[3] = {tlasBind, sceneDescBind, meshInfoBind};
    VkDescriptorSetLayoutCreateInfo setinfo =
        vkinit::descriptorset_layout_create_info(bindings, 3);
    vkCreateDescriptorSetLayout(_engineData.device, &setinfo, nullptr,
                                &_sceneData.raytracingSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo = vkinit::descriptorset_allocate_info(
        _engineData.descriptorPool, &_sceneData.raytracingSetLayout, 1);

    vkAllocateDescriptorSets(_engineData.device, &allocateInfo,
                             &_sceneData.raytracingDescriptor);

    std::vector<VkWriteDescriptorSet> writes;

    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &_vulkanRaytracing.tlas.accel;
    VkWriteDescriptorSet accelerationStructureWrite{};
    accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    // The specialized acceleration structure descriptor has to be chained
    accelerationStructureWrite.pNext = &descASInfo;
    accelerationStructureWrite.dstSet = _sceneData.raytracingDescriptor;
    accelerationStructureWrite.dstBinding = 0;
    accelerationStructureWrite.descriptorCount = 1;
    accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes.emplace_back(accelerationStructureWrite);
    writes.emplace_back(vkinit::write_descriptor_buffer(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _sceneData.raytracingDescriptor,
        &_sceneData.sceneDescBuffer._descriptorBufferInfo, 1));
    writes.emplace_back(vkinit::write_descriptor_buffer(
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _sceneData.raytracingDescriptor,
        &_sceneData.meshInfoBuffer._descriptorBufferInfo, 2));

    vkUpdateDescriptorSets(_engineData.device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

void VulkanEngine::init_scene()
{
    std::string file_name = "../assets/cornellFixed.gltf";
    // std::string file_name = "../assets/cornellsuzanne.gltf";
    // std::string file_name = "../assets/occluderscene.gltf";
    // std::string file_name = "../assets/reflection_new.gltf";
    // std::string file_name = "../assets/shtest.gltf";
    // std::string file_name = "../assets/bedroom/bedroom.gltf";
    // std::string file_name = "../assets/livingroom/livingroom.gltf";
    // std::string file_name = "../assets/picapica/scene.gltf";
    // std::string file_name = "D:/newsponza/combined/sponza.gltf";

    tinygltf::Model tmodel;
    tinygltf::TinyGLTF tcontext;

    std::string warn, error;
    if (!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, file_name))
    {
        assert(!"Error while loading scene");
    }
    if (!warn.empty())
    {
        printf("WARNING: SCENE LOADING: %s\n", warn.c_str());
    }
    if (!error.empty())
    {
        printf("WARNING: SCENE LOADING: %s\n", error.c_str());
    }

    gltf_scene.import_materials(tmodel);
    gltf_scene.import_drawable_nodes(
        tmodel, GltfAttributes::Normal | GltfAttributes::Texcoord_0 | GltfAttributes::Tangent);

    printf("dimensions: %f %f %f\n", gltf_scene.m_dimensions.size.x,
           gltf_scene.m_dimensions.size.y, gltf_scene.m_dimensions.size.z);

    for (int i = 0; i < gltf_scene.materials.size(); i++)
    {
        GPUBasicMaterialData material = {};
        material.base_color = gltf_scene.materials[i].base_color_factor;
        material.emissive_color = gltf_scene.materials[i].emissive_factor;
        material.metallic_factor = gltf_scene.materials[i].metallic_factor;
        material.roughness_factor = gltf_scene.materials[i].roughness_factor;
        material.texture = gltf_scene.materials[i].base_color_texture;
        material.normal_texture = gltf_scene.materials[i].normal_texture;
        material.metallic_roughness_texture =
            gltf_scene.materials[i].metallic_rougness_texture;
        materials.push_back(material);
    }

    // TODO: Try to load lightmap uvs from disk first, if not available generate them
    /*
     * File format:
     * atlas width (4 bytes), atlas height (4 bytes)
     * all the data
     * vec2 data as byte
     */
    printf("Generating lightmap uvs\n");
    xatlas::Atlas* atlas = xatlas::Create();
    for (int i = 0; i < gltf_scene.nodes.size(); i++)
    {
        auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
        xatlas::MeshDecl meshDecleration = {};
        meshDecleration.vertexPositionData = &gltf_scene.positions[mesh.vtx_offset];
        meshDecleration.vertexPositionStride = sizeof(glm::vec3);
        meshDecleration.vertexCount = mesh.vtx_count;

        if (gltf_scene.texcoords0.size() > 0)
        {
            meshDecleration.vertexUvData = &gltf_scene.texcoords0[mesh.vtx_offset];
            meshDecleration.vertexUvStride = sizeof(glm::vec2);
        }

        meshDecleration.indexData = &gltf_scene.indices[mesh.first_idx];
        meshDecleration.indexCount = mesh.idx_count;
        meshDecleration.indexFormat = xatlas::IndexFormat::UInt32;
        xatlas::AddMesh(atlas, meshDecleration);
    }

    xatlas::ChartOptions chartOptions = xatlas::ChartOptions();
    // chartOptions.fixWinding = true;
    xatlas::PackOptions packOptions = xatlas::PackOptions();
    packOptions.texelsPerUnit = precalculationInfo.texelSize;
    packOptions.bilinear = true;
    packOptions.padding = 1;
    xatlas::Generate(atlas, chartOptions, packOptions);

    gltf_scene.lightmap_width = atlas->width;
    gltf_scene.lightmap_height = atlas->height;

    std::vector<GltfPrimMesh> prim_meshes;
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords0;
    std::vector<glm::vec4> tangents;

    for (int i = 0; i < gltf_scene.nodes.size(); i++)
    {
        GltfPrimMesh mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
        uint32_t orihinal_vtx_offset = mesh.vtx_offset;
        mesh.first_idx = indices.size();
        mesh.vtx_offset = positions.size();

        mesh.idx_count = atlas->meshes[i].indexCount;
        mesh.vtx_count = atlas->meshes[i].vertexCount;

        gltf_scene.nodes[i].prim_mesh = prim_meshes.size();
        prim_meshes.push_back(mesh);

        for (int j = 0; j < atlas->meshes[i].vertexCount; j++)
        {
            gltf_scene.lightmapUVs.push_back({atlas->meshes[i].vertexArray[j].uv[0],
                                              atlas->meshes[i].vertexArray[j].uv[1]});

            positions.push_back(gltf_scene.positions[atlas->meshes[i].vertexArray[j].xref +
                                                     orihinal_vtx_offset]);
            normals.push_back(gltf_scene.normals[atlas->meshes[i].vertexArray[j].xref +
                                                 orihinal_vtx_offset]);
            texcoords0.push_back(gltf_scene.texcoords0[atlas->meshes[i].vertexArray[j].xref +
                                                       orihinal_vtx_offset]);
            tangents.push_back(gltf_scene.tangents[atlas->meshes[i].vertexArray[j].xref +
                                                   orihinal_vtx_offset]);
        }

        for (int j = 0; j < atlas->meshes[i].indexCount; j++)
        {
            indices.push_back(atlas->meshes[i].indexArray[j]);
        }
    }

    gltf_scene.prim_meshes.clear();
    gltf_scene.positions.clear();
    gltf_scene.indices.clear();
    gltf_scene.normals.clear();
    gltf_scene.texcoords0.clear();
    gltf_scene.tangents.clear();

    gltf_scene.prim_meshes = prim_meshes;
    gltf_scene.positions = positions;
    gltf_scene.indices = indices;
    gltf_scene.normals = normals;
    gltf_scene.texcoords0 = texcoords0;
    gltf_scene.tangents = tangents;

    printf("Generated lightmap uvs, %d x %d\n", atlas->width, atlas->height);
    xatlas::Destroy(atlas);

    /*
    --
    */
    // materials.push_back(materials[materials.size() - 1]);
    // auto new_node = gltf_scene.nodes[gltf_scene.nodes.size() - 1];
    // auto new_mesh = gltf_scene.prim_meshes[new_node.prim_mesh];
    // new_mesh.material_idx = materials.size() - 1;
    // gltf_scene.prim_meshes.push_back(new_mesh);
    // new_node.prim_mesh = gltf_scene.prim_meshes.size() - 1;
    // gltf_scene.nodes.push_back(new_node);

    /*
     * TODO: After reading the GLTF scene, what I can do is:
     * Create the xatlas
     * Create a new vertex buffer (also normal and tex)
     * Correct the node data
     */

    _sceneData.vertexBuffer = vkutils::create_upload_buffer(
        &_engineData, gltf_scene.positions.data(),
        gltf_scene.positions.size() * sizeof(glm::vec3),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VMA_MEMORY_USAGE_GPU_ONLY);

    _sceneData.indexBuffer = vkutils::create_upload_buffer(
        &_engineData, gltf_scene.indices.data(), gltf_scene.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VMA_MEMORY_USAGE_GPU_ONLY);

    _sceneData.normalBuffer = vkutils::create_upload_buffer(
        &_engineData, gltf_scene.normals.data(), gltf_scene.normals.size() * sizeof(glm::vec3),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    _sceneData.tangentBuffer = vkutils::create_upload_buffer(
        &_engineData, gltf_scene.tangents.data(),
        gltf_scene.tangents.size() * sizeof(glm::vec4),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    _sceneData.texBuffer = vkutils::create_upload_buffer(
        &_engineData, gltf_scene.texcoords0.data(),
        gltf_scene.texcoords0.size() * sizeof(glm::vec2),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    _sceneData.lightmapTexBuffer = vkutils::create_upload_buffer(
        &_engineData, gltf_scene.lightmapUVs.data(),
        gltf_scene.lightmapUVs.size() * sizeof(glm::vec2),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    _sceneData.materialBuffer = vkutils::create_upload_buffer(
        &_engineData, materials.data(), materials.size() * sizeof(GPUBasicMaterialData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    _sceneData.vertexBufferBinding = _engineData.renderGraph->register_vertex_buffer(
        &_sceneData.vertexBuffer, VK_FORMAT_R32G32B32_SFLOAT, "VertexBuffer");
    _sceneData.indexBufferBinding = _engineData.renderGraph->register_index_buffer(
        &_sceneData.indexBuffer, VK_FORMAT_R32_UINT, "IndexBuffer");
    _sceneData.normalBufferBinding = _engineData.renderGraph->register_vertex_buffer(
        &_sceneData.normalBuffer, VK_FORMAT_R32G32B32_SFLOAT, "NormalBuffer");
    _sceneData.tangentBufferBinding = _engineData.renderGraph->register_vertex_buffer(
        &_sceneData.tangentBuffer, VK_FORMAT_R32G32B32A32_SFLOAT, "TangentBuffer");
    _sceneData.texBufferBinding = _engineData.renderGraph->register_vertex_buffer(
        &_sceneData.texBuffer, VK_FORMAT_R32G32_SFLOAT, "TexBuffer");
    _sceneData.lightmapTexBufferBinding = _engineData.renderGraph->register_vertex_buffer(
        &_sceneData.lightmapTexBuffer, VK_FORMAT_R32G32_SFLOAT, "LightmapTexBuffer");
    _sceneData.materialBufferBinding = _engineData.renderGraph->register_storage_buffer(
        &_sceneData.materialBuffer, "MaterialBuffer");

    std::vector<VkDescriptorImageInfo> image_infos;

    VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
    samplerInfo.minLod = 0.0f; // Optional
    samplerInfo.maxLod = FLT_MAX;
    samplerInfo.mipLodBias = 0.0f; // Optional

    samplerInfo.anisotropyEnable = _gpuFeatures.samplerAnisotropy;
    samplerInfo.maxAnisotropy = _gpuFeatures.samplerAnisotropy
                                    ? _gpuProperties.properties.limits.maxSamplerAnisotropy
                                    : 1.0f;

    VkSampler blockySampler;
    vkCreateSampler(_engineData.device, &samplerInfo, nullptr, &blockySampler);
    std::array<uint8_t, 4> nil = {0, 0, 0, 0};

    if (tmodel.textures.size() == 0)
    {
        AllocatedImage allocated_image;
        uint32_t mipLevels;
        vkutils::load_image_from_memory(&_engineData, nil.data(), 1, 1, allocated_image,
                                        mipLevels);

        VkImageView imageView;
        VkImageViewCreateInfo imageinfo =
            vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, allocated_image._image,
                                          VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
        vkCreateImageView(_engineData.device, &imageinfo, nullptr, &imageView);

        VkDescriptorImageInfo imageBufferInfo;
        imageBufferInfo.sampler = blockySampler;
        imageBufferInfo.imageView = imageView;
        imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(imageBufferInfo);
    }

    for (int i = 0; i < tmodel.textures.size(); i++)
    {

        auto& gltf_img = tmodel.images[tmodel.textures[i].source];
        AllocatedImage allocated_image;
        uint32_t mipLevels;

        if (gltf_img.image.size() == 0 || gltf_img.width == -1 || gltf_img.height == -1)
        {
            vkutils::load_image_from_memory(&_engineData, nil.data(), 1, 1, allocated_image,
                                            mipLevels);
        }
        else
        {
            vkutils::load_image_from_memory(&_engineData, gltf_img.image.data(),
                                            gltf_img.width, gltf_img.height, allocated_image,
                                            mipLevels);
        }

        VkImageView imageView;
        VkImageViewCreateInfo imageinfo =
            vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, allocated_image._image,
                                          VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
        vkCreateImageView(_engineData.device, &imageinfo, nullptr, &imageView);

        VkDescriptorImageInfo imageBufferInfo;
        imageBufferInfo.sampler = blockySampler;
        imageBufferInfo.imageView = imageView;
        imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(imageBufferInfo);
    }

    // TEXTURE DESCRIPTOR
    {
        VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            0);
        textureBind.descriptorCount = image_infos.size();
        VkDescriptorSetLayoutCreateInfo setinfo =
            vkinit::descriptorset_layout_create_info(&textureBind, 1);
        vkCreateDescriptorSetLayout(_engineData.device, &setinfo, nullptr,
                                    &_sceneData.textureSetLayout);

        VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(
            _engineData.descriptorPool, &_sceneData.textureSetLayout, 1);

        vkAllocateDescriptorSets(_engineData.device, &allocInfo,
                                 &_sceneData.textureDescriptor);

        VkWriteDescriptorSet textures = vkinit::write_descriptor_image(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _sceneData.textureDescriptor,
            image_infos.data(), 0, image_infos.size());

        vkUpdateDescriptorSets(_engineData.device, 1, &textures, 0, nullptr);
    }

    _vulkanRaytracing.convert_scene_to_vk_geometry(gltf_scene, _sceneData.vertexBuffer,
                                                   _sceneData.indexBuffer);
    _vulkanRaytracing.build_blas(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    _vulkanRaytracing.build_tlas(gltf_scene,
                                 VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                     VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                                 false);

    GPUSceneDesc desc = {};
    VkBufferDeviceAddressInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

    info.buffer = _sceneData.vertexBuffer._buffer;
    desc.vertexAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

    info.buffer = _sceneData.normalBuffer._buffer;
    desc.normalAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

    info.buffer = _sceneData.texBuffer._buffer;
    desc.uvAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

    info.buffer = _sceneData.indexBuffer._buffer;
    desc.indexAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

    info.buffer = _sceneData.lightmapTexBuffer._buffer;
    desc.lightmapUvAddress = vkGetBufferDeviceAddress(_engineData.device, &info);

    GPUMeshInfo* dataMesh = new GPUMeshInfo[gltf_scene.prim_meshes.size()];
    for (int i = 0; i < gltf_scene.prim_meshes.size(); i++)
    {
        dataMesh[i].indexOffset = gltf_scene.prim_meshes[i].first_idx;
        dataMesh[i].vertexOffset = gltf_scene.prim_meshes[i].vtx_offset;
        dataMesh[i].materialIndex = gltf_scene.prim_meshes[i].material_idx;
    }

    _sceneData.sceneDescBuffer = vkutils::create_upload_buffer(
        &_engineData, &desc, sizeof(GPUSceneDesc), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    _sceneData.meshInfoBuffer = vkutils::create_upload_buffer(
        &_engineData, dataMesh, sizeof(GPUMeshInfo) * gltf_scene.prim_meshes.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    delete[] dataMesh;
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd)
{
    for (int i = 0; i < gltf_scene.nodes.size(); i++)
    {
        if (true || gltf_scene
                            .materials[gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh]
                                           .material_idx]
                            .alpha_mode == 0)
        {
            auto& mesh = gltf_scene.prim_meshes[gltf_scene.nodes[i].prim_mesh];
            vkCmdDrawIndexed(cmd, mesh.idx_count, 1, mesh.first_idx, mesh.vtx_offset, i);
        }
    }
}

void VulkanEngine::init_query_pool()
{
    VkQueryPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;

    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount = 512; // REVIEW

    VK_CHECK(
        vkCreateQueryPool(_engineData.device, &createInfo, nullptr, &_engineData.queryPool));
}