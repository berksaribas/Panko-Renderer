#pragma once

#include <functional>
#include <precalculation_types.h>
#include <vk_types.h>

class Shadow;
class BRDF;
class VulkanDebugRenderer;
class GltfScene;

class DiffuseIllumination
{
public:
    void init(EngineData& engineData, PrecalculationInfo* precalculationInfo,
              PrecalculationLoadData* precalculationLoadData,
              PrecalculationResult* precalculationResult, GltfScene& scene);
    void render(VkCommandBuffer cmd, EngineData& engineData, SceneData& sceneData,
                Shadow& shadow, BRDF& brdfUtils,
                std::function<void(VkCommandBuffer cmd)>&& function, bool realtimeProbeRaycast,
                int numBasisFunctions);
    void render_ground_truth(VkCommandBuffer cmd, EngineData& engineData, SceneData& sceneData,
                             Shadow& shadow, BRDF& brdfUtils);

    void debug_draw_probes(VulkanDebugRenderer& debugRenderer, bool showProbeRays,
                           float sceneScale);
    void debug_draw_receivers(VulkanDebugRenderer& debugRenderer, float sceneScale);
    void debug_draw_specific_receiver(VulkanDebugRenderer& debugRenderer, int specificCluster,
                                      int specificReceiver, int specificReceiverRaySampleCount,
                                      bool* enabledProbes, bool showSpecificProbeRays,
                                      float sceneScale);

    AllocatedBuffer _configBuffer;
    Handle<Vrg::Bindable> _configBufferBinding;
    AllocatedBuffer _probeRaycastResultOfflineBuffer;
    Handle<Vrg::Bindable> _probeRaycastResultOfflineBufferBinding;
    AllocatedBuffer _probeRaycastResultOnlineBuffer;
    Handle<Vrg::Bindable> _probeRaycastResultOnlineBufferBinding;
    AllocatedBuffer _probeBasisBuffer;
    Handle<Vrg::Bindable> _probeBasisBufferBinding;
    AllocatedBuffer _probeRelightOutputBuffer;
    Handle<Vrg::Bindable> _probeRelightOutputBufferBinding;

    AllocatedBuffer _clusterProjectionMatricesBuffer;
    Handle<Vrg::Bindable> _clusterProjectionMatricesBufferBinding;
    AllocatedBuffer _clusterProjectionOutputBuffer;
    Handle<Vrg::Bindable> _clusterProjectionOutputBufferBinding;
    AllocatedBuffer _clusterReceiverInfos;
    Handle<Vrg::Bindable> _clusterReceiverInfosBinding;
    AllocatedBuffer _clusterProbes;
    Handle<Vrg::Bindable> _clusterProbesBinding;

    AllocatedBuffer _receiverReconstructionMatricesBuffer;
    Handle<Vrg::Bindable> _receiverReconstructionMatricesBufferBinding;
    AllocatedBuffer _clusterReceiverUvs;
    Handle<Vrg::Bindable> _clusterReceiverUvsBinding;

    AllocatedBuffer _probeLocationsBuffer;
    Handle<Vrg::Bindable> _probeLocationsBufferBinding;

    AllocatedBuffer _receiverBuffer;
    Handle<Vrg::Bindable> _receiverBufferBinding;

    AllocatedImage _giIndirectLightImage;
    Handle<Vrg::Bindable> _giIndirectLightImageBinding;
    AllocatedImage _lightmapColorImage;
    Handle<Vrg::Bindable> _lightmapColorImageBinding;
    AllocatedImage _dilatedGiIndirectLightImage;
    Handle<Vrg::Bindable> _dilatedGiIndirectLightImageBinding;

    VkExtent2D _lightmapExtent{2048, 2048};

private:
    PrecalculationInfo* _precalculationInfo;
    PrecalculationLoadData* _precalculationLoadData;
    PrecalculationResult* _precalculationResult;

    GIConfig _config = {};
    VkExtent2D _giLightmapExtent{0, 0};

    uint32_t _gpuReceiverCount;

    std::vector<GPUReceiverDataUV> receiverDataVector;
};