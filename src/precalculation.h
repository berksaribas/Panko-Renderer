#pragma once

#define RAYTRACING
#include <vector>
#include <glm/glm.hpp>
#include <gltf_scene.hpp>
#include <vk_engine.h>
#include <precalculation_types.h>

class Precalculation {
public:
	void prepare(VulkanEngine& engine, GltfScene& scene, PrecalculationInfo precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult);
	void load(const char* filename, PrecalculationInfo& precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult);
private:
	uint8_t* voxelize(GltfScene& scene, float voxelSize, int padding, int& dimX, int& dimY, int& dimZ);
	void place_probes(VulkanEngine& engine, std::vector<glm::vec4>& probes, int targetProbeCount, Receiver* receivers, int receiverCount, int nOverlaps);
	Receiver* generate_receivers(GltfScene& scene);
	void probe_raycast(VulkanEngine& engine, std::vector<glm::vec4>& probes, int rays, int sphericalHarmonicsOrder, GPUProbeRaycastResult* probeRaycastResult, float* probeRaycastBasisFunctions);
	void receiver_raycast(VulkanEngine& engine, std::vector<AABB>& aabbClusters, std::vector<glm::vec4>& probes, int rays, float radius, int sphericalHarmonicsOrder, int clusterCoefficientCount, int maxReceivers, int totalReceiverCount, int maxProbesPerCluster, float* clusterProjectionMatrices, float* receiverCoefficientMatrices, float* receiverProbeWeightData, int* clusterProbes);
};
