#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <gltf_scene.hpp>
#include <vk_engine.h>
#include "../shaders/common.glsl"

#define SPHERICAL_HARMONICS_NUM_COEFF(ORDER) ((ORDER + 1) * (ORDER + 1))

struct Receiver {
	glm::vec3 position;
	glm::vec3 normal;
	glm::ivec2 uv;
	bool exists;
};

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
	std::vector<Receiver> receivers;
};

struct GPUProbeDensityUniformData {
	int probeCount;
	float radius;
};

struct PrecalculationInfo {
	//Voxelization
	float voxelSize;
	int voxelPadding;
	//Probe placement
	int probeOverlaps;
	//Precalculation
	int raysPerProbe;
	int raysPerReceiver;
	int sphericalHarmonicsOrder;
	int clusterCoefficientCount;
	int maxReceiversInCluster;
};

struct PrecalculationLoadData {
	int probesCount;
	int totalClusterReceiverCount;
	int aabbClusterCount;
};

struct PrecalculationResult {
	//Probes
	std::vector<glm::vec4> probes;
	GPUProbeRaycastResult* probeRaycastResult;
	float* probeRaycastBasisFunctions;

	//Receivers
	Receiver* aabbReceivers;
	float* clusterProjectionMatrices;
	float* receiverCoefficientMatrices;
	ClusterReceiverInfo* clusterReceiverInfos;
	glm::ivec4* clusterReceiverUvs;
};

class Precalculation {
public:
	void prepare(VulkanEngine& engine, GltfScene& scene, PrecalculationInfo precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult);
	void load(const char* filename, PrecalculationInfo& precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult);
private:
	uint8_t* voxelize(GltfScene& scene, float voxelSize, int padding, int& dimX, int& dimY, int& dimZ);
	void place_probes(VulkanEngine& engine, std::vector<glm::vec4>& probes, int targetProbeCount, int radius);
	Receiver* generate_receivers(GltfScene& scene);
	void probe_raycast(VulkanEngine& engine, std::vector<glm::vec4>& probes, int rays, int sphericalHarmonicsOrder, GPUProbeRaycastResult* probeRaycastResult, float* probeRaycastBasisFunctions);
	void receiver_raycast(VulkanEngine& engine, std::vector<AABB>& aabbClusters, std::vector<glm::vec4>& probes, int rays, int radius, int sphericalHarmonicsOrder, int clusterCoefficientCount, float* clusterProjectionMatrices, float* receiverCoefficientMatrices);
};
