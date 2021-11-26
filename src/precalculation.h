#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <gltf_scene.hpp>
#include <vk_engine.h>
#include "../shaders/common.glsl"

#define SPHERICAL_HARMONICS_ORDER 2
#define SPHERICAL_HARMONICS_NUM_COEFF ((SPHERICAL_HARMONICS_ORDER + 1) * (SPHERICAL_HARMONICS_ORDER + 1))
#define CLUSTER_COEFFICIENT_COUNT 32

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

class Precalculation {
public:
	uint8_t* voxelize(GltfScene& scene, float voxelSize, int padding, bool save = false);
	std::vector<glm::vec4> place_probes(VulkanEngine& engine, int overlaps);
	Receiver* generate_receivers();
	void probe_raycast(VulkanEngine& engine, int rays);
	void receiver_raycast(VulkanEngine& engine, int rays);
	std::vector<glm::vec4> _probes;
	GPUProbeRaycastResult* _probeRaycastResult;
	float* _probeRaycastBasisFunctions;
	int _raysPerProbe;
	Receiver* _receivers;
	std::vector<AABB> _aabbClusters;

	int _totalClusterReceiverCount;
	float* _clusterProjectionMatrices;
	float* _receiverCoefficientMatrices;
	ClusterReceiverInfo* _clusterReceiverInfos;
	glm::ivec4* _clusterReceiverUvs;
private:
	GltfScene* _scene;
	uint8_t* _voxelData;
	float _voxelSize;
	int _dimX, _dimY, _dimZ;
	int _padding;
};
