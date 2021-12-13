#pragma once 

#include <vector>
#include <glm/glm.hpp>
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

	//Debug data
	float* receiverProbeWeightData;
};