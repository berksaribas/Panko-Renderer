#pragma once 

#include <vector>
#include <glm/glm.hpp>
#include "../shaders/common.glsl"

#define SPHERICAL_HARMONICS_NUM_COEFF(ORDER) ((ORDER + 1) * (ORDER + 1))

struct Receiver {
	glm::vec3 position;
	glm::vec3 normal;
	glm::ivec2 uv;
	int objectId;
	float dPos;
	bool exists;
	bool processed;
	std::vector<glm::vec3> poses;
	std::vector<glm::vec3> norms;
};

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
	std::vector<Receiver> receivers;
	std::vector<int> probes;
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
	int lightmapResolution;
	int texelSize;
	float desiredSpacing;
};

struct PrecalculationLoadData {
	int probesCount;
	int totalClusterReceiverCount;
	int aabbClusterCount;
	int maxProbesPerCluster;
	int totalProbesPerCluster;
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

	int* clusterProbes;

	//Debug data
	float* receiverProbeWeightData;
};