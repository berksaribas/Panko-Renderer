#include "precalculation.h"
#include <triangle_box_intersection.h>
#include <omp.h>
#include "spherical_harmonics.h"

#include "triangle_box_intersection.h"
#include <fstream>
#include <optick.h>
#include <queue>
#include <vk_initializers.h>
#include <vk_utils.h>

#include "json.hpp"
#include <sys/stat.h>

#define USE_COMPUTE_PROBE_DENSITY_CALCULATION 1
#define M_PI    3.14159265358979323846264338327950288


//Christer Ericson's Real-Time Collision Detection 
static glm::vec3 calculate_barycentric(glm::vec2 p, glm::vec2 a, glm::vec2 b, glm::vec2 c) {
	//glm::vec2 v0 = b - a, v1 = c - a, v2 = p - a;
	//float d00 = glm::dot(v0, v0);
	//float d01 = glm::dot(v0, v1);
	//float d11 = glm::dot(v1, v1);
	//float d20 = glm::dot(v2, v0);
	//float d21 = glm::dot(v2, v1);
	//float denom = d00 * d11 - d01 * d01;
	//
	//float v = (d11 * d20 - d01 * d21) / denom;
	//float w = (d00 * d21 - d01 * d20) / denom;
	//float u = 1.0f - v - w;
	//return { u, v, w };

	glm::vec2 v0 = b - a, v1 = c - a, v2 = p - a;
	float den = v0.x * v1.y - v1.x * v0.y;
	float v = (v2.x * v1.y - v1.x * v2.y) / den;
	float w = (v0.x * v2.y - v2.x * v0.y) / den;
	float u = 1.0f - v - w;
	return { u, v, w };
}

static glm::vec3 apply_barycentric(glm::vec3 barycentricCoordinates, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
	return {
		barycentricCoordinates.x * a.x + barycentricCoordinates.y * b.x + barycentricCoordinates.z * c.x,
		barycentricCoordinates.x * a.y + barycentricCoordinates.y * b.y + barycentricCoordinates.z * c.y,
		barycentricCoordinates.x * a.z + barycentricCoordinates.y * b.z + barycentricCoordinates.z * c.z
	};
}

static float calculate_density(float length, float radius) {
	OPTICK_EVENT();

	float t = length / radius;
	if (t >= 0 && t <= 1) {
		float tSquared = t * t;
		return (2 * t * tSquared) - (3 * tSquared) + 1;
	}
	return 0;
}

static float calculate_spatial_weight(float radius, int probeIndex, std::vector<glm::vec4>& probes) {
	OPTICK_EVENT();

	float total_weight = 0.f;
#pragma omp parallel for reduction(+:total_weight) 
	for (int i = 0; i < probes.size(); i++) {
		if (i != probeIndex) {
			total_weight += calculate_density(glm::distance(probes[i], probes[probeIndex]), radius);
		}
	}
	return total_weight;
}

static float calculate_radius(Receiver* receivers, int receiverSize, std::vector<glm::vec4>& probes, int overlaps) {
	OPTICK_EVENT();

	float radius = 0;
	int receiverCount = 0;
	for (int r = 0; r < receiverSize; r += 1) {
		std::priority_queue <float, std::vector<float>, std::greater<float>> minheap;

		if (!receivers[r].exists) {
			continue;
		}

		for (int p = 0; p < probes.size(); p++) {
			minheap.push(glm::distance(glm::vec3(probes[p]), receivers[r].position));
		}

		for (int i = 0; i < overlaps - 1; i++) {
			minheap.pop();
		}

		radius += minheap.top();
		receiverCount++;
	}

	return radius / receiverCount;
}

static void divide_aabb(std::vector<AABB>& aabbClusters, AABB node, Receiver* receivers, int receiverCount, int divideReceivers) {
	int receiversInIt = 0;
	for (int i = 0; i < receiverCount; i++) {
		auto receiver = receivers[i];
		if (!receiver.exists) {
			continue;
		}
		if (receiver.position.x >= node.min.x && receiver.position.x < node.max.x &&
			receiver.position.y >= node.min.y && receiver.position.y < node.max.y &&
			receiver.position.z >= node.min.z && receiver.position.z < node.max.z)
		{
			node.receivers.push_back(receiver);
		}
	}

	if (node.receivers.size() > divideReceivers) {
		//printf("There are %d receivers in this AABB, splitting\n", node.receivers.size());

		auto size = node.max - node.min;
		float dimX = size.x;
		float dimY = size.y;
		float dimZ = size.z;

		//printf("Sizes are: %f %f %f\n", dimX, dimY, dimZ);

		AABB first = {};
		AABB second = {};

		if (dimX >= dimY && dimX >= dimZ) {
			first.min = node.min;
			first.max = { node.min.x + dimX / 2.f, node.max.y, node.max.z };
			second.min = { node.min.x + dimX / 2.f, node.min.y, node.min.z };
			second.max = node.max;
		}
		else if (dimY >= dimX && dimY >= dimZ) {
			first.min = node.min;
			first.max = { node.max.x, node.min.y + dimY / 2.f, node.max.z };
			second.min = { node.min.x, node.min.y + dimY / 2.f, node.min.z };
			second.max = node.max;
		}
		else {
			first.min = node.min;
			first.max = { node.max.x, node.max.y, node.min.z + dimZ / 2.f };
			second.min = { node.min.x, node.min.y, node.min.z + dimZ / 2.f };
			second.max = node.max;
		}

		divide_aabb(aabbClusters, first, node.receivers.data(), node.receivers.size(), divideReceivers);
		divide_aabb(aabbClusters, second, node.receivers.data(), node.receivers.size(), divideReceivers);
	}
	else if (node.receivers.size() > 0) {
		aabbClusters.push_back(node);
	}
}

static void save_binary(std::string filename, void* data, size_t size) {
	FILE* ptr;
	fopen_s(&ptr, filename.c_str(), "wb");
	fwrite(data, size, 1, ptr);
	fclose(ptr);
}

size_t seek_file(std::string filename) {
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

void load_binary(std::string filename, void* destination, size_t size) {
	FILE* ptr;
	fopen_s(&ptr, filename.c_str(), "rb");
	fread_s(destination, size, size, 1, ptr);
	fclose(ptr);
}

void Precalculation::prepare(VulkanEngine& engine, GltfScene& scene, PrecalculationInfo precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult) {
	int voxelDimX, voxelDimY, voxelDimZ;
	uint8_t* voxelData = voxelize(scene, precalculationInfo.voxelSize, precalculationInfo.voxelPadding, voxelDimX, voxelDimY, voxelDimZ);
	Receiver* receivers = generate_receivers(scene);
	std::vector<glm::vec4> probes;

	//Place probes everywhere in the voxelized scene
	for (int k = precalculationInfo.voxelPadding; k < voxelDimZ - precalculationInfo.voxelPadding; k++) {
		for (int j = precalculationInfo.voxelPadding; j < voxelDimY - precalculationInfo.voxelPadding; j++) {
			for (int i = precalculationInfo.voxelPadding; i < voxelDimX - precalculationInfo.voxelPadding; i++) {
				int index = i + j * voxelDimX + k * voxelDimX * voxelDimY;
				if (voxelData[index] == 3) {
					probes.push_back({
						scene.m_dimensions.min.x + precalculationInfo.voxelSize * (i - precalculationInfo.voxelPadding) + precalculationInfo.voxelSize / 2.f,
						scene.m_dimensions.min.y + precalculationInfo.voxelSize * (j - precalculationInfo.voxelPadding) + precalculationInfo.voxelSize / 2.f,
						scene.m_dimensions.min.z + precalculationInfo.voxelSize * (k - precalculationInfo.voxelPadding) + precalculationInfo.voxelSize / 2.f,
						0.f
					});
				}
			}
		}
	}

	int targetProbeCount = voxelDimX - precalculationInfo.voxelPadding * 2;
	printf("Targeted amount of probes is %d\n", targetProbeCount);

	float radius = calculate_radius(receivers, scene.lightmap_width * scene.lightmap_height, probes, precalculationInfo.probeOverlaps);
	printf("Radius is %f\n", radius);

	place_probes(engine, probes, targetProbeCount, radius);

	outPrecalculationLoadData.probesCount = probes.size();
	outPrecalculationResult.probes = probes;
	outPrecalculationResult.probeRaycastResult = (GPUProbeRaycastResult*)malloc(precalculationInfo.raysPerProbe * probes.size() * sizeof(GPUProbeRaycastResult));
	outPrecalculationResult.probeRaycastBasisFunctions = (float*)malloc(precalculationInfo.raysPerProbe * probes.size() * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * sizeof(float));
	probe_raycast(engine, probes, precalculationInfo.raysPerProbe, precalculationInfo.sphericalHarmonicsOrder, outPrecalculationResult.probeRaycastResult, outPrecalculationResult.probeRaycastBasisFunctions);

	//AABB clustering
	std::vector<AABB> aabbClusters;
	AABB initial_node = {};
	initial_node.min = scene.m_dimensions.min - glm::vec3(1.0); //adding some padding
	initial_node.max = scene.m_dimensions.max + glm::vec3(1.0); //adding some padding
	divide_aabb(aabbClusters, initial_node, receivers, scene.lightmap_width * scene.lightmap_height, precalculationInfo.maxReceiversInCluster);
	
	outPrecalculationLoadData.aabbClusterCount = aabbClusters.size();

	//Allocate memory for the results
	outPrecalculationResult.clusterProjectionMatrices = new float[aabbClusters.size() * probes.size() * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * precalculationInfo.clusterCoefficientCount];
	outPrecalculationResult.clusterReceiverInfos = new ClusterReceiverInfo[aabbClusters.size()];
	int clusterReceiverCount = 0;
	for (int i = 0; i < aabbClusters.size(); i++) {
		outPrecalculationResult.clusterReceiverInfos[i].receiverCount = aabbClusters[i].receivers.size();
		outPrecalculationResult.clusterReceiverInfos[i].receiverOffset = clusterReceiverCount;
		clusterReceiverCount += aabbClusters[i].receivers.size();
	}

	printf("Total receivers avaialble for aabb: %d\n", clusterReceiverCount);

	outPrecalculationLoadData.totalClusterReceiverCount = clusterReceiverCount;
	outPrecalculationResult.aabbReceivers = new Receiver[clusterReceiverCount];
	outPrecalculationResult.receiverCoefficientMatrices = new float[clusterReceiverCount * precalculationInfo.clusterCoefficientCount];
	outPrecalculationResult.clusterReceiverUvs = new glm::ivec4[clusterReceiverCount];
	int currOffset = 0;
	for (int i = 0; i < aabbClusters.size(); i++) {
		for (int j = 0; j < aabbClusters[i].receivers.size(); j++) {
			outPrecalculationResult.clusterReceiverUvs[currOffset + j] = glm::ivec4(aabbClusters[i].receivers[j].uv.x, aabbClusters[i].receivers[j].uv.y, 0, 0);
			outPrecalculationResult.aabbReceivers[currOffset + j] = aabbClusters[i].receivers[j];
		}
		currOffset +=aabbClusters[i].receivers.size();
	}

	float newRadius = calculate_radius(receivers, scene.lightmap_width * scene.lightmap_height, probes, precalculationInfo.probeOverlaps);
	receiver_raycast(engine, aabbClusters, probes, precalculationInfo.raysPerReceiver, newRadius, precalculationInfo.sphericalHarmonicsOrder, precalculationInfo.clusterCoefficientCount, outPrecalculationResult.clusterProjectionMatrices, outPrecalculationResult.receiverCoefficientMatrices);

	// Save everything
	std::string filename = "../../precomputation/precalculation";

	nlohmann::json config;
	config["voxelSize"] = precalculationInfo.voxelSize;
	config["voxelPadding"] = precalculationInfo.voxelPadding;
	config["probeOverlaps"] = precalculationInfo.probeOverlaps;
	config["raysPerProbe"] = precalculationInfo.raysPerProbe;
	config["raysPerReceiver"] = precalculationInfo.raysPerReceiver;
	config["sphericalHarmonicsOrder"] = precalculationInfo.sphericalHarmonicsOrder;
	config["clusterCoefficientCount"] = precalculationInfo.clusterCoefficientCount;
	config["maxReceiversInCluster"] = precalculationInfo.maxReceiversInCluster;

	config["probesCount"] = outPrecalculationLoadData.probesCount;
	config["totalClusterReceiverCount"] = outPrecalculationLoadData.totalClusterReceiverCount;
	config["aabbClusterCount"] = outPrecalculationLoadData.aabbClusterCount;

	config["fileProbes"] = filename + ".Probes";
	save_binary(filename + ".Probes", outPrecalculationResult.probes.data(), sizeof(glm::vec4) * outPrecalculationResult.probes.size());
	
	config["fileProbeRaycastResult"] = filename + ".ProbeRaycastResult";
	save_binary(filename + ".ProbeRaycastResult", outPrecalculationResult.probeRaycastResult, precalculationInfo.raysPerProbe * probes.size() * sizeof(GPUProbeRaycastResult));
	
	config["fileProbeRaycastBasisFunctions"] = filename + ".ProbeRaycastBasisFunctions";
	save_binary(filename + ".ProbeRaycastBasisFunctions", outPrecalculationResult.probeRaycastBasisFunctions, precalculationInfo.raysPerProbe * probes.size() * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * sizeof(float));

	config["fileAabbReceivers"] = filename + ".AabbReceivers";
	save_binary(filename + ".AabbReceivers", outPrecalculationResult.aabbReceivers, sizeof(Receiver) * clusterReceiverCount);

	config["fileClusterProjectionMatrices"] = filename + ".ClusterProjectionMatrices";
	save_binary(filename + ".ClusterProjectionMatrices", outPrecalculationResult.clusterProjectionMatrices, aabbClusters.size() * probes.size() * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * precalculationInfo.clusterCoefficientCount * sizeof(float));

	config["fileReceiverCoefficientMatrices"] = filename + ".ReceiverCoefficientMatrices";
	save_binary(filename + ".ReceiverCoefficientMatrices", outPrecalculationResult.receiverCoefficientMatrices, clusterReceiverCount * precalculationInfo.clusterCoefficientCount * sizeof(float));

	config["fileClusterReceiverInfos"] = filename + ".ClusterReceiverInfos";
	save_binary(filename + ".ClusterReceiverInfos", outPrecalculationResult.clusterReceiverInfos, aabbClusters.size() * sizeof(ClusterReceiverInfo));

	config["fileClusterReceiverUvs"] = filename + ".ClusterReceiverUvs";
	save_binary(filename + ".ClusterReceiverUvs", outPrecalculationResult.clusterReceiverUvs, clusterReceiverCount * sizeof(glm::ivec4));

	std::ofstream file(filename + ".cfg");
	file << config.dump();
	file.close();

	// Deal with this later
	/*
	if (true) {
		std::ofstream file("mesh_voxelized_res_with_probes.obj");
		int counter = 0;

		for (int k = 0; k < voxelDimZ; k++) {
			for (int j = 0; j < voxelDimY; j++) {
				for (int i = 0; i < voxelDimX; i++) {
					int index = i + j * voxelDimX + k * dimX * voxelDimY;

					if (voxelData[index] != 1) {
						continue;
					}

					float voxelX = scene.m_dimensions.min.x + voxelSize * (i - padding);
					float voxelY = scene.m_dimensions.min.y + voxelSize * (j - padding);
					float voxelZ = scene.m_dimensions.min.z + voxelSize * (k - padding);

					//Vertices
					file << "v " << voxelX << " " << voxelY << " " << voxelZ << "\n";
					file << "v " << voxelX << " " << voxelY << " " << voxelZ + voxelSize << "\n";
					file << "v " << voxelX + voxelSize << " " << voxelY << " " << voxelZ + voxelSize << "\n";
					file << "v " << voxelX + +voxelSize << " " << voxelY << " " << voxelZ << "\n";
					file << "v " << voxelX << " " << voxelY + voxelSize << " " << voxelZ << "\n";
					file << "v " << voxelX << " " << voxelY + voxelSize << " " << voxelZ + voxelSize << "\n";
					file << "v " << voxelX + voxelSize << " " << voxelY + voxelSize << " " << voxelZ + voxelSize << "\n";
					file << "v " << voxelX + +voxelSize << " " << voxelY + voxelSize << " " << voxelZ << "\n";

					int idx = counter * 8 + 1;

					//Face 1
					file << "f " << idx << " " << idx + 1 << " " << idx + 2 << "\n";
					file << "f " << idx << " " << idx + 2 << " " << idx + 3 << "\n";
					//Face 2
					file << "f " << idx + 4 << " " << idx + 5 << " " << idx + 6 << "\n";
					file << "f " << idx + 4 << " " << idx + 6 << " " << idx + 7 << "\n";
					//Face 3
					file << "f " << idx << " " << idx + 4 << " " << idx + 5 << "\n";
					file << "f " << idx << " " << idx + 5 << " " << idx + 1 << "\n";
					//Face 4
					file << "f " << idx + 7 << " " << idx + 3 << " " << idx + 2 << "\n";
					file << "f " << idx + 7 << " " << idx + 2 << " " << idx + 6 << "\n";
					//Face 5
					file << "f " << idx << " " << idx + 4 << " " << idx + 7 << "\n";
					file << "f " << idx << " " << idx + 7 << " " << idx + 3 << "\n";
					//Face 6
					file << "f " << idx + 1 << " " << idx + 5 << " " << idx + 6 << "\n";
					file << "f " << idx + 1 << " " << idx + 6 << " " << idx + 2 << "\n";

					counter++;
				}
			}
		}

		for (int i = 0; i < _probes.size(); i++) {
			float voxelX = _probes[i].x;
			float voxelY = _probes[i].y;
			float voxelZ = _probes[i].z;

			//Vertices
			file << "v " << voxelX << " " << voxelY << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX << " " << voxelY << " " << voxelZ + 0.2 << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX + 0.2 << " " << voxelY << " " << voxelZ + 0.2 << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX + +0.2 << " " << voxelY << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX << " " << voxelY + 0.2 << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX << " " << voxelY + 0.2 << " " << voxelZ + 0.2 << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX + 0.2 << " " << voxelY + 0.2 << " " << voxelZ + 0.2 << " " << 1 << " " << 0 << " " << 0 << "\n";
			file << "v " << voxelX + +0.2 << " " << voxelY + 0.2 << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";

			int idx = counter * 8 + 1;

			//Face 1
			file << "f " << idx << " " << idx + 1 << " " << idx + 2 << "\n";
			file << "f " << idx << " " << idx + 2 << " " << idx + 3 << "\n";
			//Face 2
			file << "f " << idx + 4 << " " << idx + 5 << " " << idx + 6 << "\n";
			file << "f " << idx + 4 << " " << idx + 6 << " " << idx + 7 << "\n";
			//Face 3
			file << "f " << idx << " " << idx + 4 << " " << idx + 5 << "\n";
			file << "f " << idx << " " << idx + 5 << " " << idx + 1 << "\n";
			//Face 4
			file << "f " << idx + 7 << " " << idx + 3 << " " << idx + 2 << "\n";
			file << "f " << idx + 7 << " " << idx + 2 << " " << idx + 6 << "\n";
			//Face 5
			file << "f " << idx << " " << idx + 4 << " " << idx + 7 << "\n";
			file << "f " << idx << " " << idx + 7 << " " << idx + 3 << "\n";
			//Face 6
			file << "f " << idx + 1 << " " << idx + 5 << " " << idx + 6 << "\n";
			file << "f " << idx + 1 << " " << idx + 6 << " " << idx + 2 << "\n";

			counter++;
		}
		file.close();
	}
*/	
}

void Precalculation::load(const char* filename, PrecalculationInfo& precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult)
{
	std::ifstream i(filename);
	nlohmann::json config;
	i >> config;

	precalculationInfo.voxelSize = config["voxelSize"];
	precalculationInfo.voxelPadding = config["voxelPadding"];
	precalculationInfo.probeOverlaps = config["probeOverlaps"];
	precalculationInfo.raysPerProbe = config["raysPerProbe"];
	precalculationInfo.raysPerReceiver = config["raysPerReceiver"];
	precalculationInfo.sphericalHarmonicsOrder = config["sphericalHarmonicsOrder"];
	precalculationInfo.clusterCoefficientCount = config["clusterCoefficientCount"];
	precalculationInfo.maxReceiversInCluster = config["maxReceiversInCluster"];

	outPrecalculationLoadData.probesCount = config["probesCount"];
	outPrecalculationLoadData.totalClusterReceiverCount = config["totalClusterReceiverCount"];
	outPrecalculationLoadData.aabbClusterCount = config["aabbClusterCount"];

	{
		std::string file = config["fileProbes"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		printf("The size is %d\n", size);
		outPrecalculationResult.probes.resize(size / sizeof(glm::vec4));
		load_binary(file, outPrecalculationResult.probes.data(), size);
	}

	{
		std::string file = config["fileProbeRaycastResult"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.probeRaycastResult = (GPUProbeRaycastResult*)malloc(size);
		load_binary(file, outPrecalculationResult.probeRaycastResult, size);
	}

	{
		std::string file = config["fileProbeRaycastBasisFunctions"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.probeRaycastBasisFunctions = (float*)malloc(size);
		load_binary(file, outPrecalculationResult.probeRaycastBasisFunctions, size);
	}

	{
		std::string file = config["fileAabbReceivers"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.aabbReceivers = (Receiver*)malloc(size);
		load_binary(file, outPrecalculationResult.aabbReceivers, size);
	}

	{
		std::string file = config["fileClusterProjectionMatrices"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.clusterProjectionMatrices = (float*)malloc(size);
		load_binary(file, outPrecalculationResult.clusterProjectionMatrices, size);
	}

	{
		std::string file = config["fileReceiverCoefficientMatrices"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.receiverCoefficientMatrices = (float*)malloc(size);
		load_binary(file, outPrecalculationResult.receiverCoefficientMatrices, size);
	}

	{
		std::string file = config["fileClusterReceiverInfos"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.clusterReceiverInfos = (ClusterReceiverInfo*) malloc(size);
		load_binary(file, outPrecalculationResult.clusterReceiverInfos, size);
	}

	{
		std::string file = config["fileClusterReceiverUvs"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.clusterReceiverUvs = (glm::ivec4*)malloc(size);
		load_binary(file, outPrecalculationResult.clusterReceiverUvs, size);
	}
}

uint8_t* Precalculation::voxelize(GltfScene& scene, float voxelSize, int padding, int& dimX, int& dimY, int& dimZ)
{
	OPTICK_EVENT();

	int x = (scene.m_dimensions.max.x - scene.m_dimensions.min.x) / voxelSize + padding * 2;
	int y = (scene.m_dimensions.max.y - scene.m_dimensions.min.y) / voxelSize + padding * 2;
	int z = (scene.m_dimensions.max.z - scene.m_dimensions.min.z) / voxelSize + padding * 2;
	
	glm::vec3 halfSize = { voxelSize / 2.f, voxelSize / 2.f, voxelSize / 2.f };
	dimX = x;
	dimY = y;
	dimZ = z;

	int size = x * y * z;
	int paddinglessSize = (x - padding * 2) * (y - padding * 2) * (z - padding * 2);
	
	//if (_voxelData != nullptr) {
	//	delete[] _voxelData;
	//}

	uint8_t* voxelData = new uint8_t[size];
	memset(voxelData, 0, size);

	printf("Creating a voxel scene with: %d x %d x %d\n", x, y, z);
	int totalVertexMarked = 0;
#pragma omp parallel for
	for (int nodeIndex = 0; nodeIndex < scene.nodes.size(); nodeIndex++) {
		auto& mesh = scene.prim_meshes[scene.nodes[nodeIndex].prim_mesh];

		for (int triangle = 0; triangle < mesh.idx_count; triangle += 3) {
			glm::vec3 vertices[3] = {};
			int minX = x, minY = y, minZ = z;
			int maxX = 0, maxY = 0, maxZ = 0;
			for (int i = 0; i < 3; i++) {
				glm::vec4 vertex = scene.nodes[nodeIndex].world_matrix * glm::vec4(scene.positions[mesh.vtx_offset + scene.indices[mesh.first_idx + triangle + i]], 1.0);
				vertices[i].x = vertex.x / vertex.w;
				vertices[i].y = vertex.y / vertex.w;
				vertices[i].z = vertex.z / vertex.w;
				int voxelX = (int)((vertices[i].x - scene.m_dimensions.min.x) / voxelSize) + padding;
				int voxelY = (int)((vertices[i].y - scene.m_dimensions.min.y) / voxelSize) + padding;
				int voxelZ = (int)((vertices[i].z - scene.m_dimensions.min.z) / voxelSize) + padding;
			
				if (voxelX < minX) {
					minX = voxelX;
				}
				if (voxelX > maxX) {
					maxX = voxelX;
				}
				if (voxelY < minY) {
					minY = voxelY;
				}
				if (voxelY > maxY) {
					maxY = voxelY;
				}
				if (voxelZ < minZ) {
					minZ = voxelZ;
				}
				if (voxelZ > maxZ) {
					maxZ = voxelZ;
				}
			}

			//printf("-------------\n");
			//printf("The voxel MIN is %d %d %d\n", minZ, minY, minX);
			//printf("The voxel MAX is %d %d %d\n", maxZ, maxY, maxX);

			for (int k = minZ; k <= maxZ; k++) {
				for (int j = minY; j <= maxY; j++) {
					for (int i = minX; i <= maxX; i++) {
						if (!tri_box_overlap({ scene.m_dimensions.min.x + (i - padding) * voxelSize + voxelSize / 2.f,
											scene.m_dimensions.min.y + (j - padding) * voxelSize + voxelSize / 2.f,
											scene.m_dimensions.min.z + (k - padding) * voxelSize + voxelSize / 2.f },
											halfSize, vertices[0], vertices[1], vertices[2])) {
							continue;
						}

						int index = i + j * x + k * x * y;
						if (index < size) {
							voxelData[index] = 1;
							totalVertexMarked++;
						}
					}
				}
			}
		}
	}

	printf("A total of voxels marked: %d\n", totalVertexMarked);
	
	//FLOOD FILL
	{
		std::queue<glm::i32vec3> fillQueue;
		fillQueue.push({ 0, 0, 0 });
		int index = 0;
		voxelData[index] = 2;
		while (fillQueue.size() > 0)
		{
			glm::i32vec3 curr = fillQueue.front();
			fillQueue.pop();

			glm::i32vec3 neighbors[6] =
			{
				{curr.x - 1, curr.y, curr.z},
				{curr.x + 1, curr.y, curr.z},
				{curr.x, curr.y - 1, curr.z},
				{curr.x, curr.y + 1, curr.z},
				{curr.x, curr.y, curr.z - 1},
				{curr.x, curr.y, curr.z + 1},
			};

			for (int i = 0; i < 6; i++) {
				glm::i32vec3 neighbor = neighbors[i];
				if (neighbor.x >= 0 && neighbor.x < x && neighbor.y >= 0 && neighbor.y < y && neighbor.z >= 0 && neighbor.z < z) {
					int ind = neighbor.x + neighbor.y * x + neighbor.z * x * y;
					if (voxelData[ind] == 0) {
						fillQueue.push(neighbor);
						voxelData[ind] = 2;
					}
					if (voxelData[ind] == 1) {
						int currInd = curr.x + curr.y * x + curr.z * x * y;
						voxelData[currInd] = 3;
					}
				}
			}
		}

		for (int k = 1; k < z - 1; k++) {
			for (int j = 1; j < y - 1; j++) {
				for (int i = 1; i < x - 1; i++) {
					int index = i + j * x + k * x * y;
					if (voxelData[index] == 0) {
						voxelData[index] = 1;
					}
				}
			}
		}
	}
	
	return voxelData;
}

void Precalculation::place_probes(VulkanEngine& engine, std::vector<glm::vec4>& probes, int targetProbeCount, int radius)
{
	OPTICK_EVENT()

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
	ComputeInstance instance = {};
	engine.vulkanCompute.create_buffer(instance, UNIFORM, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(GPUProbeDensityUniformData));
	engine.vulkanCompute.create_buffer(instance, STORAGE, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::vec4) * probes.size());
	engine.vulkanCompute.create_buffer(instance, STORAGE, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(float) * probes.size());
	engine.vulkanCompute.build(instance, engine._descriptorPool, "../../shaders/precalculate_probe_density_weights.comp.spv");
	vkutils::cpu_to_gpu(engine._allocator, instance.bindings[1].buffer, probes.data(), sizeof(glm::vec4) * probes.size());
#endif

	float currMaxWeight = -1;
	int toRemoveIndex = -1;
	while (probes.size() > targetProbeCount) {
#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
		GPUProbeDensityUniformData ub = { probes.size(), radius };
		vkutils::cpu_to_gpu(engine._allocator, instance.bindings[0].buffer, &ub, sizeof(GPUProbeDensityUniformData));
		int groupcount = ((probes.size()) / 256) + 1;
		engine.vulkanCompute.compute(instance, groupcount, 1, 1);
#endif
		for (int i = 0; i < probes.size(); i++) {
#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
			void* gpuWeightData;
			vmaMapMemory(engine._allocator, instance.bindings[2].buffer._allocation, &gpuWeightData);
			float weight = ((float*)gpuWeightData)[i];
			vmaUnmapMemory(engine._allocator, instance.bindings[2].buffer._allocation);
#else
			float weight = calculate_spatial_weight(radius, i, probes);
#endif
			if (weight >= currMaxWeight) {
				toRemoveIndex = i;
				currMaxWeight = weight;
			}
		}

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
		void* gpuProbeData;
		glm::vec4* castedGpuProbeData;
		vmaMapMemory(engine._allocator, instance.bindings[1].buffer._allocation, &gpuProbeData);
		castedGpuProbeData = (glm::vec4*)gpuProbeData;
		castedGpuProbeData[toRemoveIndex] = castedGpuProbeData[probes.size() - 1];
		vmaUnmapMemory(engine._allocator, instance.bindings[1].buffer._allocation);
#endif

		if (toRemoveIndex != -1) {
			probes[toRemoveIndex] = probes[probes.size() - 1];
			probes.pop_back();
			currMaxWeight = -1;
			toRemoveIndex = -1;
		}
		printf("Size of probes %d\n", probes.size());
	}

	printf("Found this many probes: %d\n", probes.size());

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
	engine.vulkanCompute.destroy_compute_instance(instance);
#endif
}

Receiver* Precalculation::generate_receivers(GltfScene& scene)
{
	OPTICK_EVENT();
	printf("Generating receivers!\n");

	int receiverCount = scene.lightmap_width * scene.lightmap_height;
	Receiver* _receivers = new Receiver[receiverCount];
	memset(_receivers, 0, sizeof(Receiver) * receiverCount);
	int receiverCounter = 0;
	
	for (int nodeIndex = 0; nodeIndex < scene.nodes.size(); nodeIndex++) {
		auto& mesh = scene.prim_meshes[scene.nodes[nodeIndex].prim_mesh];
			for (int triangle = 0; triangle < mesh.idx_count; triangle += 3) {
				glm::vec3 worldVertices[3];
				glm::vec3 worldNormals[3];
				glm::vec2 texVertices[3];
				int minX = scene.lightmap_width, minY = scene.lightmap_height;
				int maxX = 0, maxY = 0;
				for (int i = 0; i < 3; i++) {
					int vertexIndex = mesh.vtx_offset + scene.indices[mesh.first_idx + triangle + i];

					glm::vec4 vertex = scene.nodes[nodeIndex].world_matrix * glm::vec4(scene.positions[vertexIndex], 1.0);
					worldVertices[i] = glm::vec3(vertex / vertex.w);
						

					worldNormals[i] = glm::mat3(glm::transpose(glm::inverse(scene.nodes[nodeIndex].world_matrix))) * glm::vec4(scene.normals[vertexIndex], 1.0);

					texVertices[i] = scene.lightmapUVs[vertexIndex];

					if (texVertices[i].x < minX) {
						minX = texVertices[i].x;
					}
					if (texVertices[i].x > maxX) {
						maxX = texVertices[i].x;
					}
					if (texVertices[i].y < minY) {
						minY = texVertices[i].y;
					}
					if (texVertices[i].y > maxY) {
						maxY = texVertices[i].y;
					}
				}

				for (int j = minY; j <= maxY; j++) {
					for (int i = minX; i <= maxX; i++) {
						glm::vec2 pixelMiddle = { i + 0.5f, j + 0.5f };
						glm::vec3 barycentric = calculate_barycentric(pixelMiddle,
							texVertices[0], texVertices[1], texVertices[2]);
						if (barycentric.x >= -0.01 && barycentric.y >= -0.01 && barycentric.z >= -0.01) {
							Receiver receiver = {};
							receiver.position = apply_barycentric(barycentric, worldVertices[0], worldVertices[1], worldVertices[2]);
							receiver.normal = apply_barycentric(barycentric, worldNormals[0], worldNormals[1], worldNormals[2]);
							receiver.uv = glm::ivec2(i, j);
							receiver.exists = true;
							int receiverIndex = i + j * scene.lightmap_width;
							if (!_receivers[receiverIndex].exists) {
								_receivers[receiverIndex] = receiver;
								receiverCounter++;
							}
						}
					}
				}
			}
	}
	char* image = new char[scene.lightmap_height * scene.lightmap_width];
	memset(image, 0, scene.lightmap_height * scene.lightmap_width);
	for (int i = 0; i < scene.lightmap_height; i++) {
		for (int j = 0; j < scene.lightmap_width; j++) {
			if(_receivers[j + i * scene.lightmap_width].exists)
			image[j + i * scene.lightmap_width] = 255;
		}
	}

	//FILE* ptr;
	//fopen_s(&ptr, "../../precomputation/receiver_image.bin", "wb");
	//fwrite(image, scene.lightmap_height * scene.lightmap_width, 1, ptr);
	//fclose(ptr);
	//printf("Created receivers: %d!\n", receiverCounter);

	return _receivers;
}

//Maybe instead of doing this, do that:
//Do this in runtime (cast rays from each probe)
//Then trace another ray from the hit location to light (to realize shadowing)
//This might help me enabling reflections as well (perhaps)?
void Precalculation::probe_raycast(VulkanEngine& engine, std::vector<glm::vec4>& probes, int rays, int sphericalHarmonicsOrder, GPUProbeRaycastResult* probeRaycastResult, float* probeRaycastBasisFunctions)
{
	RaytracingPipeline rtPipeline = {};
	VkDescriptorSetLayout rtDescriptorSetLayout;
	VkDescriptorSet rtDescriptorSet;

	//We need a buffer where we store raytracing results
	//material_id, u, v

	//Descriptors: Acceleration structure, storage buffer to save results, Materials
	VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0);
	VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
	VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);
	VkDescriptorSetLayoutBinding probeLocationsBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3);
	VkDescriptorSetLayoutBinding outBuffer = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4);
	VkDescriptorSetLayoutBinding bindings[5] = { tlasBind, sceneDescBind, meshInfoBind, probeLocationsBind, outBuffer };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 5);
	vkCreateDescriptorSetLayout(engine._device, &setinfo, nullptr, &rtDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engine._descriptorPool, &rtDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engine._device, &allocateInfo, &rtDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &engine.vulkanRaytracing.tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = rtDescriptorSet;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	writes.emplace_back(accelerationStructureWrite);

	AllocatedBuffer sceneDescBuffer = vkutils::create_buffer(engine._allocator, sizeof(GPUSceneDesc), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, rtDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));

	AllocatedBuffer meshInfoBuffer = vkutils::create_buffer(engine._allocator, sizeof(GPUMeshInfo) * engine.gltf_scene.prim_meshes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));
	
	AllocatedBuffer probeLocationsBuffer = vkutils::create_buffer(engine._allocator, sizeof(glm::vec4) * probes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &probeLocationsBuffer._descriptorBufferInfo, 3));

	AllocatedBuffer outputBuffer = vkutils::create_buffer(engine._allocator, rays * probes.size() * sizeof(GPUProbeRaycastResult), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &outputBuffer._descriptorBufferInfo, 4));

	vkUpdateDescriptorSets(engine._device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vkinit::pipeline_layout_create_info(&rtDescriptorSetLayout, 1);

	engine.vulkanRaytracing.create_new_pipeline(rtPipeline, pipelineLayoutCreateInfo,
		"../../shaders/precalculate_probe_rt.rgen.spv",
		"../../shaders/precalculate_probe_rt.rmiss.spv",
		"../../shaders/precalculate_probe_rt.rchit.spv");

	{
		GPUSceneDesc desc = {};
		VkBufferDeviceAddressInfo info = { };
		info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

		info.buffer = engine.vertex_buffer._buffer;
		desc.vertexAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.normal_buffer._buffer;
		desc.normalAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.tex_buffer._buffer;
		desc.uvAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.index_buffer._buffer;
		desc.indexAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.lightmap_tex_buffer._buffer;
		desc.lightmapUvAddress = vkGetBufferDeviceAddress(engine._device, &info);

		vkutils::cpu_to_gpu(engine._allocator, sceneDescBuffer, &desc, sizeof(GPUSceneDesc));
	}

	{
		vkutils::cpu_to_gpu(engine._allocator, probeLocationsBuffer, probes.data(), sizeof(glm::vec4) * probes.size());
	}

	{
		void* data;
		vmaMapMemory(engine._allocator, meshInfoBuffer._allocation, &data);
		GPUMeshInfo* dataMesh = (GPUMeshInfo*) data;
		for (int i = 0; i < engine.gltf_scene.prim_meshes.size(); i++) {
			dataMesh[i].indexOffset = engine.gltf_scene.prim_meshes[i].first_idx;
			dataMesh[i].vertexOffset = engine.gltf_scene.prim_meshes[i].vtx_offset;
			dataMesh[i].materialIndex = engine.gltf_scene.prim_meshes[i].material_idx;
		}
		vmaUnmapMemory(engine._allocator, meshInfoBuffer._allocation);
	}

	{
		VkCommandBuffer cmdBuf = vkutils::create_command_buffer(engine._device, engine.vulkanRaytracing._raytracingContext._commandPool, true);
		std::vector<VkDescriptorSet> descSets{ rtDescriptorSet };
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipeline);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipelineLayout, 0,
			(uint32_t)descSets.size(), descSets.data(), 0, nullptr);
		vkCmdTraceRaysKHR(cmdBuf, &rtPipeline.rgenRegion, &rtPipeline.missRegion, &rtPipeline.hitRegion, &rtPipeline.callRegion, rays, probes.size(), 1);
		vkutils::submit_and_free_command_buffer(engine._device, engine.vulkanRaytracing._raytracingContext._commandPool, cmdBuf, engine.vulkanRaytracing._queue, engine.vulkanRaytracing._raytracingContext._fence);
	}

	void* mappedOutputData;
	vmaMapMemory(engine._allocator, outputBuffer._allocation, &mappedOutputData);
	//printf("Ray cast result: %d\n", ((ProbeRaycastResult*)data)[i].objectId);
	memcpy(probeRaycastResult, mappedOutputData, rays* probes.size() * sizeof(GPUProbeRaycastResult));
	vmaUnmapMemory(engine._allocator, outputBuffer._allocation);
	
	int shCoeff = SPHERICAL_HARMONICS_NUM_COEFF(sphericalHarmonicsOrder);

	for (int i = 0; i < rays * probes.size(); i++) {
		Eigen::Vector3d dir(probeRaycastResult[i].direction.x,
			probeRaycastResult[i].direction.y,
			probeRaycastResult[i].direction.z);
		dir.normalize();
		int ctr = 0;
		for (int l = 0; l <= sphericalHarmonicsOrder; l++) {
			for (int m = -l; m <= l; m++) {
				probeRaycastBasisFunctions[i * shCoeff + ctr] = (float) (sh::EvalSH(l, m, dir) * 4 * M_PI / rays);
				//printf("%d: %f\n", ctr, _probeRaycastBasisFunctions[i * SPHERICAL_HARMONICS_NUM_COEFF + ctr]);
				ctr++;
			} 
		}
	}

	engine.vulkanRaytracing.destroy_raytracing_pipeline(rtPipeline);

	vmaDestroyBuffer(engine._allocator, sceneDescBuffer._buffer, sceneDescBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, meshInfoBuffer._buffer, meshInfoBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, probeLocationsBuffer._buffer, probeLocationsBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, outputBuffer._buffer, outputBuffer._allocation);
	vkDestroyDescriptorSetLayout(engine._device, rtDescriptorSetLayout, nullptr);
}

void Precalculation::receiver_raycast(VulkanEngine& engine, std::vector<AABB>& aabbClusters, std::vector<glm::vec4>& probes, int rays, int radius, int sphericalHarmonicsOrder, int clusterCoefficientCount, float* clusterProjectionMatrices, float* receiverCoefficientMatrices)
{
	int shNumCoeff = SPHERICAL_HARMONICS_NUM_COEFF(sphericalHarmonicsOrder);

	RaytracingPipeline rtPipeline = {};
	VkDescriptorSetLayout rtDescriptorSetLayout;
	VkDescriptorSet rtDescriptorSet;

	//Descriptors: Acceleration structure, storage buffer to save results, Materials
	VkDescriptorSetLayoutBinding tlasBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0);
	VkDescriptorSetLayoutBinding sceneDescBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1);
	VkDescriptorSetLayoutBinding meshInfoBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2);
	VkDescriptorSetLayoutBinding probeLocationsBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3);
	VkDescriptorSetLayoutBinding receiverLocationsBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
		VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 4);
	VkDescriptorSetLayoutBinding outBuffer = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 5);
	VkDescriptorSetLayoutBinding bindings[6] = { tlasBind, sceneDescBind, meshInfoBind, probeLocationsBind, receiverLocationsBind, outBuffer };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 6);
	vkCreateDescriptorSetLayout(engine._device, &setinfo, nullptr, &rtDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engine._descriptorPool, &rtDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engine._device, &allocateInfo, &rtDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &engine.vulkanRaytracing.tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = rtDescriptorSet;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	writes.emplace_back(accelerationStructureWrite);

	AllocatedBuffer sceneDescBuffer = vkutils::create_buffer(engine._allocator, sizeof(GPUSceneDesc), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, rtDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));

	AllocatedBuffer meshInfoBuffer = vkutils::create_buffer(engine._allocator, sizeof(GPUMeshInfo) * engine.gltf_scene.prim_meshes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));

	AllocatedBuffer probeLocationsBuffer = vkutils::create_buffer(engine._allocator, sizeof(glm::vec4) * probes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &probeLocationsBuffer._descriptorBufferInfo, 3));

	AllocatedBuffer receiverLocationsBuffer = vkutils::create_buffer(engine._allocator, sizeof(GPUReceiverData) * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &receiverLocationsBuffer._descriptorBufferInfo, 4));

	AllocatedBuffer outputBuffer = vkutils::create_buffer(engine._allocator, 1024 * rays * probes.size() * sizeof(GPUReceiverRaycastResult), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &outputBuffer._descriptorBufferInfo, 5));

	vkUpdateDescriptorSets(engine._device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vkinit::pipeline_layout_create_info(&rtDescriptorSetLayout, 1);

	VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_RAYGEN_BIT_KHR , 0, sizeof(int) };
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;

	engine.vulkanRaytracing.create_new_pipeline(rtPipeline, pipelineLayoutCreateInfo,
		"../../shaders/precalculate_receiver_rt.rgen.spv",
		"../../shaders/precalculate_probe_rt.rmiss.spv",
		"../../shaders/precalculate_probe_rt.rchit.spv");

	{
		GPUSceneDesc desc = {};
		VkBufferDeviceAddressInfo info = { };
		info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

		info.buffer = engine.vertex_buffer._buffer;
		desc.vertexAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.normal_buffer._buffer;
		desc.normalAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.tex_buffer._buffer;
		desc.uvAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.index_buffer._buffer;
		desc.indexAddress = vkGetBufferDeviceAddress(engine._device, &info);

		info.buffer = engine.lightmap_tex_buffer._buffer;
		desc.lightmapUvAddress = vkGetBufferDeviceAddress(engine._device, &info);
		
		vkutils::cpu_to_gpu(engine._allocator, sceneDescBuffer, &desc, sizeof(GPUSceneDesc));
	}

	{
		vkutils::cpu_to_gpu(engine._allocator, probeLocationsBuffer, probes.data(), sizeof(glm::vec4) * probes.size());
	}

	{
		void* data;
		vmaMapMemory(engine._allocator, meshInfoBuffer._allocation, &data);
		GPUMeshInfo* dataMesh = (GPUMeshInfo*)data;
		for (int i = 0; i < engine.gltf_scene.prim_meshes.size(); i++) {
			dataMesh[i].indexOffset = engine.gltf_scene.prim_meshes[i].first_idx;
			dataMesh[i].vertexOffset = engine.gltf_scene.prim_meshes[i].vtx_offset;
			dataMesh[i].materialIndex = engine.gltf_scene.prim_meshes[i].material_idx;
		}
		vmaUnmapMemory(engine._allocator, meshInfoBuffer._allocation);
	}
	
	int nodeReceiverDataOffset = 0;
	for (int nodeIndex = 0; nodeIndex < aabbClusters.size(); nodeIndex++) {

		{
			void* data;
			vmaMapMemory(engine._allocator, receiverLocationsBuffer._allocation, &data);
			GPUReceiverData* dataReceiver = (GPUReceiverData*)data;
			for (int i = 0; i < aabbClusters[nodeIndex].receivers.size(); i++) {
				dataReceiver[i].pos = aabbClusters[nodeIndex].receivers[i].position;
				dataReceiver[i].normal = aabbClusters[nodeIndex].receivers[i].normal;
			}
			vmaUnmapMemory(engine._allocator, receiverLocationsBuffer._allocation);
		}

		{
			int probeCount = probes.size();
			VkCommandBuffer cmdBuf = vkutils::create_command_buffer(engine._device, engine.vulkanRaytracing._raytracingContext._commandPool, true);
			std::vector<VkDescriptorSet> descSets{ rtDescriptorSet };
			vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipeline);
			vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipelineLayout, 0,
				(uint32_t)descSets.size(), descSets.data(), 0, nullptr);
			vkCmdPushConstants(cmdBuf, rtPipeline.pipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0,
				sizeof(int), &probeCount);
			vkCmdTraceRaysKHR(cmdBuf, &rtPipeline.rgenRegion, &rtPipeline.missRegion, &rtPipeline.hitRegion, &rtPipeline.callRegion, aabbClusters[nodeIndex].receivers.size(), rays, 1);
			vkutils::submit_and_free_command_buffer(engine._device, engine.vulkanRaytracing._raytracingContext._commandPool, cmdBuf, engine.vulkanRaytracing._queue, engine.vulkanRaytracing._raytracingContext._fence);
		}

		Eigen::MatrixXf clusterMatrix = Eigen::MatrixXf(aabbClusters[nodeIndex].receivers.size(), shNumCoeff * probes.size());
		clusterMatrix.fill(0);
		void* mappedOutputData;
		vmaMapMemory(engine._allocator, outputBuffer._allocation, &mappedOutputData);

		// I need below as compute shader
#pragma omp parallel for
		for (int i = 0; i < aabbClusters[nodeIndex].receivers.size(); i++) {
			std::vector<float> weights;
			weights.resize(probes.size());
			for (int j = 0; j < probes.size(); j++) {
				weights[j] = calculate_density(glm::distance(aabbClusters[nodeIndex].receivers[i].position, glm::vec3(probes[j])), radius);
			}

			int validRays = 0;
			for (int j = 0; j < rays; j++) {
				float totalWeight = 0.f;
				for (int k = 0; k < probes.size(); k++) {
					int result = ((GPUReceiverRaycastResult*)mappedOutputData)[k + j * probes.size() + i * probes.size() * rays].visibility;
					totalWeight += result * weights[k];					
				}
				if (totalWeight > 0.f) {
					for (int k = 0; k < probes.size(); k++) {
						int ctr = 0;
						auto res = ((GPUReceiverRaycastResult*)mappedOutputData)[k + j * probes.size() + i * probes.size() * rays];
						Eigen::Vector3d dir(res.dir.x,
							res.dir.y,
							res.dir.z);
						dir.normalize();
						float weight = res.visibility * weights[k];

						for (int l = 0; l <= sphericalHarmonicsOrder; l++) {
							for (int m = -l; m <= l; m++) {
								clusterMatrix(i, k * shNumCoeff + ctr) += ((float)(sh::EvalSH(l, m, dir)) * weight) / totalWeight;
								ctr++;
							}
						}
					}
					validRays++;
				}

			}
			if (validRays > 0) {
				clusterMatrix.row(i) *= 2 * M_PI / validRays; //maybe do not divide to valid rays? rather divide to rays?
			}
		}
		
		//std::ofstream file("eigenmatrix" + std::to_string(nodeIndex) + ".csv");
		//const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
		//file << clusterMatrix.format(CSVFormat);

		Eigen::JacobiSVD<Eigen::MatrixXf> svd(clusterMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		int nc = clusterCoefficientCount;

		auto receiverReconstructionCoefficientMatrix = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>(svd.matrixU().block(0, 0, svd.matrixU().rows(), nc));

		Eigen::MatrixXf singularMatrix(nc, svd.matrixV().cols());
		singularMatrix.setZero();
		for (int aa = 0; aa < nc; aa++) {
			singularMatrix(aa, aa) = svd.singularValues()[aa];
		}
		auto clusterProjectionMatrix = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>(singularMatrix * svd.matrixV().transpose());
		
		//printf("Singular matrix size: %d x %d\n", singularMatrix.rows(), singularMatrix.cols());
		//printf("V matrix size: %d x %d\n", svd.matrixV().rows(), svd.matrixV().cols());

		/*
		auto newMatrix = receiverReconstructionCoefficientMatrix * clusterProjectionMatrix;

		printf("Size of the new matrix: %d x %d", newMatrix.rows(), newMatrix.cols());

		auto diff = clusterMatrix - newMatrix;

		printf("diff: %f\n", diff.array().abs().sum());
		*/
		
		/*
		int globalcounterr = 0;
		for (int y = 0; y < clusterProjectionMatrix.rows(); y++) {
			for (int x = 0; x < clusterProjectionMatrix.cols(); x++) {
				printf("%f vs %f\n", clusterProjectionMatrix(y, x), *(clusterProjectionMatrix.data() + globalcounterr));
				globalcounterr++;
			}
		}
		*/

		memcpy(clusterProjectionMatrices + clusterProjectionMatrix.size() * nodeIndex, clusterProjectionMatrix.data(), clusterProjectionMatrix.size() * sizeof(float));
		memcpy(receiverCoefficientMatrices + nodeReceiverDataOffset, receiverReconstructionCoefficientMatrix.data(), receiverReconstructionCoefficientMatrix.size() * sizeof(float));
		nodeReceiverDataOffset += receiverReconstructionCoefficientMatrix.size();
		printf("Famous offset %d: %d\n", nodeIndex, nodeReceiverDataOffset);
		vmaUnmapMemory(engine._allocator, outputBuffer._allocation);
		printf("Node tracing done %d/%d\n", nodeIndex, aabbClusters.size());
	}

	engine.vulkanRaytracing.destroy_raytracing_pipeline(rtPipeline);

	vmaDestroyBuffer(engine._allocator, sceneDescBuffer._buffer, sceneDescBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, meshInfoBuffer._buffer, meshInfoBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, probeLocationsBuffer._buffer, probeLocationsBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, receiverLocationsBuffer._buffer, probeLocationsBuffer._allocation);
	vmaDestroyBuffer(engine._allocator, outputBuffer._buffer, outputBuffer._allocation);
	vkDestroyDescriptorSetLayout(engine._device, rtDescriptorSetLayout, nullptr);
}
