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

#include <chrono>
#include <unordered_set>
#include <set>
#include <vk_pipeline.h>

#include <file_helper.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define USE_COMPUTE_PROBE_DENSITY_CALCULATION 1
#define M_PI    3.14159265358979323846264338327950288

void calcY(float* o, vec3 r, int order) {
	float x = r.x, y = r.y, z = r.z;

	if (order >= 0) {
		o[0] = 0.28209479177387814;
	}
	if (order >= 1) {
		o[1] = (-0.4886025119029199) * (y);
		o[2] = (0.4886025119029199) * (z);
		o[3] = (-0.4886025119029199) * (x);
	}
	if (order >= 2) {
		o[4] = (-1.0925484305920792) * (x) * (y);
		o[5] = (-1.0925484305920792) * (y) * (z);
		o[6] = (-0.31539156525252005) * ((x * x) + (y * y) + ((-2.) * (z * z)));
		o[7] = (-1.0925484305920792) * (x) * (z);
		o[8] = (0.5462742152960396) * ((x * x) + ((-1.) * (y * y)));
	}
	if (order >= 3) {
		o[9] = (0.5900435899266435) * (y) * (((-3.) * (x * x)) + (y * y));
		o[10] = (-2.890611442640554) * (x) * (y) * (z);
		o[11] = (0.4570457994644658) * (y) * ((x * x) + (y * y) + ((-4.) * (z * z)));
		o[12] = (0.3731763325901154) * (z) * (((-3.) * (x * x)) + ((-3.) * (y * y)) + ((2.) * (z * z)));
		o[13] = (0.4570457994644658) * (x) * ((x * x) + (y * y) + ((-4.) * (z * z)));
		o[14] = (1.445305721320277) * ((x * x) + ((-1.) * (y * y))) * (z);
		o[15] = (-0.5900435899266435) * (x) * ((x * x) + ((-3.) * (y * y)));
	}
	if (order >= 4) {
		o[16] = (2.5033429417967046) * (x) * (y) * (((-1.) * (x * x)) + (y * y));
		o[17] = (1.7701307697799304) * (y) * (((-3.) * (x * x)) + (y * y)) * (z);
		o[18] = (0.9461746957575601) * (x) * (y) * ((x * x) + (y * y) + ((-6.) * (z * z)));
		o[19] = (0.6690465435572892) * (y) * (z) * (((3.) * (x * x)) + ((3.) * (y * y)) + ((-4.) * (z * z)));
		o[20] = (0.10578554691520431) * (((3.) * (x * x * x * x)) + ((3.) * (y * y * y * y)) + ((-24.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)) + ((6.) * (x * x) * ((y * y) + ((-4.) * (z * z)))));
		o[21] = (0.6690465435572892) * (x) * (z) * (((3.) * (x * x)) + ((3.) * (y * y)) + ((-4.) * (z * z)));
		o[22] = (-0.47308734787878004) * ((x * x) + ((-1.) * (y * y))) * ((x * x) + (y * y) + ((-6.) * (z * z)));
		o[23] = (-1.7701307697799304) * (x) * ((x * x) + ((-3.) * (y * y))) * (z);
		o[24] = (0.6258357354491761) * ((x * x * x * x) + ((-6.) * (x * x) * (y * y)) + (y * y * y * y));
	}
	if (order >= 5) {
		o[25] = (-0.6563820568401701) * (y) * (((5.) * (x * x * x * x)) + ((-10.) * (x * x) * (y * y)) + (y * y * y * y));
		o[26] = (8.302649259524166) * (x) * (y) * (((-1.) * (x * x)) + (y * y)) * (z);
		o[27] = (-0.4892382994352504) * (y) * (((-3.) * (x * x)) + (y * y)) * ((x * x) + (y * y) + ((-8.) * (z * z)));
		o[28] = (4.793536784973324) * (x) * (y) * (z) * ((x * x) + (y * y) + ((-2.) * (z * z)));
		o[29] = (-0.45294665119569694) * (y) * ((x * x * x * x) + (y * y * y * y) + ((-12.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)) + ((2.) * (x * x) * ((y * y) + ((-6.) * (z * z)))));
		o[30] = (0.1169503224534236) * (z) * (((15.) * (x * x * x * x)) + ((15.) * (y * y * y * y)) + ((-40.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)) + ((10.) * (x * x) * (((3.) * (y * y)) + ((-4.) * (z * z)))));
		o[31] = (-0.45294665119569694) * (x) * ((x * x * x * x) + (y * y * y * y) + ((-12.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)) + ((2.) * (x * x) * ((y * y) + ((-6.) * (z * z)))));
		o[32] = (-2.396768392486662) * ((x * x) + ((-1.) * (y * y))) * (z) * ((x * x) + (y * y) + ((-2.) * (z * z)));
		o[33] = (0.4892382994352504) * (x) * ((x * x) + ((-3.) * (y * y))) * ((x * x) + (y * y) + ((-8.) * (z * z)));
		o[34] = (2.0756623148810416) * ((x * x * x * x) + ((-6.) * (x * x) * (y * y)) + (y * y * y * y)) * (z);
		o[35] = (-0.6563820568401701) * (x) * ((x * x * x * x) + ((-10.) * (x * x) * (y * y)) + ((5.) * (y * y * y * y)));
	}
	if (order >= 6) {
		o[36] = (-1.3663682103838286) * (x) * (y) * (((3.) * (x * x * x * x)) + ((-10.) * (x * x) * (y * y)) + ((3.) * (y * y * y * y)));
		o[37] = (-2.366619162231752) * (y) * (((5.) * (x * x * x * x)) + ((-10.) * (x * x) * (y * y)) + (y * y * y * y)) * (z);
		o[38] = (2.0182596029148967) * (x) * (y) * ((x * x) + ((-1.) * (y * y))) * ((x * x) + (y * y) + ((-10.) * (z * z)));
		o[39] = (-0.9212052595149236) * (y) * (((-3.) * (x * x)) + (y * y)) * (z) * (((3.) * (x * x)) + ((3.) * (y * y)) + ((-8.) * (z * z)));
		o[40] = (-0.9212052595149236) * (x) * (y) * ((x * x * x * x) + (y * y * y * y) + ((-16.) * (y * y) * (z * z)) + ((16.) * (z * z * z * z)) + ((2.) * (x * x) * ((y * y) + ((-8.) * (z * z)))));
		o[41] = (-0.5826213625187314) * (y) * (z) * (((5.) * (x * x * x * x)) + ((5.) * (y * y * y * y)) + ((-20.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)) + ((10.) * (x * x) * ((y * y) + ((-2.) * (z * z)))));
		o[42] = (-0.06356920226762842) * (((5.) * (x * x * x * x * x * x)) + ((5.) * (y * y * y * y * y * y)) + ((-90.) * (y * y * y * y) * (z * z)) + ((120.) * (y * y) * (z * z * z * z)) + ((-16.) * (z * z * z * z * z * z)) + ((15.) * (x * x * x * x) * ((y * y) + ((-6.) * (z * z)))) + ((15.) * (x * x) * ((y * y * y * y) + ((-12.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)))));
		o[43] = (-0.5826213625187314) * (x) * (z) * (((5.) * (x * x * x * x)) + ((5.) * (y * y * y * y)) + ((-20.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)) + ((10.) * (x * x) * ((y * y) + ((-2.) * (z * z)))));
		o[44] = (0.4606026297574618) * ((x * x) + ((-1.) * (y * y))) * ((x * x * x * x) + (y * y * y * y) + ((-16.) * (y * y) * (z * z)) + ((16.) * (z * z * z * z)) + ((2.) * (x * x) * ((y * y) + ((-8.) * (z * z)))));
		o[45] = (0.9212052595149236) * (x) * ((x * x) + ((-3.) * (y * y))) * (z) * (((3.) * (x * x)) + ((3.) * (y * y)) + ((-8.) * (z * z)));
		o[46] = (-0.5045649007287242) * ((x * x * x * x) + ((-6.) * (x * x) * (y * y)) + (y * y * y * y)) * ((x * x) + (y * y) + ((-10.) * (z * z)));
		o[47] = (-2.366619162231752) * (x) * ((x * x * x * x) + ((-10.) * (x * x) * (y * y)) + ((5.) * (y * y * y * y))) * (z);
		o[48] = (0.6831841051919143) * ((x * x * x * x * x * x) + ((-15.) * (x * x * x * x) * (y * y)) + ((15.) * (x * x) * (y * y * y * y)) + ((-1.) * (y * y * y * y * y * y)));
	}
	if (order >= 7) {
		o[49] = (0.7071627325245962) * (y) * (((-7.) * (x * x * x * x * x * x)) + ((35.) * (x * x * x * x) * (y * y)) + ((-21.) * (x * x) * (y * y * y * y)) + (y * y * y * y * y * y));
		o[50] = (-5.291921323603801) * (x) * (y) * (((3.) * (x * x * x * x)) + ((-10.) * (x * x) * (y * y)) + ((3.) * (y * y * y * y))) * (z);
		o[51] = (0.5189155787202604) * (y) * (((5.) * (x * x * x * x)) + ((-10.) * (x * x) * (y * y)) + (y * y * y * y)) * ((x * x) + (y * y) + ((-12.) * (z * z)));
		o[52] = (4.151324629762083) * (x) * (y) * ((x * x) + ((-1.) * (y * y))) * (z) * (((3.) * (x * x)) + ((3.) * (y * y)) + ((-10.) * (z * z)));
		o[53] = (0.15645893386229404) * (y) * (((-3.) * (x * x)) + (y * y)) * (((3.) * (x * x * x * x)) + ((3.) * (y * y * y * y)) + ((-60.) * (y * y) * (z * z)) + ((80.) * (z * z * z * z)) + ((6.) * (x * x) * ((y * y) + ((-10.) * (z * z)))));
		o[54] = (-0.4425326924449826) * (x) * (y) * (z) * (((15.) * (x * x * x * x)) + ((15.) * (y * y * y * y)) + ((-80.) * (y * y) * (z * z)) + ((48.) * (z * z * z * z)) + ((10.) * (x * x) * (((3.) * (y * y)) + ((-8.) * (z * z)))));
		o[55] = (0.0903316075825173) * (y) * (((5.) * (x * x * x * x * x * x)) + ((5.) * (y * y * y * y * y * y)) + ((-120.) * (y * y * y * y) * (z * z)) + ((240.) * (y * y) * (z * z * z * z)) + ((-64.) * (z * z * z * z * z * z)) + ((15.) * (x * x * x * x) * ((y * y) + ((-8.) * (z * z)))) + ((15.) * (x * x) * ((y * y * y * y) + ((-16.) * (y * y) * (z * z)) + ((16.) * (z * z * z * z)))));
		o[56] = (0.06828427691200495) * (z) * (((-35.) * (x * x * x * x * x * x)) + ((-35.) * (y * y * y * y * y * y)) + ((210.) * (y * y * y * y) * (z * z)) + ((-168.) * (y * y) * (z * z * z * z)) + ((16.) * (z * z * z * z * z * z)) + ((-105.) * (x * x * x * x) * ((y * y) + ((-2.) * (z * z)))) + ((-21.) * (x * x) * (((5.) * (y * y * y * y)) + ((-20.) * (y * y) * (z * z)) + ((8.) * (z * z * z * z)))));
		o[57] = (0.0903316075825173) * (x) * (((5.) * (x * x * x * x * x * x)) + ((5.) * (y * y * y * y * y * y)) + ((-120.) * (y * y * y * y) * (z * z)) + ((240.) * (y * y) * (z * z * z * z)) + ((-64.) * (z * z * z * z * z * z)) + ((15.) * (x * x * x * x) * ((y * y) + ((-8.) * (z * z)))) + ((15.) * (x * x) * ((y * y * y * y) + ((-16.) * (y * y) * (z * z)) + ((16.) * (z * z * z * z)))));
		o[58] = (0.2212663462224913) * ((x * x) + ((-1.) * (y * y))) * (z) * (((15.) * (x * x * x * x)) + ((15.) * (y * y * y * y)) + ((-80.) * (y * y) * (z * z)) + ((48.) * (z * z * z * z)) + ((10.) * (x * x) * (((3.) * (y * y)) + ((-8.) * (z * z)))));
		o[59] = (-0.15645893386229404) * (x) * ((x * x) + ((-3.) * (y * y))) * (((3.) * (x * x * x * x)) + ((3.) * (y * y * y * y)) + ((-60.) * (y * y) * (z * z)) + ((80.) * (z * z * z * z)) + ((6.) * (x * x) * ((y * y) + ((-10.) * (z * z)))));
		o[60] = (-1.0378311574405208) * ((x * x * x * x) + ((-6.) * (x * x) * (y * y)) + (y * y * y * y)) * (z) * (((3.) * (x * x)) + ((3.) * (y * y)) + ((-10.) * (z * z)));
		o[61] = (0.5189155787202604) * (x) * ((x * x * x * x) + ((-10.) * (x * x) * (y * y)) + ((5.) * (y * y * y * y))) * ((x * x) + (y * y) + ((-12.) * (z * z)));
		o[62] = (2.6459606618019005) * ((x * x * x * x * x * x) + ((-15.) * (x * x * x * x) * (y * y)) + ((15.) * (x * x) * (y * y * y * y)) + ((-1.) * (y * y * y * y * y * y))) * (z);
		o[63] = (-0.7071627325245962) * (x) * ((x * x * x * x * x * x) + ((-21.) * (x * x * x * x) * (y * y)) + ((35.) * (x * x) * (y * y * y * y)) + ((-7.) * (y * y * y * y * y * y)));
	}
}

static float calculate_density(float length, float radius) {
	//OPTICK_EVENT();

	float t = length / radius;
	if (t >= 0 && t <= 1) {
		float tSquared = t * t;
		return (2 * t * tSquared) - (3 * tSquared) + 1;
	}
	return 0;
}

static float calculate_spatial_weight(float radius, int probeIndex, std::vector<glm::vec4>& probes) {
	//OPTICK_EVENT();

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
	//OPTICK_EVENT();

	float radius = 0;
	int receiverCount = 0;
	Receiver* previousReceiver = nullptr;
#pragma omp parallel for
	for (int r = 0; r < receiverSize; r += 1) {
		if (!receivers[r].exists) {
			continue;
		}
		//if (previousReceiver != nullptr) {
		//	printf("Distance between them is %f\n", glm::distance(receivers[r].position, previousReceiver->position));
		//}
		//if (previousReceiver != nullptr && glm::distance(receivers[r].position, previousReceiver->position) < 100) {
		//	continue;
		//}

		std::priority_queue <float, std::vector<float>, std::greater<float>> minheap;

		for (int p = 0; p < probes.size(); p++) {
			minheap.push(glm::distance(glm::vec3(probes[p]), receivers[r].position));
		}

		for (int i = 0; i < overlaps - 1; i++) {
			minheap.pop();
		}

#pragma omp critical
		{
		radius += minheap.top();
		receiverCount++;
		}
		//previousReceiver = &receivers[r];
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

void Precalculation::prepare(VulkanEngine& engine, GltfScene& scene, PrecalculationInfo precalculationInfo, PrecalculationLoadData& outPrecalculationLoadData, PrecalculationResult& outPrecalculationResult, const char* loadProbes) {
	printf("Scene center: %f x %f x %f\n", scene.m_dimensions.center.x, scene.m_dimensions.center.y, scene.m_dimensions.center.z);
	printf("Scene dimensions: %f x %f x %f\n", scene.m_dimensions.size.x, scene.m_dimensions.size.y, scene.m_dimensions.size.z);
	Receiver* receivers = generate_receivers_cpu(engine, scene, precalculationInfo.lightmapResolution);
	std::vector<glm::vec4> probes;

	if (loadProbes == nullptr) {
		int voxelDimX, voxelDimY, voxelDimZ;
		uint8_t* voxelData = voxelize(scene, precalculationInfo.voxelSize, precalculationInfo.voxelPadding, voxelDimX, voxelDimY, voxelDimZ);

		//Place probes everywhere in the voxelized scene
		for (int k = precalculationInfo.voxelPadding; k < voxelDimZ - precalculationInfo.voxelPadding; k += 1) {
			for (int j = precalculationInfo.voxelPadding; j < voxelDimY - precalculationInfo.voxelPadding; j += 1) {
				for (int i = precalculationInfo.voxelPadding; i < voxelDimX - precalculationInfo.voxelPadding; i += 1) {
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

		float desiredSpacing = precalculationInfo.desiredSpacing;
		int targetProbeCount = (scene.m_dimensions.size.x / desiredSpacing) * (scene.m_dimensions.size.y / desiredSpacing) * (scene.m_dimensions.size.z / desiredSpacing);
		printf("Desired spacing: %d. Targeted amount of probes is %d. Current probes: %d\n", desiredSpacing, targetProbeCount, probes.size());

		//probes.erase(std::remove_if(probes.begin(), probes.end(),
		//	[](glm::vec4 i) { return i.y < 0; }), probes.end());

		if (true) {
			std::ofstream file("mesh_voxelized.obj");
			int counter = 0;

			for (int k = 0; k < voxelDimZ; k++) {
				for (int j = 0; j < voxelDimY; j++) {
					for (int i = 0; i < voxelDimX; i++) {
						int index = i + j * voxelDimX + k * voxelDimX * voxelDimY;

						if (voxelData[index] != 1) {
							continue;
						}

						float voxelX = scene.m_dimensions.min.x + precalculationInfo.voxelSize * (i - precalculationInfo.voxelPadding);
						float voxelY = scene.m_dimensions.min.y + precalculationInfo.voxelSize * (j - precalculationInfo.voxelPadding);
						float voxelZ = scene.m_dimensions.min.z + precalculationInfo.voxelSize * (k - precalculationInfo.voxelPadding);

						//Vertices
						file << "v " << voxelX << " " << voxelY << " " << voxelZ << "\n";
						file << "v " << voxelX << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + precalculationInfo.voxelSize << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + +precalculationInfo.voxelSize << " " << voxelY << " " << voxelZ << "\n";
						file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ << "\n";
						file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + precalculationInfo.voxelSize << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + +precalculationInfo.voxelSize << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ << "\n";

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
			file.close();
		}

		{
			std::ofstream file("mesh_voxelized_probe.obj");
			int counter = 0;

			for (int i = 0; i < probes.size(); i++) {
				float voxelX = probes[i].x;
				float voxelY = probes[i].y;
				float voxelZ = probes[i].z;

				//Vertices
				file << "v " << voxelX << " " << voxelY << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + precalculationInfo.voxelSize / 4.f << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + +precalculationInfo.voxelSize / 4.f << " " << voxelY << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + precalculationInfo.voxelSize / 4.f << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + +precalculationInfo.voxelSize / 4.f << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";

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


		place_probes(engine, probes, targetProbeCount, desiredSpacing);

		////////

		if (true) {
			std::ofstream file("mesh_voxelized_probe_selected.obj");
			int counter = 0;
			/*
			for (int k = 0; k < voxelDimZ; k++) {
				for (int j = 0; j < voxelDimY; j++) {
					for (int i = 0; i < voxelDimX; i++) {
						int index = i + j * voxelDimX + k * voxelDimX * voxelDimY;

						if (voxelData[index] != 1) {
							continue;
						}

						float voxelX = scene.m_dimensions.min.x + precalculationInfo.voxelSize * (i - precalculationInfo.voxelPadding);
						float voxelY = scene.m_dimensions.min.y + precalculationInfo.voxelSize * (j - precalculationInfo.voxelPadding);
						float voxelZ = scene.m_dimensions.min.z + precalculationInfo.voxelSize * (k - precalculationInfo.voxelPadding);

						//Vertices
						file << "v " << voxelX << " " << voxelY << " " << voxelZ << "\n";
						file << "v " << voxelX << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + precalculationInfo.voxelSize << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + +precalculationInfo.voxelSize << " " << voxelY << " " << voxelZ << "\n";
						file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ << "\n";
						file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + precalculationInfo.voxelSize << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ + precalculationInfo.voxelSize << "\n";
						file << "v " << voxelX + +precalculationInfo.voxelSize << " " << voxelY + precalculationInfo.voxelSize << " " << voxelZ << "\n";

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
			}*/

			for (int i = 0; i < probes.size(); i++) {
				float voxelX = probes[i].x;
				float voxelY = probes[i].y;
				float voxelZ = probes[i].z;

				//Vertices
				file << "v " << voxelX << " " << voxelY << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + precalculationInfo.voxelSize / 4.f << " " << voxelY << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + +precalculationInfo.voxelSize / 4.f << " " << voxelY << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + precalculationInfo.voxelSize / 4.f << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ + precalculationInfo.voxelSize / 4.f << " " << 1 << " " << 0 << " " << 0 << "\n";
				file << "v " << voxelX + +precalculationInfo.voxelSize / 4.f << " " << voxelY + precalculationInfo.voxelSize / 4.f << " " << voxelZ << " " << 1 << " " << 0 << " " << 0 << "\n";

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
		delete[] voxelData;
	}
	else {
		size_t size = seek_file(loadProbes);
		printf("The size is %d\n", size);
		probes.resize(size / sizeof(glm::vec4));
		load_binary(loadProbes, probes.data(), size);
		printf("Loaded %d probes.\n", probes.size());
	}
	////////

	outPrecalculationLoadData.probesCount = probes.size();
	outPrecalculationResult.probes = probes;
	outPrecalculationResult.probeRaycastResult = (GPUProbeRaycastResult*)malloc(precalculationInfo.raysPerProbe * probes.size() * sizeof(GPUProbeRaycastResult));
	outPrecalculationResult.probeRaycastBasisFunctions = (float*)malloc(precalculationInfo.raysPerProbe * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * sizeof(float));
	probe_raycast(engine, probes, precalculationInfo.raysPerProbe, precalculationInfo.sphericalHarmonicsOrder, outPrecalculationResult.probeRaycastResult, outPrecalculationResult.probeRaycastBasisFunctions);

	//Receiver radius
	float newRadius = calculate_radius(receivers, precalculationInfo.lightmapResolution * precalculationInfo.lightmapResolution, probes, precalculationInfo.probeOverlaps);
	printf("Radius for receivers: %f\n", newRadius);

	
	for (int i = 0; i < precalculationInfo.lightmapResolution * precalculationInfo.lightmapResolution; i++) {
		if (receivers[i].exists) {
			for (int k = 0; k < probes.size(); k++) {
				if (calculate_density(glm::distance(receivers[i].position, glm::vec3(probes[k])), newRadius) > 0.0f) {
					receivers[i].probes.push_back(k);
				}
			}
		}
	}
	

	//AABB clustering
	std::vector<AABB> aabbClusters;
	AABB initial_node = {};
	initial_node.min = scene.m_dimensions.min - glm::vec3(1.0); //adding some padding
	initial_node.max = scene.m_dimensions.max + glm::vec3(1.0); //adding some padding
	divide_aabb(aabbClusters, initial_node, receivers, precalculationInfo.lightmapResolution * precalculationInfo.lightmapResolution, precalculationInfo.maxReceiversInCluster);
	
	//my adaptive clustering base on normal

	
	for (int i = 0; i < aabbClusters.size(); i++) {
		printf("Current cluster (%d): %d/%d\n", aabbClusters[i].receivers.size(), i, aabbClusters.size());

		if (aabbClusters[i].receivers.size() <= 64) {
			continue;
		}

		bool first_mismatch = false;
		auto j = aabbClusters[i].receivers.begin();

		int mismatchCount = 0;

		while (j != aabbClusters[i].receivers.end())
		{
			int diff = (aabbClusters[i].receivers[0].probes.size() - (*j).probes.size());
			if (diff < 0) diff *= -1;
			if ( (glm::distance(aabbClusters[i].receivers[0].normal, (*j).normal) > 0.1 && glm::dot(aabbClusters[i].receivers[0].normal, (*j).normal) < 0.9f)) {
				mismatchCount++;
			}
			++j;
		}

		printf("Mismatch count: %d\n", mismatchCount);

		if (mismatchCount > 32 && aabbClusters[i].receivers.size() - mismatchCount > 32) {
			j = aabbClusters[i].receivers.begin();
			while (j != aabbClusters[i].receivers.end())
			{
				int diff = (aabbClusters[i].receivers[0].probes.size() - (*j).probes.size());
				if (diff < 0) diff *= -1;
				if ( (glm::distance(aabbClusters[i].receivers[0].normal, (*j).normal) > 0.1 && glm::dot(aabbClusters[i].receivers[0].normal, (*j).normal) < 0.9f)) {
					if (!first_mismatch) {
						AABB new_cluster = {};
						aabbClusters.push_back(new_cluster);
						first_mismatch = true;
					}
					aabbClusters[aabbClusters.size() - 1].receivers.push_back((*j));
					j = aabbClusters[i].receivers.erase(j);
				}
				else {
					++j;
				}
			}
		}
	}
	
	
	delete[] receivers;

	/*
	for (int i = 0; i < aabbClusters.size(); i++) {
		std::sort(aabbClusters[i].receivers.begin(), aabbClusters[i].receivers.end(), [](Receiver a, Receiver b) {
			float adot = glm::dot(a.normal, a.normal);
			float bdot = glm::dot(b.normal, b.normal);
	
			if (std::abs(adot - bdot) < 0.00001) {
				if (std::abs(a.normal.x - b.normal.x) > 0.00001) {
					return a.normal.x > b.normal.x;
				}
				else if (std::abs(a.normal.y - b.normal.y) > 0.00001) {
					return a.normal.y > b.normal.y;
				}
				else if (std::abs(a.normal.z - b.normal.z) > 0.00001) {
					return a.normal.z > b.normal.z;
				}
				else {
					return false;
				}
			}
			else {
				return adot > bdot;
			}
		});
		//printf("--------------------------------------------------------------\n");
		//for (int j = 0; j < aabbClusters[i].receivers.size(); j++) {
		//	printf("%f %f %f\n", aabbClusters[i].receivers[j].normal.x, aabbClusters[i].receivers[j].normal.y, aabbClusters[i].receivers[j].normal.z);
		//}
	}*/

	outPrecalculationLoadData.aabbClusterCount = aabbClusters.size();

	//Allocate memory for the results
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

	int maxProbesPerCluster = 0;
	int totalProbesPerCluster = 0;
	{
		int recCounter = 0;
		std::set<int> supportingProbes;
		for (int i = 0; i < aabbClusters.size(); i++) {
			//#pragma omp parallel for
			for (int j = 0; j < aabbClusters[i].receivers.size(); j++) {
				for (int k = 0; k < probes.size(); k++) {
					if (calculate_density(glm::distance(aabbClusters[i].receivers[j].position, glm::vec3(probes[k])), newRadius) > 0.0f) {
						supportingProbes.insert(k);
					}
				}
			}

			for (const auto elem : supportingProbes) {
				aabbClusters[i].probes.push_back(elem);
			}

			maxProbesPerCluster = MAX(maxProbesPerCluster, supportingProbes.size());
			totalProbesPerCluster += supportingProbes.size();
			recCounter += aabbClusters[i].receivers.size();
			aabbClusters[i].probeCount = supportingProbes.size();
			supportingProbes.clear();
		}
		printf("MAX probe count per cluster: %d\n", maxProbesPerCluster);
		printf("Total probes: %d\n", totalProbesPerCluster);
	}


	outPrecalculationLoadData.maxProbesPerCluster = maxProbesPerCluster;
	outPrecalculationLoadData.totalProbesPerCluster = totalProbesPerCluster;

	outPrecalculationResult.receiverProbeWeightData = new float[clusterReceiverCount * maxProbesPerCluster];
	outPrecalculationResult.clusterProjectionMatrices = new float[totalProbesPerCluster * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * precalculationInfo.clusterCoefficientCount];
	outPrecalculationResult.clusterProbes = new int[totalProbesPerCluster];

	{
		int recCounter = 0;
		int probeCounter = 0;
		int ctr = 0;
		std::set<int> supportingProbes;
		for (int i = 0; i < aabbClusters.size(); i++) {
			for (int j = 0; j < aabbClusters[i].receivers.size(); j++) {
				for (int k = 0; k < aabbClusters[i].probes.size(); k++) {
					
					float value = calculate_density(glm::distance(aabbClusters[i].receivers[j].position, glm::vec3(probes[aabbClusters[i].probes[k]])), newRadius);
					if (value > 0.0f) {
						supportingProbes.insert(aabbClusters[i].probes[k]);
					}
					outPrecalculationResult.receiverProbeWeightData[(recCounter + j) * maxProbesPerCluster + k] = value;
				}
			}

			for (const auto elem : supportingProbes) {
				outPrecalculationResult.clusterProbes[ctr] = elem;
				ctr++;
			}

			recCounter += aabbClusters[i].receivers.size();
			outPrecalculationResult.clusterReceiverInfos[i].probeCount = supportingProbes.size();
			outPrecalculationResult.clusterReceiverInfos[i].probeOffset = probeCounter;
			probeCounter += supportingProbes.size();
			supportingProbes.clear();
		}
	}

	for (int i = 0; i < aabbClusters.size(); i++) {
		if (aabbClusters[i].probeCount == 0) {
			printf("Cluster(%d) with no probe!\n", i);
		}
	}

	receiver_raycast(engine, aabbClusters, probes, precalculationInfo.raysPerReceiver, newRadius, precalculationInfo.sphericalHarmonicsOrder, precalculationInfo.clusterCoefficientCount, precalculationInfo.maxReceiversInCluster, clusterReceiverCount, maxProbesPerCluster, outPrecalculationResult.clusterProjectionMatrices, outPrecalculationResult.receiverCoefficientMatrices, outPrecalculationResult.receiverProbeWeightData, outPrecalculationResult.clusterProbes);

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
	config["lightmapResolution"] = precalculationInfo.lightmapResolution;
	config["texelSize"] = precalculationInfo.texelSize;
	config["desiredSpacing"] = precalculationInfo.desiredSpacing;

	config["probesCount"] = outPrecalculationLoadData.probesCount;
	config["totalClusterReceiverCount"] = outPrecalculationLoadData.totalClusterReceiverCount;
	config["aabbClusterCount"] = outPrecalculationLoadData.aabbClusterCount;
	config["maxProbesPerCluster"] = outPrecalculationLoadData.maxProbesPerCluster;
	config["totalProbesPerCluster"] = outPrecalculationLoadData.totalProbesPerCluster;

	config["fileProbes"] = filename + ".Probes";
	save_binary(filename + ".Probes", outPrecalculationResult.probes.data(), sizeof(glm::vec4) * outPrecalculationResult.probes.size());
	
	config["fileProbeRaycastResult"] = filename + ".ProbeRaycastResult";
	save_binary(filename + ".ProbeRaycastResult", outPrecalculationResult.probeRaycastResult, precalculationInfo.raysPerProbe * probes.size() * sizeof(GPUProbeRaycastResult));
	
	config["fileProbeRaycastBasisFunctions"] = filename + ".ProbeRaycastBasisFunctions";
	save_binary(filename + ".ProbeRaycastBasisFunctions", outPrecalculationResult.probeRaycastBasisFunctions, precalculationInfo.raysPerProbe * SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder) * sizeof(float));

	config["fileAabbReceivers"] = filename + ".AabbReceivers";
	save_binary(filename + ".AabbReceivers", outPrecalculationResult.aabbReceivers, sizeof(Receiver) * clusterReceiverCount);

	config["fileClusterProjectionMatrices"] = filename + ".ClusterProjectionMatrices";
	save_binary(filename + ".ClusterProjectionMatrices", outPrecalculationResult.clusterProjectionMatrices, totalProbesPerCluster* SPHERICAL_HARMONICS_NUM_COEFF(precalculationInfo.sphericalHarmonicsOrder)* precalculationInfo.clusterCoefficientCount * sizeof(float));

	config["fileReceiverCoefficientMatrices"] = filename + ".ReceiverCoefficientMatrices";
	save_binary(filename + ".ReceiverCoefficientMatrices", outPrecalculationResult.receiverCoefficientMatrices, clusterReceiverCount * precalculationInfo.clusterCoefficientCount * sizeof(float));

	config["fileClusterReceiverInfos"] = filename + ".ClusterReceiverInfos";
	save_binary(filename + ".ClusterReceiverInfos", outPrecalculationResult.clusterReceiverInfos, aabbClusters.size() * sizeof(ClusterReceiverInfo));

	config["fileClusterReceiverUvs"] = filename + ".ClusterReceiverUvs";
	save_binary(filename + ".ClusterReceiverUvs", outPrecalculationResult.clusterReceiverUvs, clusterReceiverCount * sizeof(glm::ivec4));

	config["fileReceiverProbeWeightData"] = filename + ".ReceiverProbeWeightData";
	save_binary(filename + ".ReceiverProbeWeightData", outPrecalculationResult.receiverProbeWeightData, clusterReceiverCount * maxProbesPerCluster * sizeof(float));

	config["fileClusterProbes"] = filename + ".ClusterProbes";
	save_binary(filename + ".ClusterProbes", outPrecalculationResult.clusterProbes, totalProbesPerCluster * sizeof(int));

	std::ofstream file(filename + ".cfg");
	file << config.dump();
	file.close();

	// Deal with this later
	
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
	precalculationInfo.lightmapResolution = config["lightmapResolution"];
	precalculationInfo.texelSize = config["texelSize"];
	precalculationInfo.desiredSpacing = config["desiredSpacing"];

	outPrecalculationLoadData.probesCount = config["probesCount"];
	outPrecalculationLoadData.totalClusterReceiverCount = config["totalClusterReceiverCount"];
	outPrecalculationLoadData.aabbClusterCount = config["aabbClusterCount"];
	outPrecalculationLoadData.maxProbesPerCluster = config["maxProbesPerCluster"];
	outPrecalculationLoadData.totalProbesPerCluster = config["totalProbesPerCluster"];

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

	{
		std::string file = config["fileReceiverProbeWeightData"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.receiverProbeWeightData = (float*)malloc(size);
		load_binary(file, outPrecalculationResult.receiverProbeWeightData, size);
	}

	{
		std::string file = config["fileClusterProbes"].get<std::string>();
		printf(file.c_str());
		printf("\n");
		size_t size = seek_file(file);
		outPrecalculationResult.clusterProbes = (int*)malloc(size);
		load_binary(file, outPrecalculationResult.clusterProbes, size);
	}
}

uint8_t* Precalculation::voxelize(GltfScene& scene, float voxelSize, int padding, int& dimX, int& dimY, int& dimZ)
{
	//OPTICK_EVENT();

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

void Precalculation::place_probes(VulkanEngine& engine, std::vector<glm::vec4>& probes, int targetProbeCount, float spacing)
{
	OPTICK_EVENT()

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
	ComputeInstance instance = {};
	engine._vulkanCompute.create_buffer(instance, UNIFORM, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(GPUProbeDensityUniformData));
	engine._vulkanCompute.create_buffer(instance, STORAGE, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::vec4) * probes.size());
	engine._vulkanCompute.create_buffer(instance, STORAGE, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(float) * probes.size());
	engine._vulkanCompute.build(instance, engine._engineData.descriptorPool, "../../shaders/precalculate_probe_density_weights.comp.spv");
	vkutils::cpu_to_gpu(engine._engineData.allocator, instance.bindings[1].buffer, probes.data(), sizeof(glm::vec4) * probes.size());
#endif


	float currMaxWeight = -1;
	int toRemoveIndex = -1;
	float radius = spacing;

	while (probes.size() > targetProbeCount) {
#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
		GPUProbeDensityUniformData ub = { probes.size(), radius };
		vkutils::cpu_to_gpu(engine._engineData.allocator, instance.bindings[0].buffer, &ub, sizeof(GPUProbeDensityUniformData));
		int groupcount = ((probes.size()) / 256) + 1;
		engine._vulkanCompute.compute(instance, groupcount, 1, 1);

		void* gpuWeightData;
		vmaMapMemory(engine._engineData.allocator, instance.bindings[2].buffer._allocation, &gpuWeightData);
#endif
		for (int i = 0; i < probes.size(); i++) {
#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
			float weight = ((float*)gpuWeightData)[i];
#else
			float weight = calculate_spatial_weight(radius, i, probes);
#endif
			if (weight >= currMaxWeight) {
				toRemoveIndex = i;
				currMaxWeight = weight;
			}
		}

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
		vmaUnmapMemory(engine._engineData.allocator, instance.bindings[2].buffer._allocation);

		void* gpuProbeData;
		glm::vec4* castedGpuProbeData;
		vmaMapMemory(engine._engineData.allocator, instance.bindings[1].buffer._allocation, &gpuProbeData);
		castedGpuProbeData = (glm::vec4*)gpuProbeData;
		castedGpuProbeData[toRemoveIndex] = castedGpuProbeData[probes.size() - 1];
		vmaUnmapMemory(engine._engineData.allocator, instance.bindings[1].buffer._allocation);
#endif

		if (toRemoveIndex != -1) {
			probes[toRemoveIndex] = probes[probes.size() - 1];
			probes.pop_back();
			currMaxWeight = -1;
			toRemoveIndex = -1;
		}
		else {
			break;
		}
		printf("Size of probes %d\n", probes.size());
	}

	printf("Found this many probes: %d\n", probes.size());

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
	engine._vulkanCompute.destroy_compute_instance(instance);
#endif
}

Receiver* Precalculation::generate_receivers_cpu(VulkanEngine& engine, GltfScene& scene, int lightmapResolution) {
	//OPTICK_EVENT();
	printf("Generating receivers!\n");

	int receiverCount = lightmapResolution * lightmapResolution;
	Receiver* _receivers = new Receiver[receiverCount];
	memset(_receivers, 0, sizeof(Receiver) * receiverCount);
	int receiverCounter = 0;

	for (int nodeIndex = 0; nodeIndex < scene.nodes.size(); nodeIndex++) {
		auto& mesh = scene.prim_meshes[scene.nodes[nodeIndex].prim_mesh];
		for (int triangle = 0; triangle < mesh.idx_count; triangle += 3) {
			glm::vec3 worldVertices[3];
			glm::vec3 worldNormals[3];
			glm::vec2 texVertices[3];
			int minX = lightmapResolution, minY = lightmapResolution;
			int maxX = 0, maxY = 0;
			for (int i = 0; i < 3; i++) {
				int vertexIndex = mesh.vtx_offset + scene.indices[mesh.first_idx + triangle + i];

				glm::vec4 vertex = scene.nodes[nodeIndex].world_matrix * glm::vec4(scene.positions[vertexIndex], 1.0);
				worldVertices[i] = glm::vec3(vertex / vertex.w);


				worldNormals[i] = glm::mat3(glm::transpose(glm::inverse(scene.nodes[nodeIndex].world_matrix))) * scene.normals[vertexIndex];

				texVertices[i] = scene.lightmapUVs[vertexIndex] * glm::vec2(lightmapResolution / (float) scene.lightmap_width, lightmapResolution / (float) scene.lightmap_height);

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
					int maxSample = TEXEL_SAMPLES;
					int receiverIndex = i + j * lightmapResolution;
					
					if (_receivers[receiverIndex].exists && nodeIndex != _receivers[receiverIndex].objectId) {
						continue;
					}

					for (int sample = 0; sample < maxSample * maxSample; sample++) {
						glm::vec2 pixelMiddle = { i + (sample / maxSample) / ((float)(maxSample - 1)), j + (sample % maxSample) / ((float)(maxSample - 1)) };
						glm::vec3 barycentric = calculate_barycentric(pixelMiddle,
							texVertices[0], texVertices[1], texVertices[2]);
						if (barycentric.x >= 0 && barycentric.y >= 0 && barycentric.z >= 0) {
							if (!_receivers[receiverIndex].exists) {
								_receivers[receiverIndex].exists = true;
								_receivers[receiverIndex].objectId = nodeIndex;
								_receivers[receiverIndex].uv = glm::ivec2(i, j);
								receiverCounter++;
							}

							_receivers[receiverIndex].position = apply_barycentric(barycentric, worldVertices[0], worldVertices[1], worldVertices[2]);
							_receivers[receiverIndex].normal = apply_barycentric(barycentric, worldNormals[0], worldNormals[1], worldNormals[2]);
							
							_receivers[receiverIndex].poses.push_back(apply_barycentric(barycentric, worldVertices[0], worldVertices[1], worldVertices[2]));
							_receivers[receiverIndex].norms.push_back(apply_barycentric(barycentric, worldNormals[0], worldNormals[1], worldNormals[2]));
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < lightmapResolution; i++) {
		for (int j = 0; j < lightmapResolution; j++) {
			if (!_receivers[j + i * lightmapResolution].exists) {
				if (j + 1 < lightmapResolution && _receivers[j + 1 + i * lightmapResolution].exists && !_receivers[j + 1 + i * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j + 1 + i * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (j - 1 >= 0 && _receivers[j - 1 + i * lightmapResolution].exists && !_receivers[j - 1 + i * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j - 1 + i * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (i + 1 < lightmapResolution && _receivers[j + (i + 1) * lightmapResolution].exists && !_receivers[j + (i + 1) * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j + (i + 1) * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (i - 1 >= 0 && _receivers[j + (i - 1) * lightmapResolution].exists && !_receivers[j + (i - 1) * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j + (i - 1) * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (j + 1 < lightmapResolution && i + 1 < lightmapResolution && _receivers[j + 1 + (i + 1) * lightmapResolution].exists && !_receivers[j + 1 + (i + 1) * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j + 1 + (i + 1) * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (j + 1 < lightmapResolution && i - 1 >= 0 && _receivers[j + 1 + (i - 1) * lightmapResolution].exists && !_receivers[j + 1 + (i - 1) * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j + 1 + (i - 1) * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (j - 1 >= 0 && i + 1 < lightmapResolution && _receivers[j - 1 + (i + 1) * lightmapResolution].exists && !_receivers[j - 1 + (i + 1) * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j - 1 + (i + 1) * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
				else if (j - 1 >= 0 && i - 1 >= 0 && _receivers[j - 1 + (i - 1) * lightmapResolution].exists && !_receivers[j - 1 + (i - 1) * lightmapResolution].processed) {
					_receivers[j + i * lightmapResolution] = _receivers[j - 1 + (i - 1) * lightmapResolution];
					_receivers[j + i * lightmapResolution].processed = true;
					_receivers[j + i * lightmapResolution].uv = glm::ivec2(j, i);
				}
			}
		}
	}

	char* image = new char[lightmapResolution * lightmapResolution];
	memset(image, 0, lightmapResolution * lightmapResolution);
	for (int i = 0; i < lightmapResolution; i++) {
		for (int j = 0; j < lightmapResolution; j++) {
			if (_receivers[j + i * lightmapResolution].exists) {
				image[j + i * lightmapResolution] = 255;
			}
		}
	}

	FILE* ptr;
	fopen_s(&ptr, "../../precomputation/receiver_image.bin", "wb");
	fwrite(image, lightmapResolution * lightmapResolution, 1, ptr);
	fclose(ptr);
	printf("Created receivers: %d!\n", receiverCounter);

	return _receivers;
}


Receiver* Precalculation::generate_receivers(VulkanEngine& engine, GltfScene& scene, int lightmapResolution)
{
	//OPTICK_EVENT();
	printf("Generating receivers!\n");

	int receiverCount = lightmapResolution * lightmapResolution;
	Receiver* _receivers = new Receiver[receiverCount];
	memset(_receivers, 0, sizeof(Receiver) * receiverCount);
	int receiverCounter = 0;
	
	VkExtent2D imageSize = { lightmapResolution, lightmapResolution };

	//Render pass
	VkRenderPass gbufferRenderPass;
	{
		VkAttachmentDescription attachmentDescs[2] = {};

		for (uint32_t i = 0; i < 2; ++i)
		{
			attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
			attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

			attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		}

		attachmentDescs[0].format = engine._engineData.color32Format;
		attachmentDescs[1].format = engine._engineData.color32Format;

		VkAttachmentReference colorReferences[2] = {};
		colorReferences[0] = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		colorReferences[1] = { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.pColorAttachments = colorReferences;
		subpass.colorAttachmentCount = static_cast<uint32_t>(2);
		subpass.pDepthStencilAttachment = nullptr;

		VkSubpassDependency dependencies[2] = {};

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.pAttachments = attachmentDescs;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(2);
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies;

		VK_CHECK(vkCreateRenderPass(engine._engineData.device, &renderPassInfo, nullptr, &gbufferRenderPass));
	}

	//Images
	AllocatedImage gbufferPosObjectIdImage;
	AllocatedImage gbufferNormalImage;

	AllocatedImage gbufferPosObjectIdImageCpu;
	AllocatedImage gbufferNormalImageCpu;

	VkImageView gbufferPosObjectIdImageView;
	VkImageView gbufferNormalImageView;
	VkFramebuffer gbufferFrameBuffer;
	{
		VkExtent3D extent3D = {
			imageSize.width,
			imageSize.height,
			1
		};

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engine._engineData.color32Format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, extent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			vmaCreateImage(engine._engineData.allocator, &dimg_info, &dimg_allocinfo, &gbufferPosObjectIdImage._image, &gbufferPosObjectIdImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engine._engineData.color32Format, gbufferPosObjectIdImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engine._engineData.device, &imageViewInfo, nullptr, &gbufferPosObjectIdImageView));
		}

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engine._engineData.color32Format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, extent3D);
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			vmaCreateImage(engine._engineData.allocator, &dimg_info, &dimg_allocinfo, &gbufferNormalImage._image, &gbufferNormalImage._allocation, nullptr);

			VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(engine._engineData.color32Format, gbufferNormalImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
			VK_CHECK(vkCreateImageView(engine._engineData.device, &imageViewInfo, nullptr, &gbufferNormalImageView));
		}

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engine._engineData.color32Format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, extent3D);
			dimg_info.tiling = VK_IMAGE_TILING_LINEAR;
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
			vmaCreateImage(engine._engineData.allocator, &dimg_info, &dimg_allocinfo, &gbufferPosObjectIdImageCpu._image, &gbufferPosObjectIdImageCpu._allocation, nullptr);
		}

		{
			VkImageCreateInfo dimg_info = vkinit::image_create_info(engine._engineData.color32Format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, extent3D);
			dimg_info.tiling = VK_IMAGE_TILING_LINEAR;
			VmaAllocationCreateInfo dimg_allocinfo = {};
			dimg_allocinfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
			vmaCreateImage(engine._engineData.allocator, &dimg_info, &dimg_allocinfo, &gbufferNormalImageCpu._image, &gbufferNormalImageCpu._allocation, nullptr);
		}

		VkImageView attachments[2] = { gbufferPosObjectIdImageView, gbufferNormalImageView };

		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(gbufferRenderPass, imageSize);
		fb_info.pAttachments = attachments;
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(engine._engineData.device, &fb_info, nullptr, &gbufferFrameBuffer));
	}
	
	//Pipeline
	VkPipeline gbufferPipeline;
	VkPipelineLayout gbufferPipelineLayout;
	{
		VkShaderModule gbufferVertShader;
		if (!vkutils::load_shader_module(engine._engineData.device, "../../shaders/precalculate_receivers.vert.spv", &gbufferVertShader))
		{
			assert("G Buffer Vertex Shader Loading Issue");
		}

		VkShaderModule gbufferFragShader;
		if (!vkutils::load_shader_module(engine._engineData.device, "../../shaders/precalculate_receivers.frag.spv", &gbufferFragShader))
		{
			assert("F Buffer Fragment Shader Loading Issue");
		}

		VkDescriptorSetLayout setLayouts[] = { engine._sceneDescriptors.globalSetLayout, engine._sceneDescriptors.objectSetLayout };
		VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(setLayouts, 2);
		VK_CHECK(vkCreatePipelineLayout(engine._engineData.device, &pipeline_layout_info, nullptr, &gbufferPipelineLayout));

		//build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
		PipelineBuilder pipelineBuilder;

		//vertex input controls how to read vertices from vertex buffers. We aren't using it yet
		pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

		//input assembly is the configuration for drawing triangle lists, strips, or individual points.
		//we are just going to draw triangle list
		pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		//configure the rasterizer to draw filled triangles
		pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterStateCI{};
		conservativeRasterStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
		conservativeRasterStateCI.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
		conservativeRasterStateCI.extraPrimitiveOverestimationSize = 1.0;

		pipelineBuilder._rasterizer.pNext = &conservativeRasterStateCI;

		//we don't use multisampling, so just run the default one
		pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

		//a single blend attachment with no blending and writing to RGBA

		VkPipelineColorBlendAttachmentState blendAttachmentState[2] = {
			vkinit::color_blend_attachment_state(),
			vkinit::color_blend_attachment_state(),
		};

		pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(2, blendAttachmentState);

		//build the mesh pipeline
		VertexInputDescription vertexDescription = Vertex::get_vertex_description();
		pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
		pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

		pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
		pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

		pipelineBuilder._shaderStages.push_back(
			vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, gbufferVertShader));

		pipelineBuilder._shaderStages.push_back(
			vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, gbufferFragShader));

		pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(false, false, VK_COMPARE_OP_LESS_OR_EQUAL);

		pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;
		pipelineBuilder._pipelineLayout = gbufferPipelineLayout;

		//build the mesh triangle pipeline
		gbufferPipeline = pipelineBuilder.build_pipeline(engine._engineData.device, gbufferRenderPass);

		vkDestroyShaderModule(engine._engineData.device, gbufferVertShader, nullptr);
		vkDestroyShaderModule(engine._engineData.device, gbufferFragShader, nullptr);
	}

	//Rendering
	{
		engine._camData.lightmapInputSize = { (float)scene.lightmap_width, (float)scene.lightmap_height };
		vkutils::cpu_to_gpu(engine._engineData.allocator, engine._cameraBuffer, &engine._camData, sizeof(GPUCameraData));

		void* objectData;
		vmaMapMemory(engine._engineData.allocator, engine._objectBuffer._allocation, &objectData);
		GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

		for (int i = 0; i < engine.gltf_scene.nodes.size(); i++)
		{
			auto& mesh = engine.gltf_scene.prim_meshes[engine.gltf_scene.nodes[i].prim_mesh];
			objectSSBO[i].model = engine.gltf_scene.nodes[i].world_matrix;
			objectSSBO[i].material_id = mesh.material_idx;
		}
		vmaUnmapMemory(engine._engineData.allocator, engine._objectBuffer._allocation);

		vkutils::immediate_submit(&engine._engineData, [&](VkCommandBuffer cmd) {
			VkClearValue clearValues[2];
			clearValues[0].color = { { 0.0f, 0.0f, 0.0f, -1.0f } };
			clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

			VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(gbufferRenderPass, imageSize, gbufferFrameBuffer);
			rpInfo.clearValueCount = 2;
			rpInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkutils::cmd_viewport_scissor(cmd, imageSize);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipelineLayout, 0, 1, &engine._sceneDescriptors.globalDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipelineLayout, 1, 1, &engine._sceneDescriptors.objectDescriptor, 0, nullptr);

			engine.draw_objects(cmd);

			vkCmdEndRenderPass(cmd);
		});
	}

	//Copy images
	vkutils::immediate_submit(&engine._engineData, [&](VkCommandBuffer cmd) {
		{
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.image = gbufferPosObjectIdImageCpu._image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = 0;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);
		}

		// Issue the copy command
		VkImageCopy imageCopyRegion{};
		imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopyRegion.srcSubresource.layerCount = 1;
		imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopyRegion.dstSubresource.layerCount = 1;
		imageCopyRegion.extent.width = imageSize.width;
		imageCopyRegion.extent.height = imageSize.height;
		imageCopyRegion.extent.depth = 1;

		vkCmdCopyImage(
			cmd,
			gbufferPosObjectIdImage._image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			gbufferPosObjectIdImageCpu._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&imageCopyRegion);

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on
		{
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = gbufferPosObjectIdImageCpu._image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);
		}
	});

	vkutils::immediate_submit(&engine._engineData, [&](VkCommandBuffer cmd) {
		{
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.image = gbufferNormalImageCpu._image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = 0;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);
		}

		// Issue the copy command
		VkImageCopy imageCopyRegion{};
		imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopyRegion.srcSubresource.layerCount = 1;
		imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopyRegion.dstSubresource.layerCount = 1;
		imageCopyRegion.extent.width = imageSize.width;
		imageCopyRegion.extent.height = imageSize.height;
		imageCopyRegion.extent.depth = 1;

		vkCmdCopyImage(
			cmd,
			gbufferNormalImage._image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			gbufferNormalImageCpu._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&imageCopyRegion);

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on
		{
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = gbufferNormalImageCpu._image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);
		}
		});

	float* posObjectId = new float[imageSize.width * imageSize.height * 4];
	float* normal = new float[imageSize.width * imageSize.height * 4];

	{
		VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
		VkSubresourceLayout subResourceLayout;
		vkGetImageSubresourceLayout(engine._engineData.device, gbufferPosObjectIdImageCpu._image, &subResource, &subResourceLayout);

		const char* data;
		vmaMapMemory(engine._engineData.allocator, gbufferPosObjectIdImageCpu._allocation, (void**)&data);
		data += subResourceLayout.offset;

		memcpy(posObjectId, data, imageSize.width* imageSize.height * 4 * sizeof(float));

		vmaUnmapMemory(engine._engineData.allocator, gbufferPosObjectIdImageCpu._allocation);
	}

	{
		VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
		VkSubresourceLayout subResourceLayout;
		vkGetImageSubresourceLayout(engine._engineData.device, gbufferNormalImageCpu._image, &subResource, &subResourceLayout);
	
		const char* data;
		vmaMapMemory(engine._engineData.allocator, gbufferNormalImageCpu._allocation, (void**)&data);
		data += subResourceLayout.offset;
	
		memcpy(normal, data, imageSize.width * imageSize.height * 4 * sizeof(float));
	
		vmaUnmapMemory(engine._engineData.allocator, gbufferNormalImageCpu._allocation);
	}
	printf("----------\n");
	for (int i = 0; i < lightmapResolution; i++) {
		for (int j = 0; j < lightmapResolution; j++) {
			int receiverId = j + i * lightmapResolution;
			if (posObjectId[(receiverId) * 4 + 3] >= 0) {
				_receivers[receiverId].exists = true;
				_receivers[receiverId].uv = glm::ivec2(j, i);
				_receivers[receiverId].position = { posObjectId[(receiverId) * 4 + 0], posObjectId[(receiverId) * 4 + 1], posObjectId[(receiverId) * 4 + 2] };
				_receivers[receiverId].normal = { normal[(receiverId) * 4 + 0], normal[(receiverId) * 4 + 1], normal[(receiverId) * 4 + 2] };
				_receivers[receiverId].objectId = posObjectId[(receiverId) * 4 + 3];
				_receivers[receiverId].dPos = normal[(receiverId) * 4 + 3];
			}
		}
	}

	char* image = new char[lightmapResolution * lightmapResolution];
	memset(image, 0, lightmapResolution* lightmapResolution);
	for (int i = 0; i < lightmapResolution; i++) {
		for (int j = 0; j < lightmapResolution; j++) {
			if(posObjectId[(j + i * lightmapResolution) * 4 + 3] >= 0)
			image[j + i * lightmapResolution] = 255;
		}
	}

	FILE* ptr;
	fopen_s(&ptr, "../../precomputation/receiver_image.bin", "wb");
	fwrite(image, lightmapResolution* lightmapResolution, 1, ptr);
	fclose(ptr);
	printf("Created receivers: %d!\n", receiverCounter);

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
	vkCreateDescriptorSetLayout(engine._engineData.device, &setinfo, nullptr, &rtDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engine._engineData.descriptorPool, &rtDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engine._engineData.device, &allocateInfo, &rtDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &engine._vulkanRaytracing.tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = rtDescriptorSet;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	writes.emplace_back(accelerationStructureWrite);

	AllocatedBuffer sceneDescBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(GPUSceneDesc), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, rtDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));

	AllocatedBuffer meshInfoBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(GPUMeshInfo) * engine.gltf_scene.prim_meshes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));
	
	AllocatedBuffer probeLocationsBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(glm::vec4) * probes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &probeLocationsBuffer._descriptorBufferInfo, 3));

	AllocatedBuffer outputBuffer = vkutils::create_buffer(engine._engineData.allocator, rays * probes.size() * sizeof(GPUProbeRaycastResult), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &outputBuffer._descriptorBufferInfo, 4));

	vkUpdateDescriptorSets(engine._engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vkinit::pipeline_layout_create_info(&rtDescriptorSetLayout, 1);

	engine._vulkanRaytracing.create_new_pipeline(rtPipeline, pipelineLayoutCreateInfo,
		"../../shaders/precalculate_probe_rt.rgen.spv",
		"../../shaders/precalculate_probe_rt.rmiss.spv",
		"../../shaders/precalculate_probe_rt.rchit.spv");

	{
		GPUSceneDesc desc = {};
		VkBufferDeviceAddressInfo info = { };
		info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

		info.buffer = engine.vertex_buffer._buffer;
		desc.vertexAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.normal_buffer._buffer;
		desc.normalAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.tex_buffer._buffer;
		desc.uvAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.index_buffer._buffer;
		desc.indexAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.lightmap_tex_buffer._buffer;
		desc.lightmapUvAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		vkutils::cpu_to_gpu(engine._engineData.allocator, sceneDescBuffer, &desc, sizeof(GPUSceneDesc));
	}

	{
		vkutils::cpu_to_gpu(engine._engineData.allocator, probeLocationsBuffer, probes.data(), sizeof(glm::vec4) * probes.size());
	}

	{
		void* data;
		vmaMapMemory(engine._engineData.allocator, meshInfoBuffer._allocation, &data);
		GPUMeshInfo* dataMesh = (GPUMeshInfo*) data;
		for (int i = 0; i < engine.gltf_scene.prim_meshes.size(); i++) {
			dataMesh[i].indexOffset = engine.gltf_scene.prim_meshes[i].first_idx;
			dataMesh[i].vertexOffset = engine.gltf_scene.prim_meshes[i].vtx_offset;
			dataMesh[i].materialIndex = engine.gltf_scene.prim_meshes[i].material_idx;
		}
		vmaUnmapMemory(engine._engineData.allocator, meshInfoBuffer._allocation);
	}

	{
		VkCommandBuffer cmdBuf = vkutils::create_command_buffer(engine._engineData.device, engine._vulkanRaytracing._raytracingContext.commandPool, true);
		std::vector<VkDescriptorSet> descSets{ rtDescriptorSet };
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipeline);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipelineLayout, 0,
			(uint32_t)descSets.size(), descSets.data(), 0, nullptr);
		vkCmdTraceRaysKHR(cmdBuf, &rtPipeline.rgenRegion, &rtPipeline.missRegion, &rtPipeline.hitRegion, &rtPipeline.callRegion, rays, probes.size(), 1);
		vkutils::submit_and_free_command_buffer(engine._engineData.device, engine._vulkanRaytracing._raytracingContext.commandPool, cmdBuf, engine._vulkanRaytracing._queue, engine._vulkanRaytracing._raytracingContext.fence);
	}

	void* mappedOutputData;
	vmaMapMemory(engine._engineData.allocator, outputBuffer._allocation, &mappedOutputData);
	//printf("Ray cast result: %d\n", ((ProbeRaycastResult*)data)[i].objectId);
	memcpy(probeRaycastResult, mappedOutputData, rays* probes.size() * sizeof(GPUProbeRaycastResult));
	vmaUnmapMemory(engine._engineData.allocator, outputBuffer._allocation);
	
	int shCoeff = SPHERICAL_HARMONICS_NUM_COEFF(sphericalHarmonicsOrder);

	for (int i = 0; i < rays; i++) {
		calcY(&probeRaycastBasisFunctions[i * shCoeff], normalize(glm::vec3(probeRaycastResult[i].direction)), sphericalHarmonicsOrder);

		for (int l = 0; l < shCoeff; l++) {
			probeRaycastBasisFunctions[i * shCoeff + l] *= 4 * M_PI / rays;
		}
	}

	engine._vulkanRaytracing.destroy_raytracing_pipeline(rtPipeline);

	vmaDestroyBuffer(engine._engineData.allocator, sceneDescBuffer._buffer, sceneDescBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, meshInfoBuffer._buffer, meshInfoBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, probeLocationsBuffer._buffer, probeLocationsBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, outputBuffer._buffer, outputBuffer._allocation);
	vkDestroyDescriptorSetLayout(engine._engineData.device, rtDescriptorSetLayout, nullptr);
}

void Precalculation::receiver_raycast(VulkanEngine& engine, std::vector<AABB>& aabbClusters, std::vector<glm::vec4>& probes, int rays, float radius, int sphericalHarmonicsOrder, int clusterCoefficientCount, int maxReceivers, int totalReceiverCount, int maxProbesPerCluster, float* clusterProjectionMatrices, float* receiverCoefficientMatrices, float* receiverProbeWeightData, int* clusterProbes)
{
	printf("About to start receiver raycasting...\n");

	int shNumCoeff = SPHERICAL_HARMONICS_NUM_COEFF(sphericalHarmonicsOrder);
	int maxReceiverInABatch = MIN(64, maxReceivers);

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
	VkDescriptorSetLayoutBinding weightBuffer = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 6);
	VkDescriptorSetLayoutBinding clusterProbesBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 7);
	VkDescriptorSetLayoutBinding bindings[8] = { tlasBind, sceneDescBind, meshInfoBind, probeLocationsBind, receiverLocationsBind, outBuffer, weightBuffer, clusterProbesBind };
	VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings, 8);
	vkCreateDescriptorSetLayout(engine._engineData.device, &setinfo, nullptr, &rtDescriptorSetLayout);

	VkDescriptorSetAllocateInfo allocateInfo =
		vkinit::descriptorset_allocate_info(engine._engineData.descriptorPool, &rtDescriptorSetLayout, 1);

	vkAllocateDescriptorSets(engine._engineData.device, &allocateInfo, &rtDescriptorSet);

	std::vector<VkWriteDescriptorSet> writes;

	VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
	descASInfo.accelerationStructureCount = 1;
	descASInfo.pAccelerationStructures = &engine._vulkanRaytracing.tlas.accel;
	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	// The specialized acceleration structure descriptor has to be chained
	accelerationStructureWrite.pNext = &descASInfo;
	accelerationStructureWrite.dstSet = rtDescriptorSet;
	accelerationStructureWrite.dstBinding = 0;
	accelerationStructureWrite.descriptorCount = 1;
	accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	writes.emplace_back(accelerationStructureWrite);

	AllocatedBuffer sceneDescBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(GPUSceneDesc), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, rtDescriptorSet, &sceneDescBuffer._descriptorBufferInfo, 1));

	AllocatedBuffer meshInfoBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(GPUMeshInfo) * engine.gltf_scene.prim_meshes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &meshInfoBuffer._descriptorBufferInfo, 2));

	AllocatedBuffer probeLocationsBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(glm::vec4) * probes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &probeLocationsBuffer._descriptorBufferInfo, 3));

	AllocatedBuffer receiverLocationsBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(GPUReceiverData) * maxReceivers * (TEXEL_SAMPLES * TEXEL_SAMPLES), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &receiverLocationsBuffer._descriptorBufferInfo, 4));

	AllocatedBuffer outputBuffer = vkutils::create_buffer(engine._engineData.allocator, maxReceiverInABatch * rays * maxProbesPerCluster * sizeof(GPUReceiverRaycastResult), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &outputBuffer._descriptorBufferInfo, 5));
	
	AllocatedBuffer receiverProbeWeights = vkutils::create_upload_buffer(&engine._engineData, receiverProbeWeightData, sizeof(float) * (maxProbesPerCluster * totalReceiverCount), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &receiverProbeWeights._descriptorBufferInfo, 6));

	AllocatedBuffer clusterProbesBuffer = vkutils::create_buffer(engine._engineData.allocator, sizeof(int) * maxProbesPerCluster, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	writes.emplace_back(vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, rtDescriptorSet, &clusterProbesBuffer._descriptorBufferInfo, 7));

	PrecalculateReceiverMatrixConfig matrixConfig = {};
	matrixConfig.basisFunctionCount = shNumCoeff;
	matrixConfig.rayCount = rays;
	matrixConfig.totalProbeCount = probes.size();
	matrixConfig.maxProbesPerCluster = maxProbesPerCluster;

	AllocatedBuffer receiverMatrixConfig = vkutils::create_upload_buffer(&engine._engineData, &matrixConfig, sizeof(PrecalculateReceiverMatrixConfig), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	AllocatedBuffer matrixBuffer = vkutils::create_buffer(engine._engineData.allocator, maxReceiverInABatch * maxProbesPerCluster * shNumCoeff * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	AllocatedBuffer matrixBufferCPU = vkutils::create_buffer(engine._engineData.allocator, maxReceiverInABatch * maxProbesPerCluster * shNumCoeff * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	vkUpdateDescriptorSets(engine._engineData.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vkinit::pipeline_layout_create_info(&rtDescriptorSetLayout, 1);

	VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_RAYGEN_BIT_KHR , 0, sizeof(int) * 3 };
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;

	engine._vulkanRaytracing.create_new_pipeline(rtPipeline, pipelineLayoutCreateInfo,
		"../../shaders/precalculate_receiver_rt.rgen.spv",
		"../../shaders/precalculate_probe_rt.rmiss.spv",
		"../../shaders/precalculate_probe_rt.rchit.spv");

	ComputeInstance matrixCompute = {};
	engine._vulkanCompute.add_buffer_binding(matrixCompute, ComputeBufferType::UNIFORM, receiverMatrixConfig);
	engine._vulkanCompute.add_buffer_binding(matrixCompute, ComputeBufferType::STORAGE, outputBuffer);
	engine._vulkanCompute.add_buffer_binding(matrixCompute, ComputeBufferType::STORAGE, receiverProbeWeights);
	engine._vulkanCompute.add_buffer_binding(matrixCompute, ComputeBufferType::STORAGE, clusterProbesBuffer);
	engine._vulkanCompute.add_buffer_binding(matrixCompute, ComputeBufferType::STORAGE, matrixBuffer);
	engine._vulkanCompute.build(matrixCompute, engine._engineData.descriptorPool, "../../shaders/precalculate_construct_receiver_matrix.comp.spv");

	{
		GPUSceneDesc desc = {};
		VkBufferDeviceAddressInfo info = { };
		info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

		info.buffer = engine.vertex_buffer._buffer;
		desc.vertexAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.normal_buffer._buffer;
		desc.normalAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.tex_buffer._buffer;
		desc.uvAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.index_buffer._buffer;
		desc.indexAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		info.buffer = engine.lightmap_tex_buffer._buffer;
		desc.lightmapUvAddress = vkGetBufferDeviceAddress(engine._engineData.device, &info);

		vkutils::cpu_to_gpu(engine._engineData.allocator, sceneDescBuffer, &desc, sizeof(GPUSceneDesc));
	}

	{
		vkutils::cpu_to_gpu(engine._engineData.allocator, probeLocationsBuffer, probes.data(), sizeof(glm::vec4) * probes.size());
	}

	{
		void* data;
		vmaMapMemory(engine._engineData.allocator, meshInfoBuffer._allocation, &data);
		GPUMeshInfo* dataMesh = (GPUMeshInfo*)data;
		for (int i = 0; i < engine.gltf_scene.prim_meshes.size(); i++) {
			dataMesh[i].indexOffset = engine.gltf_scene.prim_meshes[i].first_idx;
			dataMesh[i].vertexOffset = engine.gltf_scene.prim_meshes[i].vtx_offset;
			dataMesh[i].materialIndex = engine.gltf_scene.prim_meshes[i].material_idx;
		}
		vmaUnmapMemory(engine._engineData.allocator, meshInfoBuffer._allocation);
	}

	int nodeReceiverDataOffset = 0;
	int nodeClusterDataOffset = 0;
	int receiverOffset = 0;
	int probeOffset = 0;

	float totalError = 0;
	int validClusters = 0;
	float averageSquaredError = 0;

	for (int nodeIndex = 0; nodeIndex < aabbClusters.size(); nodeIndex++) {
		printf("Starting node: %d with max probes: %d and receivers: %d\n", nodeIndex, aabbClusters[nodeIndex].probeCount, aabbClusters[nodeIndex].receivers.size());
		auto start = std::chrono::system_clock::now();

		auto clusterMatrix = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>(aabbClusters[nodeIndex].receivers.size(), shNumCoeff * aabbClusters[nodeIndex].probeCount);
		clusterMatrix.fill(0);

		{
			void* data;
			vmaMapMemory(engine._engineData.allocator, receiverLocationsBuffer._allocation, &data);
			GPUReceiverData* dataReceiver = (GPUReceiverData*)data;
			for (int i = 0; i < aabbClusters[nodeIndex].receivers.size(); i++) {
				for (int j = 0; j < aabbClusters[nodeIndex].receivers[i].poses.size(); j++) {
					dataReceiver[i * (TEXEL_SAMPLES * TEXEL_SAMPLES) + j].pos = aabbClusters[nodeIndex].receivers[i].poses[j];
					dataReceiver[i * (TEXEL_SAMPLES * TEXEL_SAMPLES) + j].normal = aabbClusters[nodeIndex].receivers[i].norms[j];
					dataReceiver[i * (TEXEL_SAMPLES * TEXEL_SAMPLES) + j].objectId = aabbClusters[nodeIndex].receivers[i].objectId;
					dataReceiver[i * (TEXEL_SAMPLES * TEXEL_SAMPLES) + j].dPos = aabbClusters[nodeIndex].receivers[i].poses.size();
				}
			}
			
			vmaUnmapMemory(engine._engineData.allocator, receiverLocationsBuffer._allocation);
		}

		vkutils::cpu_to_gpu(engine._engineData.allocator, clusterProbesBuffer, &clusterProbes[probeOffset], sizeof(int) * aabbClusters[nodeIndex].probeCount);

		for (int i = 0; i < aabbClusters[nodeIndex].receivers.size(); i += maxReceiverInABatch) {
			int remaningSize = MIN(aabbClusters[nodeIndex].receivers.size() - i, maxReceiverInABatch);

			//printf("Batch size: %d\n I is %d\n", remaningSize, i);

			{
				matrixConfig.receiverOffset = receiverOffset;
				matrixConfig.batchOffset = i;
				matrixConfig.batchSize = remaningSize;
				matrixConfig.clusterProbeCount = aabbClusters[nodeIndex].probeCount;
				vkutils::cpu_to_gpu(engine._engineData.allocator, receiverMatrixConfig, &matrixConfig, sizeof(PrecalculateReceiverMatrixConfig));
				int probeCount = aabbClusters[nodeIndex].probeCount;
				VkCommandBuffer cmdBuf = vkutils::create_command_buffer(engine._engineData.device, engine._vulkanRaytracing._raytracingContext.commandPool, true);
				std::vector<VkDescriptorSet> descSets{ rtDescriptorSet };
				vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipeline);
				vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipelineLayout, 0,
					(uint32_t)descSets.size(), descSets.data(), 0, nullptr);

				int pushConstantVariables[3] = { aabbClusters[nodeIndex].probeCount, i, receiverOffset };

				vkCmdPushConstants(cmdBuf, rtPipeline.pipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0,
					sizeof(int) * 3, pushConstantVariables);
				vkCmdTraceRaysKHR(cmdBuf, &rtPipeline.rgenRegion, &rtPipeline.missRegion, &rtPipeline.hitRegion, &rtPipeline.callRegion, maxReceiverInABatch, rays, 1);
				{
					VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
					memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
					memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

					vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
				}
				//compute shader
				{
					int groupcount = ((probeCount * remaningSize) / 256) + 1;
					vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, matrixCompute.pipeline);
					vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, matrixCompute.pipelineLayout, 0, 1, &matrixCompute.descriptorSet, 0, nullptr);
					vkCmdDispatch(cmdBuf, groupcount, 1, 1);
				}

				{
					VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
					memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
					memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

					vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
						0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
				}

				VkBufferCopy cpy;
				cpy.size = maxReceiverInABatch * aabbClusters[nodeIndex].probeCount * shNumCoeff * sizeof(float);
				cpy.srcOffset = 0;
				cpy.dstOffset = 0;

				vkCmdCopyBuffer(cmdBuf, matrixBuffer._buffer, matrixBufferCPU._buffer, 1, &cpy);

				vkutils::submit_and_free_command_buffer(engine._engineData.device, engine._vulkanRaytracing._raytracingContext.commandPool, cmdBuf, engine._vulkanRaytracing._queue, engine._vulkanRaytracing._raytracingContext.fence);
			}

			void* mappedOutputData;
			vmaMapMemory(engine._engineData.allocator, matrixBufferCPU._allocation, &mappedOutputData);
			memcpy(clusterMatrix.data() + i * aabbClusters[nodeIndex].probeCount * shNumCoeff, mappedOutputData, remaningSize * aabbClusters[nodeIndex].probeCount * shNumCoeff * sizeof(float));
			vmaUnmapMemory(engine._engineData.allocator, matrixBufferCPU._allocation);
		}

		{
			//std::ofstream file("eigenmatrix" + std::to_string(nodeIndex) + ".csv");
			//const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
			//file << clusterMatrix.format(CSVFormat);
		}

		Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::FullPivHouseholderQRPreconditioner> svd(clusterMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		int nc = clusterCoefficientCount;

		{
			float total = 0.f;
			float realTotal = 0.f;
			for (int aa = 0; aa < nc; aa++) {
				if (aa < svd.singularValues().size()) {
					total += svd.singularValues()[aa];
				}
			}
			for (int aa = 0; aa < svd.singularValues().size(); aa++) {
				realTotal += svd.singularValues()[aa];
			}

			float error = 1.0f - sqrt(total / realTotal);

			if (!isnan(error)) {
				totalError += aabbClusters[nodeIndex].receivers.size() * error;

				if (error > 0.0000001) {
					validClusters += 1;
				}
			}

			printf("Singular Value error: %f (total singular value count: %d)\n", error, svd.singularValues().size());
		}

		auto receiverReconstructionCoefficientMatrix = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>(svd.matrixU().rows(), nc);
		receiverReconstructionCoefficientMatrix.fill(0);
		receiverReconstructionCoefficientMatrix.block(0, 0, svd.matrixU().rows(), MIN(nc, svd.matrixU().cols())) = svd.matrixU().block(0, 0, svd.matrixU().rows(), MIN(nc, svd.matrixU().cols()));

		Eigen::MatrixXf singularMatrix(nc, svd.matrixV().cols());
		int singularValueSize = svd.singularValues().size();
		singularMatrix.setZero();
		for (int aa = 0; aa < nc; aa++) {
			if (aa < singularValueSize) {
				singularMatrix(aa, aa) = svd.singularValues()[aa];
			}
			else if(aa < svd.matrixV().cols()) {
				singularMatrix(aa, aa) = 0;
			}
		}
		auto clusterProjectionMatrix = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>(singularMatrix * svd.matrixV().transpose());

		//printf("Singular matrix size: %d x %d\n", singularMatrix.rows(), singularMatrix.cols());
		//printf("V matrix size: %d x %d\n", svd.matrixV().rows(), svd.matrixV().cols());

		auto newMatrix = receiverReconstructionCoefficientMatrix * clusterProjectionMatrix;
		{
			//std::ofstream file("eigenmatrix_pca" + std::to_string(nodeIndex) + ".csv");
			//const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
			//file << newMatrix.format(CSVFormat);
		}
		
		auto diff = clusterMatrix - newMatrix;
		float squaredError = diff.squaredNorm();
		averageSquaredError += squaredError * aabbClusters[nodeIndex].receivers.size();
		printf("Squared Error: %f\n", squaredError);
		
		/*
		int globalcounterr = 0;
		for (int y = 0; y < clusterProjectionMatrix.rows(); y++) {
			for (int x = 0; x < clusterProjectionMatrix.cols(); x++) {
				printf("%f vs %f\n", clusterProjectionMatrix(y, x), *(clusterProjectionMatrix.data() + globalcounterr));
				globalcounterr++;
			}
		}
		*/

		memcpy(clusterProjectionMatrices + nodeClusterDataOffset, clusterProjectionMatrix.data(), clusterProjectionMatrix.size() * sizeof(float));
		nodeClusterDataOffset += clusterProjectionMatrix.size();

		memcpy(receiverCoefficientMatrices + nodeReceiverDataOffset, receiverReconstructionCoefficientMatrix.data(), receiverReconstructionCoefficientMatrix.size() * sizeof(float));
		nodeReceiverDataOffset += receiverReconstructionCoefficientMatrix.size();

		receiverOffset += aabbClusters[nodeIndex].receivers.size();
		probeOffset += aabbClusters[nodeIndex].probeCount;

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		printf("Node tracing done %d/%d (took %f s)\n\n", nodeIndex, aabbClusters.size(), elapsed_seconds.count());
	}

	totalError /= totalReceiverCount;
	printf("Average singular value error: %f\n", totalError);
	printf("Average squared error: %f\n", averageSquaredError / totalReceiverCount);

	engine._vulkanRaytracing.destroy_raytracing_pipeline(rtPipeline);

	vmaDestroyBuffer(engine._engineData.allocator, sceneDescBuffer._buffer, sceneDescBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, meshInfoBuffer._buffer, meshInfoBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, probeLocationsBuffer._buffer, probeLocationsBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, receiverLocationsBuffer._buffer, probeLocationsBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, outputBuffer._buffer, outputBuffer._allocation);

	vmaDestroyBuffer(engine._engineData.allocator, receiverMatrixConfig._buffer, receiverMatrixConfig._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, receiverProbeWeights._buffer, receiverProbeWeights._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, matrixBuffer._buffer, matrixBuffer._allocation);
	vmaDestroyBuffer(engine._engineData.allocator, matrixBufferCPU._buffer, matrixBufferCPU._allocation);

	vkDestroyDescriptorSetLayout(engine._engineData.device, rtDescriptorSetLayout, nullptr);
}
