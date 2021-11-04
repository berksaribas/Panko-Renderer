#include "precalculation.h"
#include <triangle_box_intersection.h>
#include <omp.h>

#define VOXELIZER_IMPLEMENTATION
#include "triangle_box_intersection.h"
#include <fstream>
#include <optick.h>
#include <queue>

#define USE_COMPUTE_PROBE_DENSITY_CALCULATION 1

/*
	DONE
	- SCENE VOXELIZATION (DONE)
	- RECEIVER PLACEMENT (ALSO CREATE THE LIGHT MAP?)
	- PROBE PLACEMENT (HARDCODED)
	TODO
	- SETUP A RAYTRACING ENVIRONMENT (RADEON RAYS OR VULKAN RAYTRACING)
	- PROBE RAYCASTING (ALSO SUPPORT THIS IN REAL TIME VIA HW RT) -> OUTPUT TO A TEXTURE (THIS IS VISIBILITY RAYS)
	- RECEIVER COEFFICIENTS (THIS IS THE PART WE USE SPHERICAL HARMONICS)
	- RECEIVER CLUSTERING (PCA AND AABB)
*/
int partition(float arr[], int l, int r) {
	float x = arr[r];
	int i = l;
	for (int j = l; j <= r - 1; j++) {
		if (arr[j] <= x) {
			std::swap(arr[i], arr[j]);
			i++;
		}
	}
	std::swap(arr[i], arr[r]);
	return i;
}

float kthSmallest(float arr[], int l, int r, int k) {
	// If k is smaller than number of
	// elements in array
	if (k > 0 && k <= r - l + 1) {

		// Partition the array around last
		// element and get position of pivot
		// element in sorted array
		int index = partition(arr, l, r);

		// If position is same as k
		if (index - l == k - 1)
			return arr[index];

		// If position is more, recur
		// for left subarray
		if (index - l > k - 1)
			return kthSmallest(arr, l, index - 1, k);

		// Else recur for right subarray
		return kthSmallest(arr, index + 1, r,
			k - index + l - 1);
	}

	// If k is more than number of
	// elements in array
	return FLT_MAX;
}

//Christer Ericson's Real-Time Collision Detection 
static glm::vec3 calculate_barycentric(glm::vec2 p, glm::vec2 a, glm::vec2 b, glm::vec2 c) {
	glm::vec2 v0 = b - a, v1 = c - a, v2 = p - a;
	float d00 = glm::dot(v0, v0);
	float d01 = glm::dot(v0, v1);
	float d11 = glm::dot(v1, v1);
	float d20 = glm::dot(v2, v0);
	float d21 = glm::dot(v2, v1);
	float denom = d00 * d11 - d01 * d01;

	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
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

static float calculate_radius(Receiver* receivers, int receiverTextureSize, std::vector<glm::vec3>& probes, int overlaps) {
	OPTICK_EVENT();

	float* distanceData = new float[probes.size()];
	float radius = 0;
	int searchSize = receiverTextureSize * receiverTextureSize;
	int receiverCount = 0;
	for (int r = 0; r < searchSize; r += 1) {
		if (!receivers[r].exists) {
			continue;
		}
		for (int p = 0; p < probes.size(); p++) {
			distanceData[p] = glm::distance(probes[p], receivers[r].position);
		}
		radius += kthSmallest(distanceData, 0, probes.size(), overlaps);
		receiverCount++;
	}

	return radius / receiverCount;
}

uint8_t* Precalculation::voxelize(GltfScene& scene, float voxelSize, int padding, bool save)
{
	OPTICK_EVENT();

	int x = (scene.m_dimensions.max.x - scene.m_dimensions.min.x) / voxelSize + padding * 2;
	int y = (scene.m_dimensions.max.y - scene.m_dimensions.min.y) / voxelSize + padding * 2;
	int z = (scene.m_dimensions.max.z - scene.m_dimensions.min.z) / voxelSize + padding * 2;
	
	glm::vec3 halfSize = { voxelSize / 2.f, voxelSize / 2.f, voxelSize / 2.f };
	_dimX = x;
	_dimY = y;
	_dimZ = z;
	_padding = padding;
	_voxelSize = voxelSize;
	_scene = &scene;

	int size = x * y * z;
	int paddinglessSize = (x - padding * 2) * (y - padding * 2) * (z - padding * 2);
	
	//if (_voxelData != nullptr) {
	//	delete[] _voxelData;
	//}

	_voxelData = new uint8_t[size];
	memset(_voxelData, 0, size);

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
							_voxelData[index] = 1;
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
		_voxelData[index] = 2;
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
					if (_voxelData[ind] == 0) {
						fillQueue.push(neighbor);
						_voxelData[ind] = 2;
					}
					if (_voxelData[ind] == 1) {
						int currInd = curr.x + curr.y * x + curr.z * x * y;
						_voxelData[currInd] = 3;
					}
				}
			}
		}

		for (int k = 1; k < z - 1; k++) {
			for (int j = 1; j < y - 1; j++) {
				for (int i = 1; i < x - 1; i++) {
					int index = i + j * x + k * x * y;
					if (_voxelData[index] == 0) {
						_voxelData[index] = 1;
					}
				}
			}
		}
	}
	
	if (save) {
		std::ofstream file("mesh_voxelized_res.obj");
		int counter = 0;

		for (int k = 0; k < z; k++) {
			for (int j = 0; j < y; j++) {
				for (int i = 0; i < x; i++) {
					int index = i + j * x + k * x * y;

					if (_voxelData[index] != 1) {
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
		file.close();
	}
	
	return _voxelData;
}

std::vector<glm::vec4> Precalculation::place_probes(VulkanEngine& engine, int overlaps)
{
	OPTICK_EVENT();

	std::vector<glm::vec4> probes;

	for (int k = _padding; k < _dimZ - _padding; k++) {
		for (int j = _padding; j < _dimY - _padding; j++) {
			for (int i = _padding; i < _dimX - _padding; i++) {
				int index = i + j * _dimX + k * _dimX * _dimY;
				if (_voxelData[index] == 3) {
					probes.push_back({
						_scene->m_dimensions.min.x + _voxelSize * (i - _padding) + _voxelSize / 2.f,
						_scene->m_dimensions.min.y + _voxelSize * (j - _padding) + _voxelSize / 2.f,
						_scene->m_dimensions.min.z + _voxelSize * (k - _padding) + _voxelSize / 2.f,
						0.f
					});
				}
			}
		}
	}

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
	ComputeInstance instance = {};
	engine.vulkanCompute.add_buffer(instance, UNIFORM, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(GPUProbeDensityUniformData));
	engine.vulkanCompute.add_buffer(instance, STORAGE, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::vec4) * probes.size());
	engine.vulkanCompute.add_buffer(instance, STORAGE, VMA_MEMORY_USAGE_GPU_TO_CPU, sizeof(float) * probes.size());
	engine.vulkanCompute.build(instance, engine._descriptorPool, "../../shaders/precalculate_probe_density_weights.comp.spv");
	engine.cpu_to_gpu(instance.buffers[1], probes.data(), sizeof(glm::vec4) * probes.size());
#endif

	int targetProbeCount = _dimX - _padding * 2;
	printf("Targeted amount of probes is %d\n", targetProbeCount);

	float radius = 5;//calculate_radius(_receivers, _receiverTextureResolution, probes, overlaps);
	printf("Radius is %f\n", radius);

	float currMaxWeight = -1;
	int toRemoveIndex = -1;
	while (probes.size() > targetProbeCount) {
#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
		GPUProbeDensityUniformData ub = { probes.size(), radius };
		engine.cpu_to_gpu(instance.buffers[0], &ub, sizeof(GPUProbeDensityUniformData));
		int groupcount = ((probes.size()) / 256) + 1;
		engine.vulkanCompute.compute(instance, groupcount, 1, 1);
#endif
		for (int i = 0; i < probes.size(); i++) {
#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
			void* gpuWeightData;
			vmaMapMemory(engine._allocator, instance.buffers[2]._allocation, &gpuWeightData);
			float weight = ((float*)gpuWeightData)[i];
			vmaUnmapMemory(engine._allocator, instance.buffers[2]._allocation);
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
		vmaMapMemory(engine._allocator, instance.buffers[1]._allocation, &gpuProbeData);
		castedGpuProbeData = (glm::vec4*)gpuProbeData;
		castedGpuProbeData[toRemoveIndex] = castedGpuProbeData[probes.size() - 1];
		vmaUnmapMemory(engine._allocator, instance.buffers[1]._allocation);
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


	if (true) {
		std::ofstream file("mesh_voxelized_res_with_probes.obj");
		int counter = 0;

		for (int k = 0; k < _dimZ; k++) {
			for (int j = 0; j < _dimY; j++) {
				for (int i = 0; i < _dimX; i++) {
					int index = i + j * _dimX + k * _dimX * _dimY;

					if (_voxelData[index] != 1) {
						continue;
					}

					float voxelX = _scene->m_dimensions.min.x + _voxelSize * (i - _padding);
					float voxelY = _scene->m_dimensions.min.y + _voxelSize * (j - _padding);
					float voxelZ = _scene->m_dimensions.min.z + _voxelSize * (k - _padding);

					//Vertices
					file << "v " << voxelX << " " << voxelY << " " << voxelZ << "\n";
					file << "v " << voxelX << " " << voxelY << " " << voxelZ + _voxelSize << "\n";
					file << "v " << voxelX + _voxelSize << " " << voxelY << " " << voxelZ + _voxelSize << "\n";
					file << "v " << voxelX + +_voxelSize << " " << voxelY << " " << voxelZ << "\n";
					file << "v " << voxelX << " " << voxelY + _voxelSize << " " << voxelZ << "\n";
					file << "v " << voxelX << " " << voxelY + _voxelSize << " " << voxelZ + _voxelSize << "\n";
					file << "v " << voxelX + _voxelSize << " " << voxelY + _voxelSize << " " << voxelZ + _voxelSize << "\n";
					file << "v " << voxelX + +_voxelSize << " " << voxelY + _voxelSize << " " << voxelZ << "\n";

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

		for (int i = 0; i < probes.size(); i++) {
			float voxelX = probes[i].x;
			float voxelY = probes[i].y;
			float voxelZ = probes[i].z;

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

#if USE_COMPUTE_PROBE_DENSITY_CALCULATION
	engine.vulkanCompute.destroy_compute_instance(instance);
#endif

	return probes;
}

Receiver* Precalculation::generate_receivers(int objectResolution)
{
	OPTICK_EVENT();

	_receiverObjectResolution = objectResolution;

	int texSep = sqrt(_scene->nodes.size()) + 1;
	_receiverTextureResolution = objectResolution * texSep;

	int receiverCount = _receiverTextureResolution * _receiverTextureResolution;
	_receivers = new Receiver[receiverCount];
	memset(_receivers, 0, sizeof(Receiver) * receiverCount);
	
	printf("Creating a total of %d receivers for texture resolution of %d\n", receiverCount, _receiverTextureResolution);

	int currTexX = 0, currTexY = 0;

	for (int nodeIndex = 0; nodeIndex < _scene->nodes.size(); nodeIndex++) {
		auto& mesh = _scene->prim_meshes[_scene->nodes[nodeIndex].prim_mesh];
			for (int triangle = 0; triangle < mesh.idx_count; triangle += 3) {
				glm::vec3 worldVertices[3] = {};
				glm::vec3 worldNormals[3] = {};
				glm::vec2 texVertices[3] = {};
				int minX = objectResolution, minY = objectResolution;
				int maxX = 0, maxY = 0;
				for (int i = 0; i < 3; i++) {
					int vertexIndex = mesh.vtx_offset + _scene->indices[mesh.first_idx + triangle + i];

					glm::vec4 vertex = _scene->nodes[nodeIndex].world_matrix * glm::vec4(_scene->positions[vertexIndex], 1.0);
					worldVertices[i] = glm::vec3(vertex / vertex.w);

					glm::vec4 normal = _scene->nodes[nodeIndex].world_matrix * glm::vec4(_scene->normals[vertexIndex], 1.0);
					worldNormals[i] = glm::vec3(normal / normal.w);

					texVertices[i] = _scene->texcoords0[vertexIndex] * (float)(objectResolution - 1);

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
				//Rasterize 2D triangle
				float area = edge_function(texVertices[0], texVertices[1], texVertices[2]);

				for (int j = minY; j <= maxY; j++) {
					for (int i = minX; i <= maxX; i++) {
						glm::vec2 pixelMiddle = { i + 0.5f, j + 0.5f };
						glm::vec3 barycentric = calculate_barycentric(pixelMiddle,
							texVertices[0], texVertices[1], texVertices[2]);
						if (barycentric.x >= 0 && barycentric.y >= 0 && barycentric.z >= 0) {
							Receiver receiver = {};
							receiver.position = apply_barycentric(barycentric, worldVertices[0], worldVertices[1], worldVertices[2]);
							receiver.normal = apply_barycentric(barycentric, worldNormals[0], worldNormals[1], worldNormals[2]);
							receiver.exists = true;
							int receiverIndex = currTexX * objectResolution + currTexY * _receiverTextureResolution * objectResolution + i + j * _receiverTextureResolution;
							if (!_receivers[receiverIndex].exists) {
								_receivers[receiverIndex] = receiver;
							}
						}
					}
				}
			}
		currTexX++;
		if (currTexX == texSep) {
			currTexX = 0;
			currTexY++;
		}
	}

	printf("Created receivers!\n");

	return _receivers;
}
