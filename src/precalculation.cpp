#include "precalculation.h"
#include <triangle_box_intersection.h>
#include <omp.h>

#define VOXELIZER_IMPLEMENTATION
#include "triangle_box_intersection.h"
#include <fstream>
#include <optick.h>
#include <queue>
/*
	TRY TO FINISH UNTIL 5TH OF NOVEMBER	
	- SCENE VOXELIZATION (DONE)

	- PROBE PLACEMENT (WIP)
	- PROBE RAYCASTING (ALSO SUPPORT THIS IN REAL TIME VIA HW RT) -> OUTPUT TO A TEXTURE (THIS IS VISIBILITY RAYS)
	- RECEIVER COEFFICIENTS (THIS IS THE PART WE USE SPHERICAL HARMONICS)
	- RECEIVER CLUSTERING (PCA AND AABB)
*/

static float calculate_density(float length, float radius) {
	float t = length / radius;
	if (t >= 0 && t <= 1) {
		float tSquared = t * t;
		return (2 * t * tSquared) - (3 * tSquared) + 1;
	}
	return 0;
}

static float calculate_spatial_weight(float radius, int probeIndex, std::vector<glm::vec3> probes) {
	float total_weight = 0.f;
	for (int i = 0; i < probes.size(); i++) {
		if (i != probeIndex) {
			total_weight += calculate_density(glm::distance(probes[i], probes[probeIndex]), radius);
		}
	}
	return total_weight;
}

void Precalculation::voxelize(GltfScene& scene)
{
	OPTICK_EVENT();

	float voxelSize = 0.1f; //10.0f for sponza
	glm::vec3 halfSize = { voxelSize / 2.f, voxelSize / 2.f, voxelSize / 2.f };
	int padding = 10;
	int x = (scene.m_dimensions.max.x - scene.m_dimensions.min.x) / voxelSize + padding * 2;	//10 padding from each side
	int y = (scene.m_dimensions.max.y - scene.m_dimensions.min.y) / voxelSize + padding * 2;	//10 padding from each side
	int z = (scene.m_dimensions.max.z - scene.m_dimensions.min.z) / voxelSize + padding * 2;	//10 padding from each side
	int size = x * y * z;
	int paddinglessSize = (x - padding * 2) * (y - padding * 2) * (z - padding * 2);
	
	uint8_t* grid = new uint8_t[size];
	memset(grid, 0, size);

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
						if (!triBoxOverlap({ scene.m_dimensions.min.x + (i - padding) * voxelSize + voxelSize / 2.f,
											scene.m_dimensions.min.y + (j - padding) * voxelSize + voxelSize / 2.f,
											scene.m_dimensions.min.z + (k - padding) * voxelSize + voxelSize / 2.f },
											halfSize, vertices[0], vertices[1], vertices[2])) {
							continue;
						}

						int index = i + j * x + k * x * y;
						if (index < size) {
							grid[index] = 1;
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
		grid[index] = 2;
		while (fillQueue.size() > 0)
		{
			glm::i32vec3 curr = fillQueue.front();
			fillQueue.pop();

			glm::i32vec3 neighbors[8] =
			{
				{curr.x - 1, curr.y, curr.z},
				{curr.x + 1, curr.y, curr.z},
				{curr.x, curr.y - 1, curr.z},
				{curr.x, curr.y + 1, curr.z},
				{curr.x, curr.y, curr.z - 1},
				{curr.x, curr.y, curr.z + 1},
			};

			for (int i = 0; i < 8; i++) {
				glm::i32vec3 neighbor = neighbors[i];
				if (neighbor.x >= 0 && neighbor.x < x && neighbor.y >= 0 && neighbor.y < y && neighbor.z >= 0 && neighbor.z < z) {
					int ind = neighbor.x + neighbor.y * x + neighbor.z * x * y;
					if (grid[ind] == 0) {
						fillQueue.push(neighbor);
						grid[ind] = 2;
					}
				}
			}
		}

		for (int k = 1; k < z - 1; k++) {
			for (int j = 1; j < y - 1; j++) {
				for (int i = 1; i < x - 1; i++) {
					int index = i + j * x + k * x * y;
					if (grid[index] == 0) {
						grid[index] = 1;
					}
				}
			}
		}
	}
	
	//return;
	std::ofstream file("mesh_voxelized_res.obj");
	int counter = 0;

	for (int k = 0; k < z; k++) {
		for (int j = 0; j < y; j++) {
			for (int i = 0; i < x; i++) {
				int index = i + j * x + k * x * y;

				if (grid[index] != 1) {
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
