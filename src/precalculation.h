#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <gltf_scene.hpp>
#include <vk_engine.h>

struct Receiver {
	glm::vec3 position;
	glm::vec3 normal;
	bool exists;
};

struct GPUProbeDensityUniformData {
	int probeCount;
	float radius;
};

class Precalculation {
public:
	uint8_t* voxelize(GltfScene& scene, float voxelSize, int padding, bool save = false);
	std::vector<glm::vec4> place_probes(VulkanEngine& engine, int overlaps);
	Receiver* generate_receivers(int objectResolution);
	void probe_raycast(VulkanEngine& engine, int rays);
	//void receiver_raycast(VulkanEngine& engine);
private:
	GltfScene* _scene;
	uint8_t* _voxelData;
	float _voxelSize;
	int _dimX, _dimY, _dimZ;
	int _padding;

	Receiver* _receivers;
	int _receiverTextureResolution;
	int _receiverObjectResolution;

	std::vector<glm::vec4> _probes;
};
