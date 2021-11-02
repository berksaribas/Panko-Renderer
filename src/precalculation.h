#include <vector>
#include <glm/glm.hpp>
#include <gltf_scene.hpp>

class Precalculation {
public:
	void voxelize(GltfScene& scene);
	std::vector<glm::vec3> place_probes(int overlaps);
private:

};
