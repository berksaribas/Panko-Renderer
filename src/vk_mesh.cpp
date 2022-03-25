#include "vk_mesh.h"
#include <iostream>

VertexInputDescription Vertex::get_vertex_description()
{
	VertexInputDescription description;

	//we will have just 1 vertex buffer binding, with a per-vertex rate
	VkVertexInputBindingDescription vertex_binding = {};
	vertex_binding.binding = 0;
	vertex_binding.stride = sizeof(glm::vec3);
	vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputBindingDescription normal_binding = {};
	normal_binding.binding = 1;
	normal_binding.stride = sizeof(glm::vec3);
	normal_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputBindingDescription tex_binding = {};
	tex_binding.binding = 2;
	tex_binding.stride = sizeof(glm::vec2);
	tex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputBindingDescription lightmap_binding = {};
	lightmap_binding.binding = 3;
	lightmap_binding.stride = sizeof(glm::vec2);
	lightmap_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputBindingDescription tangent_binding = {};
	tangent_binding.binding = 4;
	tangent_binding.stride = sizeof(glm::vec4);
	tangent_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(vertex_binding);
	description.bindings.push_back(normal_binding);
	description.bindings.push_back(tex_binding);
	description.bindings.push_back(lightmap_binding);
	description.bindings.push_back(tangent_binding);

	//Position will be stored at Location 0
	VkVertexInputAttributeDescription positionAttribute = {};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = 0;

	//Normal will be stored at Location 1
	VkVertexInputAttributeDescription normalAttribute = {};
	normalAttribute.binding = 1;
	normalAttribute.location = 1;
	normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset = 0;

	//UV will be stored at Location 3
	VkVertexInputAttributeDescription uvAttribute = {};
	uvAttribute.binding = 2;
	uvAttribute.location = 2;
	uvAttribute.format = VK_FORMAT_R32G32_SFLOAT;
	uvAttribute.offset = 0;

	VkVertexInputAttributeDescription lightmapAttribute = {};
	lightmapAttribute.binding = 3;
	lightmapAttribute.location = 3;
	lightmapAttribute.format = VK_FORMAT_R32G32_SFLOAT;
	lightmapAttribute.offset = 0;

	VkVertexInputAttributeDescription tangentAttribute = {};
	tangentAttribute.binding = 4;
	tangentAttribute.location = 4;
	tangentAttribute.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	tangentAttribute.offset = 0;

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(uvAttribute);
	description.attributes.push_back(lightmapAttribute);
	description.attributes.push_back(tangentAttribute);
	return description;
}