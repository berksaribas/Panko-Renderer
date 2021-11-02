#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

struct VertexInputDescription {

    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex
{
    static VertexInputDescription get_vertex_description();
};

