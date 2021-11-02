// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

struct AllocatedBuffer {
    VkBuffer _buffer;
    VmaAllocation _allocation;
};

struct AllocatedImage {
    VkImage _image;
    VmaAllocation _allocation;
};

struct Texture {
    AllocatedImage image;
    VkImageView imageView;
};

struct BasicMaterial {
    glm::vec4 base_color;
    glm::vec3 emissive_color;
    float metallic_factor;
    float roughness_factor;
    int texture;
    int normal_texture;
    int metallic_roughness_texture;
};

//we will add our main reusable types here