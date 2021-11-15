// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

struct AllocatedBuffer {
    VkBuffer _buffer;
    VmaAllocation _allocation;
    VkDescriptorBufferInfo _descriptorBufferInfo;
};

struct AllocatedImage {
    VkImage _image;
    VmaAllocation _allocation;
};

struct Texture {
    AllocatedImage image;
    VkImageView imageView;
};

struct CommandContext {
    VkFence _fence;
    VkCommandPool _commandPool;
};

//we will add our main reusable types here