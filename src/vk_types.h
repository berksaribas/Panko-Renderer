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
    VkFence fence;
    VkCommandPool commandPool;
};

struct EngineData {
    VkDevice device;

    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;

    VkQueue computeQueue;
    uint32_t computeQueueFamily;

    CommandContext uploadContext;

    VmaAllocator allocator;
    VkDescriptorPool descriptorPool;

    VkRenderPass colorDepthRenderPass;
    VkRenderPass colorRenderPass;

    VkSampler linearSampler;
    VkSampler nearestSampler;

    VkFormat color32Format = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkFormat color16Format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat depth32Format = VK_FORMAT_D32_SFLOAT;
};

struct SceneDescriptors {
    VkDescriptorSet globalDescriptor;
    VkDescriptorSet objectDescriptor;
    VkDescriptorSet materialDescriptor;
    VkDescriptorSet textureDescriptor;

    VkDescriptorSetLayout globalSetLayout; //vertex, fragment, compute, ray
    VkDescriptorSetLayout objectSetLayout; //vertex, fragment, compute, ray
    VkDescriptorSetLayout materialSetLayout; //vertex, fragment, compute, ray
    VkDescriptorSetLayout textureSetLayout; //fragment, compute, ray

    VkDescriptorSetLayout singleImageSetLayout; //fragment, compute, ray
};

//we will add our main reusable types here