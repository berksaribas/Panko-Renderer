#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "memory/handle.h"

const VkFormat COLOR_32_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;
const VkFormat COLOR_16_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
const VkFormat COLOR_8_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
const VkFormat DEPTH_32_FORMAT = VK_FORMAT_D32_SFLOAT;

namespace Vrg {
    class RenderGraph;
    struct Bindable;
}

struct AllocatedBuffer {
    VkBuffer _buffer;
    VmaAllocation _allocation;
    VkDescriptorBufferInfo _descriptorBufferInfo;
};

struct AllocatedImage {
    VkImage _image;
    VmaAllocation _allocation;
    uint32_t mips = 1;
    VkFormat format;
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
    VkPhysicalDevice physicalDevice;
    VkInstance instance;

    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;

    VkQueue computeQueue;
    uint32_t computeQueueFamily;

    CommandContext uploadContext;

    VmaAllocator allocator;
    VkDescriptorPool descriptorPool;

    VkQueryPool queryPool;

    Vrg::RenderGraph* renderGraph;
};

struct SceneData {
    AllocatedBuffer vertexBuffer, indexBuffer, normalBuffer, texBuffer, lightmapTexBuffer, tangentBuffer;

    AllocatedBuffer sceneDescBuffer, meshInfoBuffer;
    AllocatedBuffer cameraBuffer, objectBuffer, materialBuffer;

    Handle<Vrg::Bindable> vertexBufferBinding;
    Handle<Vrg::Bindable> normalBufferBinding;
    Handle<Vrg::Bindable> texBufferBinding;
    Handle<Vrg::Bindable> lightmapTexBufferBinding;
    Handle<Vrg::Bindable> tangentBufferBinding;
    Handle<Vrg::Bindable> indexBufferBinding;
    Handle<Vrg::Bindable> cameraBufferBinding;
    Handle<Vrg::Bindable> objectBufferBinding;
    Handle<Vrg::Bindable> materialBufferBinding;

    VkDescriptorSet textureDescriptor;
    VkDescriptorSetLayout textureSetLayout;

    VkDescriptorSet raytracingDescriptor;
    VkDescriptorSetLayout raytracingSetLayout;
};