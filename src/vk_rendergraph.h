#pragma once

#include "memory/frame_allocator.h"
#include "vk_cache.h"
#include "vk_raytracing.h"
#include "vk_rendergraph_types.h"
#include "vk_timer.h"
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "memory/handle_pool.h"

namespace Vrg
{
class RenderGraph
{
public:
    RenderGraph(EngineData* _engineData);
    void enable_raytracing(VulkanRaytracing* _vulkanRaytracing);
    RenderPass* add_render_pass(RenderPass renderPass);
    Handle<Bindable> register_image_view(AllocatedImage* image, ImageView imageView,
                                         std::string resourceName);
    Handle<Bindable> register_storage_buffer(AllocatedBuffer* buffer,
                                             std::string resourceName);
    Handle<Bindable> register_uniform_buffer(AllocatedBuffer* buffer,
                                             std::string resourceName);
    Handle<Bindable> register_vertex_buffer(AllocatedBuffer* buffer, VkFormat format,
                                            std::string resourceName);
    Handle<Bindable> register_index_buffer(AllocatedBuffer* buffer, VkFormat format,
                                           std::string resourceName);
    Handle<Bindable> get_resource(
        std::string resourceName); // TODO: Implement if needed. Right now, not needed.

    void insert_barrier(VkCommandBuffer cmd, Handle<Bindable> bindable,
                        PipelineType pipelineType, bool isWrite, uint32_t mip = 0);
    void handle_render_pass_barriers(VkCommandBuffer cmd, RenderPass& renderPass);
    void bind_pipeline_and_descriptors(VkCommandBuffer cmd, RenderPass& renderPass);
    VkImageView get_image_view(VkImage image, ImageView& imageView, VkFormat format);
    VkImageLayout get_current_image_layout(VkImage image, uint32_t mip);
    void inform_current_image_layout(VkImage image, uint32_t mip, VkImageLayout layout);

    void execute(VkCommandBuffer cmd);
    void rebuild_pipelines();

    void destroy_resource(AllocatedImage& image);
    void destroy_resource(AllocatedBuffer& buffer);

    VulkanTimer vkTimer;
    Pool<Bindable> bindings;

private:
    // TODO: Remove render pass dependency and expose to public
    VkPipeline get_pipeline(RenderPass& renderPass);
    RaytracingPipeline* get_raytracing_pipeline(RenderPass& renderPass);
    VkPipelineLayout get_pipeline_layout(RenderPass& renderPass);
    VkDescriptorSet get_descriptor_set(RenderPass& renderPass, int set);
    VkDescriptorSetLayout get_descriptor_set_layout(RenderPass& renderPass, int set);

    //
    EngineData* engineData;
    VulkanRaytracing* vulkanRaytracing;
    FrameAllocator frameAllocator;

    // render passes
    std::vector<RenderPass> renderPasses;

    // bindings
    uint32_t bindingCount = 0;
    std::unordered_map<VkBuffer, ResourceAccessType> bufferBindingAccessType;
    std::unordered_map<ImageMipCache, ResourceAccessType, ImageMipCache_hash>
        imageBindingAccessType;
    std::unordered_map<ImageMipCache, VkImageLayout, ImageMipCache_hash> bindingImageLayout;

    // caches
    std::unordered_map<std::string, VkPipeline> pipelineCache;
    std::unordered_map<std::string, RaytracingPipeline> raytracingPipelineCache;
    std::unordered_map<ImageViewCache, VkImageView, ImageViewCache_hash> imageViewCache;
    std::unordered_map<DescriptorSetCache, VkDescriptorSet, DescriptorSetCache_hash>
        descriptorSetCache;
    std::unordered_map<DescriptorSetLayoutCache, VkDescriptorSetLayout,
                       DescriptorSetLayoutCache_hash>
        descriptorSetLayoutCache;
    std::unordered_map<PipelineLayoutCache, VkPipelineLayout, PipelineLayoutCache_hash>
        pipelineLayoutCache;

    // samplers
    VkSampler samplers[2];
};
} // namespace Vrg