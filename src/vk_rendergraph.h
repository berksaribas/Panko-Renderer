#pragma once

#include <string>
#include "vk_rendergraph_types.h"
#include <vector>
#include <functional>
#include <initializer_list>
#include "vk_cache.h"
#include "memory.h"
#include "vk_raytracing.h"
#include "vk_timer.h"

namespace Vrg {
	class RenderGraph {
	public:
		RenderGraph(EngineData* _engineData);
		void enable_raytracing(VulkanRaytracing* _vulkanRaytracing);
		void add_render_pass(RenderPass renderPass);
		Bindable* register_image_view(AllocatedImage* image, ImageView imageView, std::string resourceName);
		Bindable* register_storage_buffer(AllocatedBuffer* buffer, std::string resourceName);
		Bindable* register_uniform_buffer(AllocatedBuffer* buffer, std::string resourceName);
		Bindable* register_vertex_buffer(AllocatedBuffer* buffer, VkFormat format, std::string resourceName);
		Bindable* register_index_buffer(AllocatedBuffer* buffer, VkFormat format, std::string resourceName);
		Bindable* get_resource(std::string resourceName); //TODO: Implement if needed. Right now, not needed.
		void compile();
		
		void execute(VkCommandBuffer cmd);
		VulkanTimer vkTimer;
	private:
		VkPipeline get_pipeline(RenderPass& renderPass);
		RaytracingPipeline* get_raytracing_pipeline(RenderPass& renderPass);
		VkPipelineLayout get_pipeline_layout(RenderPass& renderPass);
		VkDescriptorSet get_descriptor_set(RenderPass& renderPass, int set);
		VkDescriptorSetLayout get_descriptor_set_layout(RenderPass& renderPass, int set);
		VkImageView get_image_view(VkImage image, ImageView& imageView, VkFormat format);
		void insert_barrier(VkCommandBuffer cmd, Vrg::Bindable* binding, PipelineType pipelineType, bool isWrite, uint32_t mip = 0);
		VkImageLayout get_current_image_layout(VkImage image, uint32_t mip);

		//
		EngineData* engineData;
		VulkanRaytracing* vulkanRaytracing;
		FrameAllocator frameAllocator;

		//render passes
		std::vector<RenderPass> renderPasses;

		//bindings
		uint32_t bindingCount = 0;
		std::vector<Bindable> bindings;
		std::unordered_map<std::string, uint32_t> bindingNames;
		std::unordered_map<VkBuffer, ResourceAccessType> bufferBindingAccessType;
		std::unordered_map<ImageMipCache, ResourceAccessType, ImageMipCache_hash> imageBindingAccessType;
		std::unordered_map<ImageMipCache, VkImageLayout, ImageMipCache_hash> bindingImageLayout;

		//caches
		std::unordered_map<std::string, VkPipeline> pipelineCache;
		std::unordered_map<std::string, RaytracingPipeline> raytracingPipelineCache;
		std::unordered_map<ImageViewCache, VkImageView, ImageViewCache_hash> imageViewCache;
		std::unordered_map<DescriptorSetCache, VkDescriptorSet, DescriptorSetCache_hash> descriptorSetCache;
		std::unordered_map<DescriptorSetLayoutCache, VkDescriptorSetLayout, DescriptorSetLayoutCache_hash> descriptorSetLayoutCache;
		std::unordered_map<PipelineLayoutCache, VkPipelineLayout, PipelineLayoutCache_hash> pipelineLayoutCache;

		//samplers
		VkSampler samplers[2];
	};
}