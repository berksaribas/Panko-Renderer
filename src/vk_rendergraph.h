#pragma once

#include <string>
#include "vk_rendergraph_types.h"
#include <vector>
#include <functional>
#include <initializer_list>
#include "vk_cache.h"
#include "memory.h"

namespace Vrg {
	class RenderGraph {
	public:
		RenderGraph(EngineData* _engineData);
		void add_render_pass(RenderPass renderPass);
		Bindable* register_image_view(AllocatedImage* image, ImageView imageView, std::string resourceName);
		Bindable* register_storage_buffer(AllocatedBuffer* buffer, std::string resourceName);
		Bindable* register_uniform_buffer(AllocatedBuffer* buffer, std::string resourceName);
		Bindable* register_vertex_buffer(AllocatedBuffer* buffer, VkFormat format, std::string resourceName);
		Bindable* register_index_buffer(AllocatedBuffer* buffer, VkFormat format, std::string resourceName);
		Bindable* get_resource(std::string resourceName); //TODO: Implement if needed. Right now, not needed.
		void compile();
		void execute(VkCommandBuffer cmd);
	private:
		VkPipeline get_pipeline(RenderPass& renderPass);
		VkPipelineLayout get_pipeline_layout(RenderPass& renderPass);
		VkDescriptorSet get_descriptor_set(RenderPass& renderPass, int set);
		VkDescriptorSetLayout get_descriptor_set_layout(RenderPass& renderPass, int set);
		VkImageView get_image_view(VkImage image, ImageView& imageView, VkFormat format, bool isDepth = false);

		VkImageLayout get_current_image_layout(VkImage image);

		//
		EngineData* engineData;
		FrameAllocator frameAllocator;

		//render passes
		std::vector<RenderPass> renderPasses;

		//bindings
		uint32_t bindingCount = 0;
		std::vector<Bindable> bindings;
		std::unordered_map<std::string, uint32_t> bindingNames;
		std::unordered_map<VkImage, ResourceAccessType> imageBindingAccessType;
		std::unordered_map<VkBuffer, ResourceAccessType> bufferBindingAccessType;
		std::unordered_map<VkImage, VkImageLayout> bindingImageLayout;

		//caches
		std::unordered_map<std::string, VkPipeline> pipelineCache;
		std::unordered_map<ImageViewCache, VkImageView, ImageViewCache_hash> imageViewCache;
		std::unordered_map<DescriptorSetCache, VkDescriptorSet, DescriptorSetCache_hash> descriptorSetCache;
		std::unordered_map<DescriptorSetLayoutCache, VkDescriptorSetLayout, DescriptorSetLayoutCache_hash> descriptorSetLayoutCache;
		std::unordered_map<PipelineLayoutCache, VkPipelineLayout, PipelineLayoutCache_hash> pipelineLayoutCache;

		//samplers
		VkSampler samplers[2];
	};
}