﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

namespace vkinit
{

VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex,
                                                 VkCommandPoolCreateFlags flags = 0);

VkCommandBufferAllocateInfo command_buffer_allocate_info(
    VkCommandPool pool, uint32_t count = 1,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags = 0);

VkFramebufferCreateInfo framebuffer_create_info(VkRenderPass renderPass, VkExtent2D extent);

VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0);

VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags = 0);

VkSubmitInfo submit_info(VkCommandBuffer* cmd);

VkPresentInfoKHR present_info();

VkRenderPassBeginInfo renderpass_begin_info(VkRenderPass renderPass, VkExtent2D windowExtent,
                                            VkFramebuffer framebuffer);

VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                                                  VkShaderModule shaderModule);

VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info();

VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info(
    VkPrimitiveTopology topology);

VkPipelineRasterizationStateCreateInfo rasterization_state_create_info(
    VkPolygonMode polygonMode);

VkPipelineMultisampleStateCreateInfo multisampling_state_create_info();

VkPipelineColorBlendAttachmentState color_blend_attachment_state();

VkPipelineColorBlendStateCreateInfo color_blend_state_create_info(
    int attachmentCount, VkPipelineColorBlendAttachmentState* attachments);

VkPipelineLayoutCreateInfo pipeline_layout_create_info(VkDescriptorSetLayout* layouts,
                                                       int numLayouts);

VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags,
                                    VkExtent3D extent, uint32_t mipLevels = 1);

VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image,
                                            VkImageAspectFlags aspectFlags,
                                            uint32_t mipLevels = 1);

VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info(bool bDepthTest,
                                                                bool bDepthWrite,
                                                                VkCompareOp compareOp);

VkDescriptorSetLayoutCreateInfo descriptorset_layout_create_info(
    VkDescriptorSetLayoutBinding* bindings, int numBindings);

VkDescriptorSetAllocateInfo descriptorset_allocate_info(VkDescriptorPool descriptorPool,
                                                        VkDescriptorSetLayout* layouts,
                                                        int numLayouts);

VkDescriptorSetLayoutBinding descriptorset_layout_binding(VkDescriptorType type,
                                                          VkShaderStageFlags stageFlags,
                                                          uint32_t binding);

VkWriteDescriptorSet write_descriptor_buffer(VkDescriptorType type, VkDescriptorSet dstSet,
                                             VkDescriptorBufferInfo* bufferInfo,
                                             uint32_t binding);

VkSamplerCreateInfo sampler_create_info(
    VkFilter filters,
    VkSamplerAddressMode samplerAddressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);

VkWriteDescriptorSet write_descriptor_image(VkDescriptorType type, VkDescriptorSet dstSet,
                                            VkDescriptorImageInfo* imageInfo, uint32_t binding,
                                            int count);
} // namespace vkinit
