#include "vk_debug_renderer.h"
#include <vk_pipeline.h>
#include <vk_initializers.h>
#include <vk_utils.h>

void VulkanDebugRenderer::init(VkDevice device, VmaAllocator allocator, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout)
{
	_device = device;
	_allocator = allocator;

	VkShaderModule unlitVertShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/debug_unlit.vert.spv", &unlitVertShader))
	{
		assert("Unlit Vertex Shader Loading Issue");
	}

	VkShaderModule unlitFragShader;
	if (!vkutils::load_shader_module(_device, "../../shaders/debug_unlit.frag.spv", &unlitFragShader))
	{
		assert("Unlit Fragment Shader Loading Issue");
	}

	PipelineBuilder pipelineBuilder;
	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_POINT);
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();
	auto colorBlendAttachment = vkinit::color_blend_attachment_state();
	pipelineBuilder._colorBlending = vkinit::color_blend_state_create_info(1, &colorBlendAttachment);

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	//we will have just 1 vertex buffer binding, with a per-vertex rate
	VkVertexInputBindingDescription vertexBinding = {};
	vertexBinding.binding = 0;
	vertexBinding.stride = sizeof(glm::vec3);
	vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputBindingDescription colorBinding = {};
	colorBinding.binding = 1;
	colorBinding.stride = sizeof(glm::vec3);
	colorBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	//Position will be stored at Location 0
	VkVertexInputAttributeDescription positionAttribute = {};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = 0;

	VkVertexInputAttributeDescription colorAttribute = {};
	colorAttribute.binding = 1;
	colorAttribute.location = 1;
	colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset = 0;

	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;
	bindings.push_back(vertexBinding);
	bindings.push_back(colorBinding);
	attributes.push_back(positionAttribute);
	attributes.push_back(colorAttribute);

	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = attributes.data();
	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();

	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = bindings.data();
	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = bindings.size();
	
	//add the other shaders
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, unlitVertShader));

	//make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, unlitFragShader));

	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info(&globalSetLayout, 1);
	VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_pipelineLayout));
	pipelineBuilder._pipelineLayout = _pipelineLayout;

	//build the mesh triangle pipeline
	_pointPipeline = pipelineBuilder.build_pipeline(_device, renderPass);

	pipelineBuilder._inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	pipelineBuilder._rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
	_linePipeline = pipelineBuilder.build_pipeline(_device, renderPass);

	vkDestroyShaderModule(_device, unlitVertShader, nullptr);
	vkDestroyShaderModule(_device, unlitFragShader, nullptr);

	//Create the buffers
	_pointVertexBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_pointColorBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	
	_lineVertexBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_lineColorBuffer = vkutils::create_buffer(_allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
}

void VulkanDebugRenderer::draw_line(glm::vec3 start, glm::vec3 end, glm::vec3 color)
{
	linePositions.push_back(start);
	linePositions.push_back(end);
	lineColors.push_back(color);
	lineColors.push_back(color);
}

void VulkanDebugRenderer::draw_point(glm::vec3 point, glm::vec3 color)
{
	pointPositions.push_back(point);
	pointColors.push_back(color);
}

void VulkanDebugRenderer::render(VkCommandBuffer cmd, VkDescriptorSet globalDescriptorSet)
{
	

	/*
	* DRAW POINTS
	*/
	if (pointPositions.size() > 0) {
		vkutils::cpu_to_gpu(_allocator, _pointVertexBuffer, pointPositions.data(), pointPositions.size() * sizeof(glm::vec3));
		vkutils::cpu_to_gpu(_allocator, _pointColorBuffer, pointColors.data(), pointColors.size() * sizeof(glm::vec3));

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pointPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout, 0, 1, &globalDescriptorSet, 0, nullptr);
		VkDeviceSize pointOffsets[] = { 0, 0, 0 };
		VkBuffer pointBuffers[] = { _pointVertexBuffer._buffer, _pointColorBuffer._buffer };
		vkCmdBindVertexBuffers(cmd, 0, 2, pointBuffers, pointOffsets);
		vkCmdDraw(cmd, pointPositions.size(), 1, 0, 0);

		pointPositions.clear();
		pointColors.clear();
	}

	if (linePositions.size() > 0) {
		vkutils::cpu_to_gpu(_allocator, _lineVertexBuffer, linePositions.data(), linePositions.size() * sizeof(glm::vec3));
		vkutils::cpu_to_gpu(_allocator, _lineColorBuffer, lineColors.data(), lineColors.size() * sizeof(glm::vec3));

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _linePipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout, 0, 1, &globalDescriptorSet, 0, nullptr);
		VkDeviceSize lineOffsets[] = { 0, 0, 0 };
		VkBuffer lineBuffers[] = { _lineVertexBuffer._buffer, _lineColorBuffer._buffer };
		vkCmdBindVertexBuffers(cmd, 0, 2, lineBuffers, lineOffsets);
		vkCmdDraw(cmd, linePositions.size(), 1, 0, 0);

		linePositions.clear();
		lineColors.clear();
	}
}

void VulkanDebugRenderer::cleanup()
{
	vmaDestroyBuffer(_allocator, _pointVertexBuffer._buffer, _pointVertexBuffer._allocation);
	vmaDestroyBuffer(_allocator, _pointColorBuffer._buffer, _pointColorBuffer._allocation);

	vkDestroyPipeline(_device, _pointPipeline, nullptr);
	vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);
}
