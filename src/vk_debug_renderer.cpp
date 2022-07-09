#include "vk_debug_renderer.h"
#include <vk_pipeline.h>
#include <vk_initializers.h>
#include <vk_utils.h>
#include "vk_rendergraph.h"

void VulkanDebugRenderer::init(EngineData& _engineData)
{
	//Create the buffers
	_pointVertexBuffer = vkutils::create_buffer(_engineData.allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_pointColorBuffer = vkutils::create_buffer(_engineData.allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_pointVertexBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_pointVertexBuffer, VK_FORMAT_R32G32B32_SFLOAT, "DebugPointVertexPosBuffer");
	_pointColorBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_pointColorBuffer, VK_FORMAT_R32G32B32_SFLOAT, "DebugPointVertexColorBuffer");
	
	_lineVertexBuffer = vkutils::create_buffer(_engineData.allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_lineColorBuffer = vkutils::create_buffer(_engineData.allocator, sizeof(glm::vec3) * 1024000, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	_lineVertexBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_lineVertexBuffer, VK_FORMAT_R32G32B32_SFLOAT, "DebugLineVertexPosBuffer");
	_lineColorBufferBinding = _engineData.renderGraph->register_vertex_buffer(&_lineColorBuffer, VK_FORMAT_R32G32B32_SFLOAT, "DebugLineVertexColorBuffer");
}

void VulkanDebugRenderer::draw_line(glm::vec3 start, glm::vec3 end, glm::vec3 color)
{
	_linePositions.push_back(start);
	_linePositions.push_back(end);
	_lineColors.push_back(color);
	_lineColors.push_back(color);
}

void VulkanDebugRenderer::draw_point(glm::vec3 point, glm::vec3 color)
{
	_pointPositions.push_back(point);
	_pointColors.push_back(color);
}

Vrg::RenderPass* pointPass;
Vrg::RenderPass* linePass;

void VulkanDebugRenderer::render(EngineData& _engineData, SceneData& _sceneData, VkExtent2D size, Vrg::Bindable* renderTarget)
{
	pointPass = nullptr;
	linePass = nullptr;
	VkClearValue clearValue;

	if (_pointPositions.size() > 0) {
		vkutils::cpu_to_gpu(_engineData.allocator, _pointVertexBuffer, _pointPositions.data(), _pointPositions.size() * sizeof(glm::vec3));
		vkutils::cpu_to_gpu(_engineData.allocator, _pointColorBuffer, _pointColors.data(), _pointColors.size() * sizeof(glm::vec3));
	
		pointPass = _engineData.renderGraph->add_render_pass({
			.name = "DebugPointPass",
			.pipelineType = Vrg::PipelineType::RASTER_TYPE,
			.rasterPipeline = {
				.vertexShader = "../../shaders/debug_unlit.vert.spv",
				.fragmentShader = "../../shaders/debug_unlit.frag.spv",
				.size = size,
				.inputAssembly = Vrg::InputAssembly::POINT,
				.polygonMode = Vrg::PolygonMode::POINT,
				.depthState = { false, false, VK_COMPARE_OP_NEVER },
				.cullMode = Vrg::CullMode::NONE,
				.blendAttachmentStates = {
					vkinit::color_blend_attachment_state(),
				},
				.vertexBuffers = {
					_pointVertexBufferBinding,
					_pointColorBufferBinding
				},
				.colorOutputs = {
					{renderTarget, clearValue, true},
				},

			},
			.reads = {
				{0, _sceneData.cameraBufferBinding}
			},
			.execute = [&](VkCommandBuffer cmd) {
				vkCmdDraw(cmd, _pointPositions.size(), 1, 0, 0);
			},
			.skipExecution = true
		});
	}

	if (_linePositions.size() > 0) {
		vkutils::cpu_to_gpu(_engineData.allocator, _lineVertexBuffer, _linePositions.data(), _linePositions.size() * sizeof(glm::vec3));
		vkutils::cpu_to_gpu(_engineData.allocator, _lineColorBuffer, _lineColors.data(), _lineColors.size() * sizeof(glm::vec3));

		linePass = _engineData.renderGraph->add_render_pass({
			.name = "DebugLinePass",
			.pipelineType = Vrg::PipelineType::RASTER_TYPE,
			.rasterPipeline = {
				.vertexShader = "../../shaders/debug_unlit.vert.spv",
				.fragmentShader = "../../shaders/debug_unlit.frag.spv",
				.size = size,
				.inputAssembly = Vrg::InputAssembly::LINE,
				.polygonMode = Vrg::PolygonMode::LINE,
				.depthState = { false, false, VK_COMPARE_OP_NEVER },
				.cullMode = Vrg::CullMode::NONE,
				.blendAttachmentStates = {
					vkinit::color_blend_attachment_state(),
				},
				.vertexBuffers = {
					_lineVertexBufferBinding,
					_lineColorBufferBinding
				},
				.colorOutputs = {
					{renderTarget, clearValue, true},
				},

			},
			.reads = {
				{0, _sceneData.cameraBufferBinding}
			},
			.execute = [&](VkCommandBuffer cmd) {
				vkCmdDraw(cmd, _linePositions.size(), 1, 0, 0);
			},
			.skipExecution = true
		});
	}

}

void VulkanDebugRenderer::custom_execute(VkCommandBuffer cmd, EngineData& _engineData)
{
	if (pointPass != nullptr) {
		_engineData.renderGraph->handle_render_pass_barriers(cmd, *pointPass);
		_engineData.renderGraph->bind_pipeline_and_descriptors(cmd, *pointPass);
		pointPass->execute(cmd);
	}

	if (linePass != nullptr) {
		_engineData.renderGraph->handle_render_pass_barriers(cmd, *linePass);
		_engineData.renderGraph->bind_pipeline_and_descriptors(cmd, *linePass);
		linePass->execute(cmd);
	}

	_pointPositions.clear();
	_pointColors.clear();

	_linePositions.clear();
	_lineColors.clear();
}