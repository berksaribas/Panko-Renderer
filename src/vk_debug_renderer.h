#pragma once
#include <vk_types.h>
#include <vector>

class VulkanDebugRenderer {
public:
	void init(VkDevice device, VmaAllocator allocator, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout);
	void draw_line(glm::vec3 start, glm::vec3 end, glm::vec3 color);
	void draw_point(glm::vec3 point, glm::vec3 color);
	void render(VkCommandBuffer cmd, VkDescriptorSet globalDescriptorSet);
	void cleanup();
private:
	VkDevice _device;
	VmaAllocator _allocator;
	VkPipelineLayout _pipelineLayout;

	//Point
	std::vector<glm::vec3> pointPositions, pointColors;
	AllocatedBuffer _pointVertexBuffer, _pointColorBuffer;
	VkPipeline _pointPipeline;
	//Line
	std::vector<glm::vec3> linePositions, lineColors;
	AllocatedBuffer _lineVertexBuffer, _lineColorBuffer;
	VkPipeline _linePipeline;

};