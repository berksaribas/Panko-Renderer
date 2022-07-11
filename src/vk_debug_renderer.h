#pragma once
#include <vk_types.h>
#include <vector>
#include <glm/vec3.hpp>

class VulkanDebugRenderer {
public:
	void init(EngineData& _engineData);
	void draw_line(glm::vec3 start, glm::vec3 end, glm::vec3 color);
	void draw_point(glm::vec3 point, glm::vec3 color);
	void render(EngineData& _engineData, SceneData& _sceneData, VkExtent2D size, Handle<Vrg::Bindable> renderTarget);
	void custom_execute(VkCommandBuffer cmd, EngineData& _engineData);
private:
	//Point
	std::vector<glm::vec3> _pointPositions, _pointColors;
	AllocatedBuffer _pointVertexBuffer, _pointColorBuffer;
	Handle<Vrg::Bindable> _pointVertexBufferBinding;
	Handle<Vrg::Bindable> _pointColorBufferBinding;
	//Line
	std::vector<glm::vec3> _linePositions, _lineColors;
	AllocatedBuffer _lineVertexBuffer, _lineColorBuffer;
	Handle<Vrg::Bindable> _lineVertexBufferBinding;
	Handle<Vrg::Bindable> _lineColorBufferBinding;
};