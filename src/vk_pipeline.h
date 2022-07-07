#include <vk_types.h>
#include <vector>

class PipelineBuilder {
public:

	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendStateCreateInfo _colorBlending;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;
	VkPipelineLayout _pipelineLayout;
	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass, VkPipelineRenderingCreateInfo* dynamicRendering = nullptr);
};


