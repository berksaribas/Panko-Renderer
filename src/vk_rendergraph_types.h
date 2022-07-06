#pragma once
#include <vk_types.h>
#include <slice.h>
#include <functional>

namespace Vrg {
	enum class ResourceAccessType {
		NONE,
		COLOR_WRITE,
		DEPTH_WRITE,
		COMPUTE_WRITE,
		COMPUTE_READ,
		FRAGMENT_READ,
		RAYTRACING_WRITE,
		RAYTRACING_READ
	};

	enum class BindType {
		UNIFORM, STORAGE, IMAGE_VIEW, VERTEX, INDEX
	};

	enum class Sampler {
		NEAREST,
		LINEAR
	};

	enum class PipelineType {
		COMPUTE_TYPE,
		RASTER_TYPE,
		RAYTRACING_TYPE
	};
	struct ImageView {
		Sampler sampler;
		uint32_t baseMipLevel;
		uint32_t mipLevelCount;
		bool operator==(ImageView const& rhs) const noexcept {
			return sampler == rhs.sampler && baseMipLevel == rhs.baseMipLevel && mipLevelCount == rhs.mipLevelCount;
		}
	};

	struct Bindable {
		AllocatedBuffer* buffer;
		AllocatedImage* image;
		ImageView imageView;
		VkFormat format;
		BindType type;
		int count;
		uint32_t id;
	};

	struct DescriptorBinding {
		int set_index;
		Bindable* binding;
	};

	struct DescriptorSetBinding {
		int set_index;
		VkDescriptorSet descriptorSet;
		VkDescriptorSetLayout descriptorSetLayout;
	};

	struct RenderTargetBinding {
		Bindable* binding;
		VkClearValue clearValue;
		bool isSwapChain;
	};

	struct PushConstant {
		void* data;
		size_t size;
	};

	enum class InputAssembly {
		TRIANGLE,
		POINT,
		LINE
	};

	enum class PolygonMode {
		FILL,
		POINT,
		LINE
	};

	enum class CullMode {
		NONE,
		CLOCK_WISE,
		COUNTER_CLOCK_WISE
	};

	struct DepthState {
		bool depthTest;
		bool depthWrite;
		VkCompareOp compareOp;
		bool operator==(const DepthState& p) const {
			return depthTest == p.depthTest && depthWrite == p.depthWrite && compareOp == p.compareOp;
		}
	};

	struct RasterPipeline {
		std::string vertexShader;
		std::string fragmentShader;
		VkExtent2D size;
		InputAssembly inputAssembly = InputAssembly::TRIANGLE;
		PolygonMode polygonMode = PolygonMode::FILL;
		DepthState depthState = { true, true, VK_COMPARE_OP_LESS_OR_EQUAL };
		CullMode cullMode = CullMode::COUNTER_CLOCK_WISE;
		bool enableConservativeRasterization = false;
		Slice<VkPipelineColorBlendAttachmentState> blendAttachmentStates;
		Slice<Bindable*> vertexBuffers;
		Bindable* indexBuffer;
		Slice<RenderTargetBinding> colorOutputs;
		RenderTargetBinding depthOutput;
	};

	struct ComputePipeline {
		std::string shader;
		uint32_t dimX;
		uint32_t dimY;
		uint32_t dimZ;
	};

	struct RayPipeline {
		std::string rgenShader;
		std::string missShader;
		std::string hitShader;
		int recursionDepth = 1;
		VkSpecializationInfo rgenSpecialization;
		VkSpecializationInfo missSpecialization;
		VkSpecializationInfo hitSpecialization;
		uint32_t width = 1;
		uint32_t height = 1;
		uint32_t depth = 1;
	};

	struct RenderPass {
		std::string name;
		PipelineType pipelineType;
		ComputePipeline computePipeline = {}; //type 0
		RasterPipeline rasterPipeline = {}; //type 1
		RayPipeline raytracingPipeline = {}; //type 2
		Slice<DescriptorBinding> writes;
		Slice<DescriptorBinding> reads;
		Slice<PushConstant> constants;
		Slice<DescriptorSetBinding> extraDescriptorSets;
		uint32_t descriptorSetCount;
		std::function<void(VkCommandBuffer cmd)> execute;
	};
	
	struct MemoryPass {
		Bindable* src;
		Bindable* dst;
		//TODO: Blit support
	};

	struct DescriptorSet {
		VkDescriptorType type;
		AllocatedBuffer* buffer;
		VkImage image;
		ImageView imageView;
		VkImageLayout imageLayout;
		VkFormat format;
	};
}