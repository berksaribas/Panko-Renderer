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
		UNIFORM, STORAGE, IMAGE, TEXTURE_SAMPLED, VERTEX, INDEX
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
		int baseMipLevel;
		int mipLevelCount;
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
		Slice<VkPipelineColorBlendAttachmentState> blendAttachmentStates;
		Slice<Bindable*> vertexBuffers;
		Bindable* indexBuffer;
		Slice<RenderTargetBinding> colorOutputs;
		RenderTargetBinding depthOutput;
	};

	struct ComputePipeline {
		std::string shader;
		int dimX;
		int dimY;
		int dimZ;
	};

	struct RaytracingPipeline {
		std::string rgenShader;
		std::string missShader;
		std::string hitShader;
		int recursionDepth;
		VkSpecializationInfo* rgenSpecialization;
		VkSpecializationInfo* missSpecialization;
		VkSpecializationInfo* hitSpecialization;
		int dimX;
		int dimY;
	};

	struct RenderPass {
		std::string name;
		PipelineType pipelineType;
		ComputePipeline computePipeline = {}; //type 0
		RasterPipeline rasterPipeline = {}; //type 1
		RaytracingPipeline raytracingPipeline = {}; //type 2
		Slice<DescriptorBinding> writes;
		Slice<DescriptorBinding> reads;
		Slice<VkPushConstantRange> pushConstantRanges;
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