#pragma once
#include <vk_types.h>
#include <vector>
#include <gltf_scene.hpp>

template <class T> constexpr T align_up(T x, size_t a) noexcept {
	return T((x + (T(a) - 1)) & ~T(a - 1));
}

struct BlasInput {
	// Data used to build acceleration structure geometry
	std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
	std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
	VkBuildAccelerationStructureFlagsKHR flags{ 0 };
};

struct AccelKHR
{
	VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
	AllocatedBuffer buffer;
};

struct BuildAccelerationStructure {
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo;
	AccelKHR as;
	AccelKHR cleanupAs;
};

struct RaytracingPipeline {
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;

	AllocatedBuffer rtSBTBuffer;
	VkStridedDeviceAddressRegionKHR rgenRegion{};
	VkStridedDeviceAddressRegionKHR missRegion{};
	VkStridedDeviceAddressRegionKHR hitRegion{};
	VkStridedDeviceAddressRegionKHR callRegion{};
};

class VulkanRaytracing {
public:
	//Setup functions
	void init(VkDevice device, VkPhysicalDeviceRayTracingPipelinePropertiesKHR gpuRaytracingProperties, VmaAllocator allocator, VkQueue queue, uint32_t queueFamily);
	void convert_scene_to_vk_geometry(GltfScene& scene, AllocatedBuffer& vertexBuffer, AllocatedBuffer& indexBuffer);
	void build_blas(VkBuildAccelerationStructureFlagsKHR flags);
	void build_tlas(GltfScene& scene, VkBuildAccelerationStructureFlagsKHR flags, bool update);
	//Per pipeline
	void create_new_pipeline(RaytracingPipeline& raytracingPipeline, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo, const char* rgenPath, const char* missPath, const char* hitPath);
	void destroy_raytracing_pipeline(RaytracingPipeline& raytracingPipeline);
	AccelKHR tlas;
	CommandContext _raytracingContext;
	VkQueue _queue;
private:
	void cmd_create_blas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkDeviceAddress scratchAddress, VkQueryPool queryPool);
	void cmd_compact_blas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkQueryPool queryPool);
	void cmd_create_tlas(VkCommandBuffer cmdBuf, uint32_t countInstance, VkDeviceAddress instBufferAddr, AllocatedBuffer& scratchBuffer, VkBuildAccelerationStructureFlagsKHR flags, bool update);
	AccelKHR create_acceleration(VkAccelerationStructureCreateInfoKHR& accel);
	VkDevice _device;
	uint32_t _queueFamily;
	VmaAllocator _allocator;
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR _gpuRaytracingProperties;
	std::vector<BlasInput> _blasInputs;
	std::vector<AccelKHR> _blases;
};