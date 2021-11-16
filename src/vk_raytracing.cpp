#include "vk_raytracing.h"
#include <vk_initializers.h>
#include <vk_engine.h>
#include <vk_utils.h>

static bool hasFlag(VkFlags item, VkFlags flag) {
	return (item & flag) == flag;
}

void VulkanRaytracing::init(VkDevice device, VkPhysicalDeviceRayTracingPipelinePropertiesKHR gpuRaytracingProperties, VmaAllocator allocator, VkQueue queue, uint32_t queueFamily)
{
	_device = device;
	_allocator = allocator;
	_queue = queue;
	_queueFamily = queueFamily;
	_gpuRaytracingProperties = gpuRaytracingProperties;
	//create pool for compute context
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_queueFamily);
	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_raytracingContext._commandPool));

	//create fence
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info();
	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_raytracingContext._fence));
}

void VulkanRaytracing::convert_scene_to_vk_geometry(GltfScene& scene, AllocatedBuffer& vertexBuffer, AllocatedBuffer& indexBuffer)
{
	VkBufferDeviceAddressInfo info = { };
	info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

	info.buffer = vertexBuffer._buffer;
	VkDeviceAddress vertexAddress = vkGetBufferDeviceAddress(_device, &info);

	info.buffer = indexBuffer._buffer;
	VkDeviceAddress indexAddress = vkGetBufferDeviceAddress(_device, &info);

	for (int i = 0; i < scene.prim_meshes.size(); i++) {
		VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
		triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
		triangles.vertexData.deviceAddress = vertexAddress;
		triangles.vertexStride = sizeof(glm::vec3);
		// Describe index data (32-bit unsigned int)
		triangles.indexType = VK_INDEX_TYPE_UINT32;
		triangles.indexData.deviceAddress = indexAddress;
		// Indicate identity transform by setting transformData to null device pointer.
		//triangles.transformData = {};
		triangles.maxVertex = scene.prim_meshes[i].vtx_count;

		// Identify the above data as containing opaque triangles.
		VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		asGeom.geometry.triangles = triangles;

		// The entire array will be used to build the BLAS.
		VkAccelerationStructureBuildRangeInfoKHR offset;
		offset.firstVertex = scene.prim_meshes[i].vtx_offset;
		offset.primitiveCount = scene.prim_meshes[i].idx_count / 3;
		offset.primitiveOffset = scene.prim_meshes[i].first_idx * sizeof(uint32_t); // why?
		offset.transformOffset = 0;

		BlasInput input;
		input.asGeometry.emplace_back(asGeom);
		input.asBuildOffsetInfo.emplace_back(offset);

		_blasInputs.push_back(input);
	}
}

void VulkanRaytracing::build_blas(VkBuildAccelerationStructureFlagsKHR flags)
{
	uint32_t     nbBlas = static_cast<uint32_t>(_blasInputs.size());
	VkDeviceSize asTotalSize{ 0 };     // Memory size of all allocated BLAS
	uint32_t     nbCompactions{ 0 };   // Nb of BLAS requesting compaction
	VkDeviceSize maxScratchSize{ 0 };  // Largest scratch size
	std::vector<BuildAccelerationStructure> buildAs(nbBlas);
	for (uint32_t idx = 0; idx < nbBlas; idx++)
	{
		// Filling partially the VkAccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
		// Other information will be filled in the createBlas (see #2)
		buildAs[idx].buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildAs[idx].buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		buildAs[idx].buildInfo.flags = _blasInputs[idx].flags | flags;
		buildAs[idx].buildInfo.geometryCount = static_cast<uint32_t>(_blasInputs[idx].asGeometry.size());
		buildAs[idx].buildInfo.pGeometries = _blasInputs[idx].asGeometry.data();

		// Build range information
		buildAs[idx].rangeInfo = _blasInputs[idx].asBuildOffsetInfo.data();

		// Finding sizes to create acceleration structures and scratch
		std::vector<uint32_t> maxPrimCount(_blasInputs[idx].asBuildOffsetInfo.size());
		for (auto tt = 0; tt < _blasInputs[idx].asBuildOffsetInfo.size(); tt++) {
			maxPrimCount[tt] = _blasInputs[idx].asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
		}
		vkGetAccelerationStructureBuildSizesKHR(_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&buildAs[idx].buildInfo, maxPrimCount.data(), &buildAs[idx].sizeInfo);

		// Extra info
		asTotalSize += buildAs[idx].sizeInfo.accelerationStructureSize;
		maxScratchSize = std::max(maxScratchSize, buildAs[idx].sizeInfo.buildScratchSize);
		nbCompactions += hasFlag(buildAs[idx].buildInfo.flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
	}
	// Allocate the scratch buffers holding the temporary data of the acceleration structure builder
	AllocatedBuffer scratchBuffer = vkutils::create_buffer(_allocator, maxScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VkBufferDeviceAddressInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratchBuffer._buffer };
	VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(_device, &bufferInfo);
	// Allocate a query pool for storing the needed size for every BLAS compaction.
	VkQueryPool queryPool{ VK_NULL_HANDLE };
	if (nbCompactions > 0)  // Is compaction requested?
	{
		assert(nbCompactions == nbBlas);  // Don't allow mix of on/off compaction
		VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
		qpci.queryCount = nbBlas;
		qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
		vkCreateQueryPool(_device, &qpci, nullptr, &queryPool);
	}

	// Batching creation/compaction of BLAS to allow staying in restricted amount of memory
	std::vector<uint32_t> indices;  // Indices of the BLAS to create
	VkDeviceSize          batchSize{ 0 };
	VkDeviceSize          batchLimit{ 256'000'000 };  // 256 MB
	for (uint32_t idx = 0; idx < nbBlas; idx++)
	{
		indices.push_back(idx);
		batchSize += buildAs[idx].sizeInfo.accelerationStructureSize;
		// Over the limit or last BLAS element
		if (batchSize >= batchLimit || idx == nbBlas - 1)
		{
			VkCommandBuffer cmdBuf = vkutils::create_command_buffer(_device, _raytracingContext._commandPool, true);
			cmd_create_blas(cmdBuf, indices, buildAs, scratchAddress, queryPool);
			vkutils::submit_and_free_command_buffer(_device, _raytracingContext._commandPool, cmdBuf, _queue, _raytracingContext._fence);

			if (queryPool)
			{
				VkCommandBuffer cmdBuf = vkutils::create_command_buffer(_device, _raytracingContext._commandPool, true);
				cmd_compact_blas(cmdBuf, indices, buildAs, queryPool);
				vkutils::submit_and_free_command_buffer(_device, _raytracingContext._commandPool, cmdBuf, _queue, _raytracingContext._fence);

				// Destroy the non-compacted version
				for (auto i : indices) {
					vkDestroyAccelerationStructureKHR(
						_device, buildAs[i].cleanupAs.accel, nullptr);
					vmaDestroyBuffer(_allocator, buildAs[i].cleanupAs.buffer._buffer, buildAs[i].cleanupAs.buffer._allocation);
				}
			}
			// Reset

			batchSize = 0;
			indices.clear();
		}
	}

	// Keeping all the created acceleration structures
	for (auto& b : buildAs)
	{
		_blases.emplace_back(b.as);
	}

	vkDestroyQueryPool(_device, queryPool, nullptr);
	vmaDestroyBuffer(_allocator, scratchBuffer._buffer, scratchBuffer._allocation);
}

void VulkanRaytracing::build_tlas(GltfScene& scene, VkBuildAccelerationStructureFlagsKHR flags, bool update)
{
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	instances.reserve(scene.nodes.size());
	for (int i = 0; i < scene.nodes.size(); i++)
	{
		glm::mat4 transposed = glm::transpose(scene.nodes[i].world_matrix);
		VkTransformMatrixKHR vkMatrix;
		memcpy(&vkMatrix, &transposed, sizeof(VkTransformMatrixKHR));

		VkAccelerationStructureDeviceAddressInfoKHR addr_info{
			VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
		addr_info.accelerationStructure = _blases[scene.nodes[i].prim_mesh].accel;

		VkAccelerationStructureInstanceKHR rayInst{};
		rayInst.transform = vkMatrix;  // Position of the instance
		rayInst.instanceCustomIndex = scene.nodes[i].prim_mesh;                               // gl_InstanceCustomIndexEXT
		rayInst.accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR(_device, &addr_info);
		rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		rayInst.mask = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
		rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
		instances.emplace_back(rayInst);
	}

	// Cannot call buildTlas twice except to update.
	assert(tlas.accel == VK_NULL_HANDLE || update);
	uint32_t countInstance = static_cast<uint32_t>(instances.size());

	// Command buffer to create the TLAS
	VkCommandBuffer cmdBuf = vkutils::create_command_buffer(_device, _raytracingContext._commandPool, true);

	// Create a buffer holding the actual instance data (matrices++) for use by the AS builder
	AllocatedBuffer instancesBuffer;  // Buffer of instances containing the matrices and BLAS ids
	instancesBuffer = vkutils::create_buffer(_allocator, sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	// TODO: BUG HERE
	vkutils::cpu_to_gpu_staging(_allocator, cmdBuf, instancesBuffer, instances.data(), sizeof(VkAccelerationStructureInstanceKHR) * instances.size());

	VkBufferDeviceAddressInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, instancesBuffer._buffer };
	VkDeviceAddress           instBufferAddr = vkGetBufferDeviceAddress(_device, &bufferInfo);

	// Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
	VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
	vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		0, 1, &barrier, 0, nullptr, 0, nullptr);

	// Creating the TLAS
	AllocatedBuffer scratchBuffer;
	cmd_create_tlas(cmdBuf, countInstance, instBufferAddr, scratchBuffer, flags, update);

	// Finalizing and destroying temporary data
	vkutils::submit_and_free_command_buffer(_device, _raytracingContext._commandPool, cmdBuf, _queue, _raytracingContext._fence);

	vmaDestroyBuffer(_allocator, scratchBuffer._buffer, scratchBuffer._allocation);
	vmaDestroyBuffer(_allocator, instancesBuffer._buffer, instancesBuffer._allocation);
}

void VulkanRaytracing::create_new_pipeline(RaytracingPipeline& raytracingPipeline, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo, const char* rgenPath, const char* missPath, const char* hitPath)
{
	enum StageIndices
	{
		eRaygen,
		eMiss,
		eClosestHit,
		eShaderGroupCount
	};

	// All stages
	VkPipelineShaderStageCreateInfo stages[eShaderGroupCount] = {};
	VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	stage.pName = "main";  // All the same entry point
	// Raygen
	stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	stages[eRaygen] = stage;
	vkutils::load_shader_module(_device, rgenPath, &stages[eRaygen].module);
	// Miss
	stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	stages[eMiss] = stage;
	vkutils::load_shader_module(_device, missPath, &stages[eMiss].module);
	// Hit Group - Closest Hit
	stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	stages[eClosestHit] = stage;
	vkutils::load_shader_module(_device, hitPath, &stages[eClosestHit].module);

	VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	// Raygen
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	raytracingPipeline.shaderGroups.push_back(group);

	// Miss
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	raytracingPipeline.shaderGroups.push_back(group);

	// closest hit shader
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	raytracingPipeline.shaderGroups.push_back(group);

	vkCreatePipelineLayout(_device, &pipelineLayoutCreateInfo, nullptr, &raytracingPipeline.pipelineLayout);

	// Assemble the shader stages and recursion depth info into the ray tracing pipeline
	VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rayPipelineInfo.stageCount = static_cast<uint32_t>(eShaderGroupCount);  // Stages are shaders
	rayPipelineInfo.pStages = stages;

	// In this case, m_rtShaderGroups.size() == 3: we have one raygen group,
	// one miss shader group, and one hit group.
	rayPipelineInfo.groupCount = static_cast<uint32_t>(raytracingPipeline.shaderGroups.size());
	rayPipelineInfo.pGroups = raytracingPipeline.shaderGroups.data();
	rayPipelineInfo.maxPipelineRayRecursionDepth = 1;  // Ray depth
	rayPipelineInfo.layout = raytracingPipeline.pipelineLayout;

	vkCreateRayTracingPipelinesKHR(_device, {}, {}, 1, & rayPipelineInfo, nullptr, &raytracingPipeline.pipeline);

	for (auto& s : stages) {
		vkDestroyShaderModule(_device, s.module, nullptr);
	}

	// Create the binding table
	uint32_t missCount{ 1 };
	uint32_t hitCount{ 1 };
	auto     handleCount = 1 + missCount + hitCount;
	uint32_t handleSize = _gpuRaytracingProperties.shaderGroupHandleSize;

	// The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
	uint32_t handleSizeAligned = align_up(handleSize, _gpuRaytracingProperties.shaderGroupHandleAlignment);
	
	raytracingPipeline.rgenRegion.stride = align_up(handleSizeAligned, _gpuRaytracingProperties.shaderGroupBaseAlignment);
	raytracingPipeline.rgenRegion.size = raytracingPipeline.rgenRegion.stride;  // The size member of pRayGenShaderBindingTable must be equal to its stride member
	raytracingPipeline.missRegion.stride = handleSizeAligned;
	raytracingPipeline.missRegion.size = align_up(missCount * handleSizeAligned, _gpuRaytracingProperties.shaderGroupBaseAlignment);
	raytracingPipeline.hitRegion.stride = handleSizeAligned;
	raytracingPipeline.hitRegion.size = align_up(hitCount * handleSizeAligned, _gpuRaytracingProperties.shaderGroupBaseAlignment);

	// Get the shader group handles
	uint32_t             dataSize = handleCount * handleSize;
	std::vector<uint8_t> handles(dataSize);
	auto result = vkGetRayTracingShaderGroupHandlesKHR(_device, raytracingPipeline.pipeline, 0, handleCount, dataSize, handles.data());
	assert(result == VK_SUCCESS);

	// Allocate a buffer for storing the SBT.
	VkDeviceSize sbtSize = raytracingPipeline.rgenRegion.size + raytracingPipeline.missRegion.size + raytracingPipeline.hitRegion.size + raytracingPipeline.callRegion.size;
	raytracingPipeline.rtSBTBuffer = vkutils::create_buffer(_allocator, sbtSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		| VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Find the SBT addresses of each group
	VkBufferDeviceAddressInfo info{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, raytracingPipeline.rtSBTBuffer._buffer };
	VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(_device, &info);
	raytracingPipeline.rgenRegion.deviceAddress = sbtAddress;
	raytracingPipeline.missRegion.deviceAddress = sbtAddress + raytracingPipeline.rgenRegion.size;
	raytracingPipeline.hitRegion.deviceAddress = sbtAddress + raytracingPipeline.rgenRegion.size + raytracingPipeline.missRegion.size;

	// Helper to retrieve the handle data
	auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

	// Map the SBT buffer and write in the handles.
	void* pSBTBuffer_void;
	vmaMapMemory(_allocator, raytracingPipeline.rtSBTBuffer._allocation, &pSBTBuffer_void);

	auto* pSBTBuffer = reinterpret_cast<uint8_t*>(pSBTBuffer_void);
	uint8_t* pData{ nullptr };
	uint32_t handleIdx{ 0 };

	// Raygen
	pData = pSBTBuffer;
	memcpy(pData, getHandle(handleIdx++), handleSize);

	// Miss
	pData = pSBTBuffer + raytracingPipeline.rgenRegion.size;
	for (uint32_t c = 0; c < missCount; c++)
	{
		memcpy(pData, getHandle(handleIdx++), handleSize);
		pData += raytracingPipeline.missRegion.stride;
	}

	// Hit
	pData = pSBTBuffer + raytracingPipeline.rgenRegion.size + raytracingPipeline.missRegion.size;
	for (uint32_t c = 0; c < hitCount; c++)
	{
		memcpy(pData, getHandle(handleIdx++), handleSize);
		pData += raytracingPipeline.hitRegion.stride;
	}

	vmaUnmapMemory(_allocator, raytracingPipeline.rtSBTBuffer._allocation);
}

void VulkanRaytracing::destroy_raytracing_pipeline(RaytracingPipeline& raytracingPipeline)
{
	vkDestroyPipeline(_device, raytracingPipeline.pipeline, nullptr);
	vkDestroyPipelineLayout(_device, raytracingPipeline.pipelineLayout, nullptr);
	vmaDestroyBuffer(_allocator, raytracingPipeline.rtSBTBuffer._buffer, raytracingPipeline.rtSBTBuffer._allocation);
}

void VulkanRaytracing::cmd_create_blas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkDeviceAddress scratchAddress, VkQueryPool queryPool)
{
	if (queryPool)  // For querying the compaction size
		vkResetQueryPool(_device, queryPool, 0, static_cast<uint32_t>(indices.size()));
	uint32_t queryCnt{ 0 };
	for (const auto& idx : indices)
	{
		// Actual allocation of buffer and acceleration structure.
		VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		createInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.
		buildAs[idx].as = create_acceleration(createInfo);

		// BuildInfo #2 part
		buildAs[idx].buildInfo.dstAccelerationStructure = buildAs[idx].as.accel;  // Setting where the build lands
		buildAs[idx].buildInfo.scratchData.deviceAddress = scratchAddress;  // All build are using the same scratch buffer

		// Building the bottom-level-acceleration-structure
		vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildAs[idx].buildInfo, &buildAs[idx].rangeInfo);

		// Since the scratch buffer is reused across builds, we need a barrier to ensure one build
// is finished before starting the next one.
		VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
			VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

		if (queryPool)
		{
			// Add a query to find the 'real' amount of memory needed, use for compaction
			vkCmdWriteAccelerationStructuresPropertiesKHR(cmdBuf, 1, &buildAs[idx].buildInfo.dstAccelerationStructure,
				VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, queryCnt++);
		}
	}
}

void VulkanRaytracing::cmd_compact_blas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkQueryPool queryPool)
{
	uint32_t                    queryCtn{ 0 };
	std::vector<AccelKHR> cleanupAS;  // previous AS to destroy

	// Get the compacted size result back
	std::vector<VkDeviceSize> compactSizes(static_cast<uint32_t>(indices.size()));
	vkGetQueryPoolResults(_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
		compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

	for (auto idx : indices)
	{
		buildAs[idx].cleanupAs = buildAs[idx].as;           // previous AS to destroy
		buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++];  // new reduced size

		// Creating a compact version of the AS
		VkAccelerationStructureCreateInfoKHR asCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		asCreateInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;
		asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildAs[idx].as = create_acceleration(asCreateInfo);

		// Copy the original BLAS to a compact version
		VkCopyAccelerationStructureInfoKHR copyInfo{ VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
		copyInfo.src = buildAs[idx].buildInfo.dstAccelerationStructure;
		copyInfo.dst = buildAs[idx].as.accel;
		copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
		vkCmdCopyAccelerationStructureKHR(cmdBuf, &copyInfo);
	}
}

void VulkanRaytracing::cmd_create_tlas(VkCommandBuffer cmdBuf, uint32_t countInstance, VkDeviceAddress instBufferAddr, AllocatedBuffer& scratchBuffer, VkBuildAccelerationStructureFlagsKHR flags, bool update)
{
	// Wraps a device pointer to the above uploaded instances.
	VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
	instancesVk.data.deviceAddress = instBufferAddr;

	// Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
	VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	topASGeometry.geometry.instances = instancesVk;

	// Find sizes
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	buildInfo.flags = flags;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &topASGeometry;
	buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetAccelerationStructureBuildSizesKHR(_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
		&countInstance, &sizeInfo);

	VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
	createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	createInfo.size = sizeInfo.accelerationStructureSize;

	tlas = create_acceleration(createInfo);

	// Allocate the scratch memory
	scratchBuffer = vkutils::create_buffer(_allocator, sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	VkBufferDeviceAddressInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratchBuffer._buffer };
	VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(_device, &bufferInfo);

	// Update build information
	buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
	buildInfo.dstAccelerationStructure = tlas.accel;
	buildInfo.scratchData.deviceAddress = scratchAddress;

	// Build Offsets info: n instances
	VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{ countInstance, 0, 0, 0 };
	const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

	// Build the TLAS
	vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, &pBuildOffsetInfo);
}

AccelKHR VulkanRaytracing::create_acceleration(VkAccelerationStructureCreateInfoKHR& accel)
{
	AccelKHR accelResult;
	// Allocating the buffer to hold the acceleration structure
	AllocatedBuffer accelBuffer = vkutils::create_buffer(_allocator, accel.size,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	// Setting the buffer
	accelResult.buffer = accelBuffer;
	accel.buffer = accelResult.buffer._buffer;
	// Create the acceleration structure
	vkCreateAccelerationStructureKHR(_device, &accel, nullptr,
		&accelResult.accel);

	return accelResult;
}
