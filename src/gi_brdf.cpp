#include <gi_brdf.h>
#include <vector>
#include <vk_utils.h>
#include <vk_initializers.h>

void BRDF::init_images(EngineData& engineData)
{
	size_t size = 512 * 512 * sizeof(uint16_t) * 2;
	std::vector<uint16_t> buffer(size);

	FILE* ptr;
	fopen_s(&ptr, "../../data/brdf_lut.bin", "rb");
	fread_s(buffer.data(), size, size, 1, ptr);
	fclose(ptr);

	VkFormat image_format = VK_FORMAT_R16G16_SFLOAT;
	AllocatedBuffer stagingBuffer = vkutils::create_buffer(engineData.allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	vkutils::cpu_to_gpu(engineData.allocator, stagingBuffer, buffer.data(), size);

	VkExtent3D imageExtent = { 512, 512, 1 };
	VkImageCreateInfo dimg_info = vkinit::image_create_info(image_format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, imageExtent);
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &_brdfLutImage._image, &_brdfLutImage._allocation, nullptr);

	vkutils::immediate_submit(&engineData, [&](VkCommandBuffer cmd) {
		VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		VkImageMemoryBarrier imageBarrier_toTransfer = {};
		imageBarrier_toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.image = _brdfLutImage._image;
		imageBarrier_toTransfer.subresourceRange = range;

		imageBarrier_toTransfer.srcAccessMask = 0;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		//barrier the image into the transfer-receive layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);

		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;

		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = imageExtent;

		//copy the buffer into the image
		vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, _brdfLutImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

		imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		//barrier the image into the shader readable layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
			
	});

	vmaDestroyBuffer(engineData.allocator, stagingBuffer._buffer, stagingBuffer._allocation);

	VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(image_format, _brdfLutImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_brdfLutImageView));
}

void BRDF::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	VkDescriptorImageInfo storageImageBufferInfo;
	storageImageBufferInfo.sampler = engineData.linearSampler;
	storageImageBufferInfo.imageView = _brdfLutImageView;
	storageImageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);
	vkAllocateDescriptorSets(engineData.device, &allocInfo, &_brdfLutTextureDescriptor);
	VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _brdfLutTextureDescriptor, &storageImageBufferInfo, 0, 1);
	vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
}

void BRDF::cleanup(EngineData& engineData)
{
	vkDestroyImageView(engineData.device, _brdfLutImageView, nullptr);
	vmaDestroyImage(engineData.allocator, _brdfLutImage._image, _brdfLutImage._allocation);
}
