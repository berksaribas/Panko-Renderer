#include <gi_brdf.h>
#include <vector>
#include <vk_utils.h>
#include <vk_initializers.h>
//#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void load_image(EngineData& engineData, void* data, AllocatedImage& image, VkFormat image_format, int width, int height, size_t size) {
	AllocatedBuffer stagingBuffer = vkutils::create_buffer(engineData.allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	vkutils::cpu_to_gpu(engineData.allocator, stagingBuffer, data, size);

	VkExtent3D imageExtent = { width, height, 1 };
	VkImageCreateInfo dimg_info = vkinit::image_create_info(image_format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, imageExtent);
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(engineData.allocator, &dimg_info, &dimg_allocinfo, &image._image, &image._allocation, nullptr);

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
		imageBarrier_toTransfer.image = image._image;
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
		vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, image._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

		imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		//barrier the image into the shader readable layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);

		});

	vmaDestroyBuffer(engineData.allocator, stagingBuffer._buffer, stagingBuffer._allocation);

}

void BRDF::init_images(EngineData& engineData)
{
	{
		size_t size = 512 * 512 * sizeof(uint16_t) * 2;
		std::vector<uint16_t> buffer(size);

		FILE* ptr;
		fopen_s(&ptr, "../../data/brdf_lut.bin", "rb");
		fread_s(buffer.data(), size, size, 1, ptr);
		fclose(ptr);

		load_image(engineData, buffer.data(), _brdfLutImage, VK_FORMAT_R16G16_SFLOAT, 512, 512, size);
		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(VK_FORMAT_R16G16_SFLOAT, _brdfLutImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_brdfLutImageView));
	}

	{
		FILE* ptr;
		fopen_s(&ptr, "../../data/blue_noise/sobol_256_4d.png", "rb");
		int x, y, comp;

		auto data = stbi_load_from_file(ptr, &x, &y, &comp, 0);
		printf("%d\n", comp);

		load_image(engineData, data, _sobolImage, VK_FORMAT_R8G8B8A8_UNORM, x, y, x * y * comp * sizeof(uint8_t));
		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, _sobolImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_sobolImageView));

		fclose(ptr);
	}

	{
		FILE* ptr;
		fopen_s(&ptr, "../../data/blue_noise/scrambling_ranking_128x128_2d_1spp.png", "rb");
		int x, y, comp;

		auto data = stbi_load_from_file(ptr, &x, &y, &comp, 4);
		printf("%d\n", comp);

		load_image(engineData, data, _scramblingRanking1sppImage, VK_FORMAT_R8G8B8A8_UNORM, x, y, x * y * 4 * sizeof(uint8_t));
		VkImageViewCreateInfo imageViewInfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, _scramblingRanking1sppImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
		VK_CHECK(vkCreateImageView(engineData.device, &imageViewInfo, nullptr, &_scramblingRanking1sppImageView));

		fclose(ptr);
	}
}

void BRDF::init_descriptors(EngineData& engineData, SceneDescriptors& sceneDescriptors)
{
	{
		VkDescriptorImageInfo storageImageBufferInfo;
		storageImageBufferInfo.sampler = engineData.linearSampler;
		storageImageBufferInfo.imageView = _brdfLutImageView;
		storageImageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &sceneDescriptors.singleImageSetLayout, 1);
		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_brdfLutTextureDescriptor);
		VkWriteDescriptorSet textures = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _brdfLutTextureDescriptor, &storageImageBufferInfo, 0, 1);
		vkUpdateDescriptorSets(engineData.device, 1, &textures, 0, nullptr);
	}

	{
		VkDescriptorSetLayoutBinding data[2]{ vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0),
			vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1) };
		VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(data, 2);
		vkCreateDescriptorSetLayout(engineData.device, &setinfo, nullptr, &_blueNoiseDescriptorSetLayout);
	}

	{
		VkDescriptorImageInfo sobolBufferInfo = { engineData.nearestSampler, _sobolImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		VkDescriptorImageInfo scramblingBufferInfo = { engineData.nearestSampler, _scramblingRanking1sppImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorset_allocate_info(engineData.descriptorPool, &_blueNoiseDescriptorSetLayout, 1);
		vkAllocateDescriptorSets(engineData.device, &allocInfo, &_blueNoiseDescriptor);

		VkWriteDescriptorSet textures[2] = {
			vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _blueNoiseDescriptor, &sobolBufferInfo, 0, 1),
			vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _blueNoiseDescriptor, &scramblingBufferInfo, 1, 1),
		};

		vkUpdateDescriptorSets(engineData.device, 2, textures, 0, nullptr);
	}
}

void BRDF::cleanup(EngineData& engineData)
{
	vkDestroyImageView(engineData.device, _brdfLutImageView, nullptr);
	vmaDestroyImage(engineData.allocator, _brdfLutImage._image, _brdfLutImage._allocation);

	vkDestroyImageView(engineData.device, _scramblingRanking1sppImageView, nullptr);
	vmaDestroyImage(engineData.allocator, _scramblingRanking1sppImage._image, _scramblingRanking1sppImage._allocation);

	vkDestroyImageView(engineData.device, _sobolImageView, nullptr);
	vmaDestroyImage(engineData.allocator, _sobolImage._image, _sobolImage._allocation);
}
