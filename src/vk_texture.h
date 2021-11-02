#pragma once

#include <vk_types.h>
#include <vk_engine.h>

namespace vkutil {
	bool load_image_from_memory(VulkanEngine& engine, void* pixels, int width, int height, AllocatedImage& outImage, uint32_t& outMipLevels);
	
}