#pragma once

#include <vk_engine.h>

namespace vkutil {
	bool load_shader_module(VulkanEngine& engine, const char* filePath, VkShaderModule* outShaderModule);
}