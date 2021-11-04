#pragma once

#include <vk_engine.h>

namespace vkutil {
	bool load_shader_module(VkDevice device, const char* filePath, VkShaderModule* outShaderModule);
}