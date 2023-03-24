#pragma once

#include "memory/slice.h"
#include <string_view>
#include <unordered_map>
#include <vector>

class ShaderManager
{
public:
    void initialize();
    void destroy();

    bool get_spirv(std::string_view glslPath, Slice<std::string_view> defines,
                   Slice<uint32_t>& output);
    void clear_cache();

private:
    std::unordered_map<size_t, std::vector<uint32_t>> m_shaders;
};