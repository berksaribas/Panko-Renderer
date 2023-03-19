#pragma once

#include <string>
#include <vk_types.h>

class VulkanTimer
{
public:
    void start_recording(EngineData& engineData, VkCommandBuffer cmd, std::string name);
    void stop_recording(EngineData& engineData, VkCommandBuffer cmd);
    void get_results(EngineData& engineData);
    void reset();
    uint64_t times[512];
    std::string names[512];
    int count = 0;
    bool result;
};