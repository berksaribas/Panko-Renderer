#include "vk_timer.h"
#include <inttypes.h>

void VulkanTimer::start_recording(EngineData& engineData, VkCommandBuffer cmd, std::string name)
{
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, engineData.queryPool, count * 2);
    names[count] = name;
}

void VulkanTimer::stop_recording(EngineData& engineData, VkCommandBuffer cmd)
{
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, engineData.queryPool, count * 2 + 1);
    count++;
}

void VulkanTimer::get_results(EngineData& engineData)
{
    VkResult result = vkGetQueryPoolResults(engineData.device, engineData.queryPool, 0, count * 2, sizeof(uint64_t) * count * 2, times, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    
    if (result == VK_SUCCESS)
    {
        for (int i = 0; i < count; i++) {
            printf("[%s]: %f ms\n", names[i].c_str(), (times[i * 2 + 1] - times[i * 2]) / 1000000.0);
        }
    }

    vkResetQueryPool(engineData.device, engineData.queryPool, 0, count * 2);
    count = 0;
}
