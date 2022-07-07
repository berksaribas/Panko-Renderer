#include "vk_timer.h"
#include <inttypes.h>

void VulkanTimer::start_recording(EngineData& engineData, VkCommandBuffer cmd, std::string name)
{
    vkResetQueryPool(engineData.device, engineData.queryPool, count * 2, 2);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, engineData.queryPool, count * 2);
    names[count] = name;
}

void VulkanTimer::stop_recording(EngineData& engineData, VkCommandBuffer cmd)
{
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, engineData.queryPool, count * 2 + 1);
    count++;
}

void VulkanTimer::get_results(EngineData& engineData)
{
    auto res = vkGetQueryPoolResults(engineData.device, engineData.queryPool, 0, count * 2, sizeof(uint64_t) * count * 2, times, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);



    result = res == VK_SUCCESS;
}

void VulkanTimer::reset()
{
    count = 0;
}
