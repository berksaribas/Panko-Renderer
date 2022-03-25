#include <vk_types.h>
#include <string>

class VulkanTimer {
public:
	void start_recording(EngineData& engineData, VkCommandBuffer cmd, std::string name);
	void stop_recording(EngineData& engineData, VkCommandBuffer cmd);
	void get_results(EngineData& engineData);
private:
	uint64_t times[64];
	std::string names[64];
	int count = 0;
};