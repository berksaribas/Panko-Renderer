#pragma once
#include "vk_types.h"

struct CameraConfig;

struct SuperResolutionData {
	float jitterX;
	float jitterY;
	float mipBias;
};

class SuperResolution {
public:
	void initialize(EngineData& engineData, VkExtent2D renderSize, VkExtent2D displaySize);
	void get_data(SuperResolutionData* data);
	void dispatch(EngineData& engineData, VkCommandBuffer cmd, Handle<Vrg::Bindable> colorInput, Handle<Vrg::Bindable> depthInput, Handle<Vrg::Bindable> motionInput, CameraConfig* cameraConfig, float deltaTime);
	void destroy();
	Handle<Vrg::Bindable> outputBindable;
private:
	bool isInitialized;
	AllocatedImage outputImage;
};