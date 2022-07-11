#pragma once
#include "vk_types.h"
#include <vector>
#include <string>
#include <glm/glm.hpp>

typedef union SDL_Event;
struct SDL_Window;
class Shadow;
class GlossyDenoise;
struct GPUCameraData;
struct GPUBasicMaterialData;
struct GltfNode;

//TODO: move to somewhere else
struct CameraConfig {
	float fov = 45;
	float speed = 0.1f;
	float rotationSpeed = 0.05f;
	float nearPlane = 0.1f;
	float farPlane = 1000.0f;
};

struct Camera {
	glm::vec3 pos;
	glm::vec3 rotation;
};

struct ScreenshotSaveData {
	Camera camera;
	glm::vec4 lightPos;
};

struct EditorSettings {
	bool enableGi = false;
	bool showProbes = false;
	bool showProbeRays = false;
	bool showReceivers = false;
	bool showSpecificReceiver = false;
	int specificCluster = 1;
	int specificReceiver = 135;
	int specificReceiverRaySampleCount = 10;
	bool showSpecificProbeRays = false;
	bool probesEnabled[300];
	int renderMode = 0;

	int selectedPreset = -1;
	char customName[128];
	bool screenshot = false;

	bool useRealtimeRaycast = true;
	bool enableDenoise = true;
	bool enableGroundTruthDiffuse = false;

	bool showFileList = false;
	std::vector<std::string> load_files;
	int selected_file = 0;

	bool useSceneCamera = false;

	Handle<Vrg::Bindable> selectedRenderBinding;
};

class Editor {
public:
	void initialize(EngineData& engineData, SDL_Window* window, VkFormat targetFormat);
	void retrieve_input(SDL_Event* event);
	
	
	void prepare(EngineData& engineData);
	void prepare_camera_settings(EngineData& engineData, GPUCameraData& camData, CameraConfig& camConfig, bool sceneCameraAvailable);
	void prepare_debug_settings(EngineData& engineData);
	void prepare_performance_settings(EngineData& engineData);
	void prepare_material_settings(EngineData& engineData, SceneData& sceneData, GPUBasicMaterialData* materials, int count);
	void prepare_object_settings(EngineData& engineData, GltfNode* node, int count);
	void prepare_renderer_settings(EngineData& engineData, GPUCameraData& camData, Shadow& shadow, GlossyDenoise& glossyDenoise, int& frameNumber);
	void prepare_screenshot_loader_settings(GPUCameraData& camData, Camera& camera);

	void render(VkCommandBuffer cmd);
	void destroy(EngineData& engineData);

	EditorSettings editorSettings;
private:
	VkDescriptorPool _imguiPool;
	SDL_Window* _window;
};