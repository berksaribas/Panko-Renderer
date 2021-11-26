#ifndef COMMON_GLSL
#define COMMON_GLSL

#ifdef __cplusplus
#include <glm/glm.hpp>
// GLSL Type
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
using uint = unsigned int;
#endif

struct GPUShadowMapData 
{
	mat4 depthMVP;
    float positiveExponent;
    float negativeExponent;
    float LightBleedingReduction;
    float VSMBias;
};

struct GPUBasicMaterialData {
    vec4 base_color;
    vec3 emissive_color;
    float metallic_factor;
    float roughness_factor;
    int texture;
    int normal_texture;
    int metallic_roughness_texture;
};

struct GPUCameraData {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 lightPos;
	vec4 lightColor;
	vec2 lightmapInputSize;
	vec2 lightmapTargetSize;
};

struct GPUObjectData{
	mat4 model;
	int material_id;
	float pad0, pad1, pad2;
};

struct GPUSceneDesc {
	uint64_t vertexAddress;
	uint64_t normalAddress;
	uint64_t uvAddress;
	uint64_t lightmapUvAddress;
	uint64_t indexAddress;
};

struct GPUMeshInfo {
	uint indexOffset;
	uint vertexOffset;
	int materialIndex;
	int _pad;
};

struct GPUProbeRaycastResult {
	vec4 worldPos;
	vec4 direction;
	vec2 lightmapUv;
	vec2 texUv;
	int objectId;
	int pad0_;
	int pad1_;
	int pad2_;
};

struct GPUHitPayload
{
	int objectId;
	vec3 pos;
	vec2 lightmapUv;
	vec2 texUv;
};

struct GPUReceiverData {
	vec3 pos;
	float pad0;
	vec3 normal;
	float pad1;
};

struct GPUReceiverRaycastResult {
	vec3 dir;
	int visibility;
};

struct GIConfig {
	vec2 lightmapInputSize;
	int probeCount;
	int rayCount;
	int basisFunctionCount;
	int clusterCount;
	int pcaCoefficient;
};

struct ClusterReceiverInfo {
	int receiverCount;
	int receiverOffset;
	float pad0;
	float pad1;
};

#ifndef __cplusplus

const float PHI = 1.61803398874989484820459;

float goldNoise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

#endif

#endif