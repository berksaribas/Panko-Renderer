#ifndef COMMON_GLSL
#define COMMON_GLSL

#ifdef __cplusplus
#include <glm/glm.hpp>
// GLSL Type
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
using ivec4 = glm::ivec4;
using uint = unsigned int;
#endif

#define TEXEL_SAMPLES 8

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
	mat4 viewproj;
	mat4 viewprojInverse;
	mat4 prevViewproj;
	vec4 cameraPos;
	vec4 lightPos;
	vec4 lightColor;
	vec2 lightmapInputSize;
	vec2 lightmapTargetSize;
	int indirectDiffuse;
	int indirectSpecular;
	int useStochasticSpecular;
	int frameCount;
};

struct GPUObjectData{
	mat4 model;
	int material_id;
	float pad0, pad1, pad2;
};

#ifdef RAYTRACING
struct GPUSceneDesc {
	uint64_t vertexAddress;
	uint64_t normalAddress;
	uint64_t uvAddress;
	uint64_t lightmapUvAddress;
	uint64_t indexAddress;
};
#endif

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
	vec3 pos;
	vec3 normal;
	vec2 lightmapUv;
	vec2 texUv;
	int objectId;
};

struct GPUReceiverData {
	vec3 pos;
	int objectId;
	vec3 normal;
	float dPos;
};

struct GPUReceiverDataUV {
	vec3 pos;
	int objectId;
	vec3 normal;
	float dPos;
	ivec4 uvPad;
};

struct GPUReceiverRaycastResult {
	vec3 dir;
	float visibility;
};

struct GIConfig {
	vec2 lightmapInputSize;
	int probeCount;
	int rayCount;
	int basisFunctionCount;
	int clusterCount;
	int pcaCoefficient;
	int maxReceiversInCluster;
};

struct ClusterReceiverInfo {
	int receiverCount;
	int receiverOffset;
	int probeCount;
	int probeOffset;
};

struct PrecalculateReceiverMatrixConfig {
	int clusterProbeCount;
	int totalProbeCount;
	int basisFunctionCount;
	int rayCount;
	int receiverOffset;
	int batchOffset;
	int batchSize;
};

struct ReflectionPayload {
	vec3 color;
	float hitDistance;
	vec3 normal;
};

#ifndef __cplusplus

const float PHI = 1.61803398874989484820459;

float goldNoise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

//from: https://gamedev.stackexchange.com/questions/92015/optimized-linear-to-srgb-glsl
vec3 toLinear(vec3 sRGB)
{
	//sRGB = mat3(3.2406,-1.5372,-0.4986,0.9689,1.8759,0.0415,0.0557,-0.2040,1.0570)*sRGB;
    bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
    vec3 higher = pow((sRGB + vec3(0.055))*vec3(1./1.055), vec3(2.4));
    vec3 lower = sRGB*vec3(1./12.92);

    return mix(higher, lower, cutoff);
}

//from: https://gamedev.stackexchange.com/questions/92015/optimized-linear-to-srgb-glsl
vec3 fromLinear(vec3 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB * vec3(12.92);
	//mat3(0.4124,0.3576,0.1805,0.2126,0.7152,0.0722,0.0193,0.1192,0.9505)*
    return mix(higher, lower, cutoff);
}


vec3 world_position_from_depth(vec2 tex_coords, float ndc_depth, mat4 view_proj_inverse)
{
    // Take texture coordinate and remap to [-1.0, 1.0] range.
    vec2 screen_pos = tex_coords * 2.0 - 1.0;

    // // Create NDC position.
    vec4 ndc_pos = vec4(screen_pos, ndc_depth, 1.0);

    // Transform back into world position.
    vec4 world_pos = view_proj_inverse * ndc_pos;

    // Undo projection.
    world_pos = world_pos / world_pos.w;

    return world_pos.xyz;
}

vec3 octohedral_to_direction(vec2 e)
{
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0)
        v.xy = (1.0 - abs(v.yx)) * (step(0.0, v.xy) * 2.0 - vec2(1.0));
    return normalize(v);
}

#endif

#endif