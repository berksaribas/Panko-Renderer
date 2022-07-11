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
	vec4 clearColor;
	vec4 cameraPos;
	vec4 lightPos;
	vec4 lightColor;
	vec2 lightmapInputSize;
	vec2 lightmapTargetSize;
	vec2 jitter;
	vec2 prevJitter;
	int indirectDiffuse;
	int indirectSpecular;
	int useStochasticSpecular;
	int glossyDenoise;
	int frameCount;
	int glossyFrameCount;
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
	int svdCoeffCount;
	int svdCoeffOffset;
	int projectionMatrixOffset;
	int reconstructionMatrixOffset;
};

struct PrecalculateReceiverMatrixConfig {
	int clusterProbeCount;
	int totalProbeCount;
	int basisFunctionCount;
	int rayCount;
	int receiverOffset;
	int batchOffset;
	int batchSize;
	int maxProbesPerCluster;
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

// Ray Tracing Gems chapter 6 and https://github.com/yuphin/Lumen
vec3 offset_ray(const vec3 p, const vec3 n) {
    const float origin = 1.0f / 32.0f;
    const float float_scale = 0.001;
    const float int_scale = 256.0f;
    ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

#if 0
    return vec3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
#else
    return vec3(p.x + float_scale * n.x, p.y + float_scale * n.y,
                p.z + float_scale * n.z);
#endif
}

// Generates a seed for a random number generator from 2 inputs plus a backoff
uint initRand(uint val0, uint val1)
{
	uint v0 = val0, v1 = val1, s0 = 0;

	for (uint n = 0; n < 16; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}
	return v0;
}

// Takes our seed, updates it, and returns a pseudorandom float in [0..1]
float nextRand(inout uint s)
{
	s = (1664525u * s + 1013904223u);
	return float(s & 0x00FFFFFF) / float(0x01000000);
}

float halton(int index, int base)
{
	float result = 0;
	float f = 1;
	while (index > 0) {
		f /= base;
		result += f * (index % base);
		index = int(floor(index / float(base)));
	}
	return result;
}

vec3 getPerpendicularVector(vec3 u)
{
	vec3 a = abs(u);
	uint xm = ((a.x - a.y)<0 && (a.x - a.z)<0) ? 1 : 0;
	uint ym = (a.y - a.z)<0 ? (1 ^ xm) : 0;
	uint zm = 1 ^ (xm | ym);
	return cross(u, vec3(xm, ym, zm));
}

vec3 getCosHemisphereSample(int index, vec2 offset, vec3 hitNorm)
{
	// Generate 2 uniformly-distributed values in range 0 to 1
	float u = halton(index, 3);
	float v = halton(index, 5);
	// Apply per-texel randomization
	u = fract(u + offset.x);
	v = fract(v + offset.y);

	vec2 randVal = vec2(u, v);

	// Cosine weighted hemisphere sample from RNG
	vec3 bitangent = getPerpendicularVector(hitNorm);
	vec3 tangent = cross(bitangent, hitNorm);
	float r = sqrt(randVal.x);
	float phi = 2.0f * 3.14159265f * randVal.y;

	// Get our cosine-weighted hemisphere lobe sample direction
	return tangent * (r * cos(phi).x) + bitangent * (r * sin(phi)) + hitNorm.xyz * sqrt(1 - randVal.x);
}

#endif

#endif