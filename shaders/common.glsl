struct ShadowMapData 
{
	mat4 depthMVP;
    float positiveExponent;
    float negativeExponent;
    float LightBleedingReduction;
    float VSMBias;
};

struct BasicMaterialData {
    vec4 base_color;
    vec3 emissive_color;
    float metallic_factor;
    float roughness_factor;
    int texture;
    int normal_texture;
    int metallic_roughness_texture;
};

struct CameraData {
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 lightPos;
	vec4 lightColor;
};

struct ObjectData{
	mat4 model;
	int material_id;
	float pad0, pad1, pad2;
};

struct SceneDesc {
	uint64_t vertexAddress;
	uint64_t normalAddress;
	uint64_t uvAddress;
	uint64_t indexAddress;
};

struct MeshInfo {
	uint indexOffset;
	uint vertexOffset;
	int materialIndex;
	int _pad;
};

struct ProbeRaycastResult {
	vec4 worldPos;
	int objectId;
	float u, v;
	int pad_;
};

struct HitPayload
{
	int objectId;
	vec3 pos;
	vec2 uv;
};

const float PHI = 1.61803398874989484820459;

float goldNoise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}