#version 460
#extension GL_EXT_debug_printf : enable
layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

layout (location = 0) out vec2 texCoord;
layout (location = 1) flat out int material_id;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outLightVec;
layout (location = 4) out vec3 outLightColor;
layout (location = 5) out vec4 outFragPos;

layout(set = 0, binding = 0) uniform  CameraBuffer{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 lightPos;
	vec4 lightColor;
} cameraData;

layout (set = 0, binding = 1) uniform ShadowMapData 
{
	mat4 depthMVP;
    float positiveExponent;
    float negativeExponent;
    float LightBleedingReduction;
    float VSMBias;
} shadowMapData;

struct ObjectData{
	mat4 model;
	int material_id;
	float pad0, pad1, pad2;
};

//all object matrices
layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{

	ObjectData objects[];
} objectBuffer;

void main()
{
	debugPrintfEXT("My float is %f", 3.1415f);
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	vec4 modelPos = modelMatrix * vec4(vPosition, 1.0f);
	gl_Position = cameraData.viewproj * modelPos;
	texCoord = vTexCoord;
	material_id = objectBuffer.objects[gl_BaseInstance].material_id;

	//outNormal = vec3(modelMatrix * vec4(vNormal, 1.0f));
	outNormal = vNormal;
	outLightVec = cameraData.lightPos.xyz;
	outLightColor = cameraData.lightColor.xyz;
	outFragPos = modelPos;
}
