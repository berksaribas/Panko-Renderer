#version 460
layout (location = 0) in vec3 vPosition;

layout (location = 0) out vec4 outPosition;

layout (set = 0, binding = 0) uniform ShadowMapData 
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
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	gl_Position = shadowMapData.depthMVP * modelMatrix * vec4(vPosition, 1.0);
	outPosition = gl_Position;
}