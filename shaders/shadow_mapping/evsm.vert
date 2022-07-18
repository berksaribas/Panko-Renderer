#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../common.glsl"

layout (location = 0) in vec3 vPosition;

layout (location = 0) out vec4 outPosition;

layout (set = 0, binding = 0) uniform _ShadowMapData {GPUShadowMapData shadowMapData;};

//all object matrices
layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{

	GPUObjectData objects[];
} objectBuffer;

void main()
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	gl_Position = shadowMapData.depthMVP * modelMatrix * vec4(vPosition, 1.0);
	outPosition = gl_Position;
}