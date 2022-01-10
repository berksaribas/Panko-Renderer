#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : enable

#include "common.glsl"

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;
layout (location = 3) in vec2 vLightmapCoord;

layout (location = 0) out vec3 outWorldPosition;
layout (location = 1) flat out int outObjectId;
layout (location = 2) out vec3 outNormal;


layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };

//all object matrices
layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{

	GPUObjectData objects[];
} objectBuffer;

void main()
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	vec4 modelPos = modelMatrix * vec4(vPosition, 1.0f);

	gl_Position = vec4((vLightmapCoord / cameraData.lightmapInputSize) * 2.0 - 1.0,0,1);

	outWorldPosition = modelPos.xyz;
	outObjectId = gl_BaseInstance;
	outNormal = mat3(transpose(inverse(modelMatrix))) * vNormal;
}
