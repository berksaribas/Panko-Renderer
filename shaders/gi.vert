#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;
layout (location = 3) in vec2 vLightmapCoord;

layout (location = 0) out vec2 outTexCoord;
layout (location = 1) flat out int outMaterialId;
layout (location = 2) out vec2 outLightmapCoord;
layout (location = 3) out vec3 outNormal;
layout (location = 4) out vec4 outWorldPosition;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };

layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{

	GPUObjectData objects[];
} objectBuffer;

void main()
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	vec4 modelPos = modelMatrix * vec4(vPosition, 1.0f);

	gl_Position = cameraData.viewproj * modelPos;

	outTexCoord = vTexCoord;
	outMaterialId = objectBuffer.objects[gl_BaseInstance].material_id;
	outLightmapCoord = vLightmapCoord / cameraData.lightmapInputSize;
	outNormal = mat3(transpose(inverse(modelMatrix))) * vNormal;
	outWorldPosition = modelPos;
}