#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;
layout (location = 3) in vec2 vLightmapCoord;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outPrevPosition;
layout (location = 2) flat out int outMaterialId;
layout (location = 3) out vec3 outNormal;
layout (location = 4) out vec2 outTexCoord;
layout (location = 5) out vec2 outLightmapCoord;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };

layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{

	GPUObjectData objects[];
} objectBuffer;

void main()
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	vec4 modelPos = modelMatrix * vec4(vPosition, 1.0f);

	gl_Position = cameraData.viewproj * modelPos;

	outPosition = gl_Position;
	outPrevPosition = cameraData.prevViewproj * modelPos;

	outMaterialId = objectBuffer.objects[gl_BaseInstance].material_id;
	outNormal = mat3(transpose(inverse(modelMatrix))) * vNormal;
	outTexCoord = vTexCoord;
	outLightmapCoord = vLightmapCoord / cameraData.lightmapInputSize;
}