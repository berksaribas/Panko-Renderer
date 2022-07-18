#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../common.glsl"

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;
layout (location = 3) in vec2 vLightmapCoord;

layout (location = 0) out vec2 texCoord;
layout (location = 1) flat out int material_id;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outLightVec;
layout (location = 4) out vec3 outLightColor;
layout (location = 5) out vec4 outFragPos;
layout (location = 6) out vec2 outLightmapCoord;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };

layout(set = 0, binding = 1) uniform _ShadowMapData { GPUShadowMapData shadowMapData; };

//all object matrices
layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{

	GPUObjectData objects[];
} objectBuffer;

void main()
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	vec4 modelPos = modelMatrix * vec4(vPosition, 1.0f);

	gl_Position = vec4((vLightmapCoord / cameraData.lightmapInputSize) * 2.0 - 1.0,0,1);

	texCoord = vTexCoord;
	material_id = objectBuffer.objects[gl_BaseInstance].material_id;

	outNormal = mat3(transpose(inverse(modelMatrix))) * vNormal;
	outLightVec = cameraData.lightPos.xyz;
	outLightColor = cameraData.lightColor.xyz;
	outFragPos = modelPos;
	outLightmapCoord = vLightmapCoord / cameraData.lightmapInputSize;
}
