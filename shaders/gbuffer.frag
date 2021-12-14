#version 460

layout (location = 0) in vec3 inWorldPosition;
layout (location = 1) flat in int inMaterialId;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in vec2 inLightmapCoord;

layout(location = 0) out vec4 gbufferPositionMaterial;
layout(location = 1) out vec4 gbufferNormal;
layout(location = 2) out vec4 gbufferUV;

void main()
{
	gbufferPositionMaterial.xyz = inWorldPosition;
	gbufferPositionMaterial.w = inMaterialId;

	gbufferNormal.xyz = inNormal;

	gbufferUV.xy = inTexCoord;
	gbufferUV.zw = inLightmapCoord;
}