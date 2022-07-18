#version 460
#extension GL_EXT_debug_printf : enable

layout (location = 0) in vec3 inWorldPosition;
layout (location = 1) flat in int inObjectId;
layout (location = 2) in vec3 inNormal;

layout(location = 0) out vec4 gbufferPositionObjectId;
layout(location = 1) out vec4 gbufferNormal;

void main()
{
	gbufferPositionObjectId.xyz = inWorldPosition;
	gbufferPositionObjectId.w = inObjectId;
	gbufferNormal.xyz = normalize(inNormal);

	vec3 dUV1 = max(abs(dFdx(inWorldPosition)), abs(dFdy(inWorldPosition)));
	float dPos = max(max(dUV1.x, dUV1.y), dUV1.z);
	dPos = dPos * sqrt(2.0);
	gbufferNormal.w = dPos;
}