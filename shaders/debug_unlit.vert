#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vColor;

layout (location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };

void main()
{
	gl_PointSize = 10.0f;
	gl_Position = cameraData.viewproj * vec4(vPosition, 1.0);
	outColor = vColor;
}