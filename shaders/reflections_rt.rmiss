#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT ReflectionPayload payload;
layout(location = 1) rayPayloadInEXT vec3 reflectionColor;

void main()
{
    payload.color = vec3(0);
    payload.hitDistance = 100;
    payload.normal = vec3(0);
    reflectionColor = vec3(0);
}