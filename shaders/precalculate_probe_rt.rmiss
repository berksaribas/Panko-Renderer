#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT GPUHitPayload payload;

void main()
{
    payload.objectId = -1;
    payload.pos = vec3(-999,-999,-999);
    payload.normal = vec3(0);
    //debugPrintfEXT("         ->ray miss!\n");
}