#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D source;
layout(push_constant) uniform _PushConstantRay { ivec2 size; };

void main(void) {
    vec4 c = texture(source, InUv);
    vec2 pixelOffset = vec2(1.0) / textureSize(source, 0);
    c = c.a>0.0? c : texture(source, InUv - pixelOffset);
    c = c.a>0.0? c : texture(source, InUv + vec2(0, -pixelOffset.y));
    c = c.a>0.0? c : texture(source, InUv + vec2(pixelOffset.x, -pixelOffset.y));
    c = c.a>0.0? c : texture(source, InUv + vec2(-pixelOffset.x, 0));
    c = c.a>0.0? c : texture(source, InUv + vec2(pixelOffset.x, 0));
    c = c.a>0.0? c : texture(source, InUv + vec2(-pixelOffset.x, pixelOffset.y));
    c = c.a>0.0? c : texture(source, InUv + vec2(0, pixelOffset.y));
    c = c.a>0.0? c : texture(source, InUv + pixelOffset);
    outFragColor = c;
}