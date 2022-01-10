#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform _PushConstantRay { ivec2 size; };
layout(set = 0, binding = 0) uniform sampler2D source;

void main(void) {
    vec4 c = texture(source, InUv);
    vec2 pixelOffset = vec2(1.0) / size;
    if(c.a <= 0.0) {
        int ctr = 0;
        c = vec4(0);

        if(texture(source, InUv - pixelOffset).a > 0.0) {
            c += texture(source, InUv - pixelOffset);
            ctr++;
        }
        if(texture(source, InUv + vec2(pixelOffset.x, -pixelOffset.y)).a > 0.0) {
            c += texture(source, InUv + vec2(pixelOffset.x, -pixelOffset.y));
            ctr++;
        }
        if(texture(source, InUv + vec2(-pixelOffset.x, 0)).a > 0.0) {
            c += texture(source, InUv + vec2(-pixelOffset.x, 0));
            ctr++;
        }
        if(texture(source, InUv + vec2(pixelOffset.x, 0)).a > 0.0) {
            c += texture(source, InUv + vec2(pixelOffset.x, 0));
            ctr++;
        }
        if(texture(source, InUv + vec2(-pixelOffset.x, pixelOffset.y)).a > 0.0) {
            c += texture(source, InUv + vec2(-pixelOffset.x, pixelOffset.y));
            ctr++;
        }
        if(texture(source, InUv + vec2(0, pixelOffset.y)).a > 0.0) {
            c += texture(source, InUv + vec2(0, pixelOffset.y));
            ctr++;
        }
        if(texture(source, InUv + vec2(0, -pixelOffset.y)).a > 0.0) {
            c += texture(source, InUv + vec2(0, -pixelOffset.y));
            ctr++;
        }
        if(texture(source, InUv + pixelOffset).a > 0.0) {
            c += texture(source, InUv + pixelOffset);
            ctr++;
        }

        if(ctr == 0) {
            c = vec4(0);
        }
        else {
            c /= ctr;
        }
    }
    outFragColor = texture(source, InUv); //disabled dilation
}