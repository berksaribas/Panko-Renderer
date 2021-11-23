#version 460

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D source;

void main(void) {
    float gamma = 1. / 2.2;
    outFragColor = pow(texture(source, InUv).rgba, vec4(gamma));
}