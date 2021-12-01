#version 460

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D source;

void main(void) {
    vec3 color = texture(source, InUv).rgb;
    color = max(color, vec3(0));
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2)); 
    outFragColor = vec4(color, 1.0);
}