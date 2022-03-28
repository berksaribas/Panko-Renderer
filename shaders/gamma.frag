#version 460

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D source;

vec3 aces_film(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

void main(void) {
    vec3 texColor  = texture(source, InUv).rgb;

   texColor  *= 1;  // Hardcoded Exposure Adjustment
   vec3 color = aces_film(texColor);
 
   vec3 retColor = pow(color, vec3(1/2.2));

    outFragColor = vec4(retColor, 1.0);
}