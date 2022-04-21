#version 460

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D source;

const float GAMMA     = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;

vec3 linearTosRGB(vec3 color)
{
  return pow(color, vec3(INV_GAMMA));
}

vec3 sRGBToLinear(vec3 srgbIn)
{
  return vec3(pow(srgbIn.xyz, vec3(GAMMA)));
}

vec3 aces_film(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// http://www.thetenthplanet.de/archives/5367
// https://github.com/nvpro-samples/vk_raytrace/blob/40a6bb06fbadca968c349f4f830847c86b5b1cf9/shaders/post.frag
vec3 dither(vec3 linear_color, vec3 noise, float quant)
{
  vec3 c0    = floor(linearTosRGB(linear_color) / quant) * quant;
  vec3 c1    = c0 + quant;
  vec3 discr = mix(sRGBToLinear(c0), sRGBToLinear(c1), noise);
  return mix(c0, c1, lessThan(discr, linear_color));
}

uvec3 pcg3d(uvec3 v)
{
  v = v * 1664525u + uvec3(1013904223u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> uvec3(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

void main(void) {
    vec3 texColor  = texture(source, InUv).rgb;

    texColor  *= 1;  // Hardcoded Exposure Adjustment
    vec3 color = aces_film(texColor);
   
    //uvec3 r = pcg3d(uvec3(gl_FragCoord.xy, 0));
    //vec3 noise = uintBitsToFloat(0x3f800000 | (r >> 9)) - 1.0f;
    //color = dither(sRGBToLinear(color), noise, 1. / 255.);
 
    outFragColor = vec4(color, 1.0);
}