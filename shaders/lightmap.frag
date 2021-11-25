#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

//shader input
layout (location = 0) in vec2 texCoord;
layout (location = 1) flat in int material_id;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inLightVec;
layout (location = 4) in vec3 inLightColor;
layout (location = 5) in vec4 inFragPos;
layout (location = 6) in vec2 inLightmapCoord;

//output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform _ShadowMapData { GPUShadowMapData shadowMapData; };

layout(set = 2, binding = 0) uniform sampler2D[] textures;
layout(set = 4, binding = 0) uniform sampler2D shadowMap;
layout(set = 5, binding = 0) uniform sampler2D indirectLightmap;

//all object matrices
layout(std140,set = 3, binding = 0) readonly buffer MaterialBuffer{

	GPUBasicMaterialData materials[];
} materialBuffer;

float linstep(float minv, float maxv, float v)
{
    return maxv == minv ? 1.0 : clamp((v - minv) / (maxv - minv), 0.0, 1.0);
}

vec2 warp_depth(float depth, vec2 exponents)
{
    // Rescale depth into [-1, 1]
    depth = 2.0f * depth - 1.0f;
    float pos =  exp( exponents.x * depth);
    float neg = -exp(-exponents.y * depth);
    return vec2(pos, neg);
}

float reduce_light_bleeding(float pMax, float amount)
{
  // Remove the [0, amount] tail and linearly rescale (amount, 1].
   return linstep(amount, 1.0f, pMax);
}

float chebyshev_upper_bound(vec2 moments, float mean, float minVariance,
                          float lightBleedingReduction)
{
    // Compute variance
    float variance = moments.y - (moments.x * moments.x);
    variance = max(variance, minVariance);

    // Compute probabilistic upper bound
    float d = mean - moments.x;
    float pMax = variance / (variance + (d * d));

    pMax = reduce_light_bleeding(pMax, lightBleedingReduction);

    // One-tailed Chebyshev
    return (mean <= moments.x ? 1.0f : pMax);
}

float sample_shadow_map_evsm(in vec4 shadowPos)
{
    vec2 exponents = vec2(shadowMapData.positiveExponent, shadowMapData.negativeExponent);
    vec2 warpedDepth = warp_depth(shadowPos.z, exponents);

    vec4 occluder = texture(shadowMap, shadowPos.st);

    // Derivative of warping at depth
    vec2 depthScale = shadowMapData.VSMBias * 0.01f * exponents * warpedDepth;
    vec2 minVariance = depthScale * depthScale;

    float posContrib = chebyshev_upper_bound(occluder.xz, warpedDepth.x, minVariance.x, shadowMapData.LightBleedingReduction);
    float negContrib = chebyshev_upper_bound(occluder.yw, warpedDepth.y, minVariance.y, shadowMapData.LightBleedingReduction);
    return min(posContrib, negContrib);
}

float textureProj(vec4 shadowCoord, vec2 off)
{
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
		float dist = texture( shadowMap, shadowCoord.st + off ).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z ) 
		{
			shadow = 0;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc)
{
    ivec2 texDim = textureSize(shadowMap, 0);
    float scale = 1.5;
    float dx = scale * 1.0 / float(texDim.x);
    float dy = scale * 1.0 / float(texDim.y);

    float shadowFactor = 0.0;
    int count = 0;
    int range = 1;
    
    for (int x = -range; x <= range; x++)
    {
        for (int y = -range; y <= range; y++)
        {
            shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
            count++;
        }
    
    }
    return shadowFactor / count;
}

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

void main()
{
    vec3 color = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materialBuffer.materials[material_id].emissive_color;

    if(emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0) {
        outFragColor = vec4(emissive_color, 1.0f);
    }
    else {
        vec4 shadowPos = biasMat * shadowMapData.depthMVP * inFragPos;
        //float shadow = textureProj(shadowPos / shadowPos.w, vec2(0.0));
        float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);
        //float shadow = filterPCF(shadowPos / shadowPos.w);

        vec3 N = normalize(inNormal);
        vec3 L = normalize(inLightVec);

	    vec3 diffuse = max(dot(N, L), 0) * inLightColor * color;

        outFragColor = vec4(diffuse * shadow, 1.0f) + texture(indirectLightmap, inLightmapCoord).rgba;
    }
}
