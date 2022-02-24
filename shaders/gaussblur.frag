#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform _PushConstantRay { vec3 direction; }; //x, y -> direction, z -> miplevel

layout(set = 0, binding = 0) uniform sampler2D colorSource;
layout(set = 1, binding = 0) uniform sampler2D normalSource;

const float EPSILON = 0.001;

float bilateralWeight(vec3 n1, vec3 n2, float d1, float d2, float hd1, float hd2 ) {
	if(d1 < 0 && d2 < 0) return 0;
	if(d1 < 0) return 1;

	float normalWeight = max(0, pow(dot(n1,n2), 32));
	float depthWeight = abs(d1-d2) > 0.5 ? 0 : 1;
	float hitDistanceWeight = 1 / (EPSILON + abs(hd1-hd2) / 100.0);

	return normalWeight * depthWeight * hitDistanceWeight;
}

float bilateralWeight2(vec3 n1, vec3 n2, float d1, float d2) {
	if(d1 < 0 && d2 < 0) return 0;
	if(d1 < 0) return 1;

	float normalWeight = max(0, pow(dot(n1,n2), 32));
	float depthWeight = abs(d1-d2) > 0.5 ? 0 : 1;

	return normalWeight * depthWeight;
}

vec4 bilateralBlur(vec2 uv, vec2 resolution) {
	vec4 color = vec4(0.0);
	vec2 off1 = vec2(1) * direction.xy;
	vec2 off2 = vec2(2) * direction.xy;
	vec2 off3 = vec2(3) * direction.xy;

	vec4 mainColor = textureLod(colorSource, uv, direction.z);
	vec3 mainNormal = textureLod(normalSource, uv, direction.z ).xyz;
	float mainDepth = textureLod(normalSource, uv, direction.z ).w;

	float totalWeight = 0;
	{
		vec4 sampled = textureLod(colorSource, uv, direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv, direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv, direction.z).w;

		float weight = 0.1964825501511404 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	{
		vec4 sampled = textureLod(colorSource, uv + (off1 / resolution), direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv + (off1 / resolution), direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv + (off1 / resolution), direction.z).w;

		float weight = 0.2969069646728344 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	{
		vec4 sampled = textureLod(colorSource, uv - (off1 / resolution), direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv - (off1 / resolution), direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv - (off1 / resolution), direction.z).w;

		float weight = 0.2969069646728344 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	{
		vec4 sampled = textureLod(colorSource, uv + (off2 / resolution), direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv + (off2 / resolution), direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv + (off2 / resolution), direction.z).w;

		float weight = 0.09447039785044732 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	{
		vec4 sampled = textureLod(colorSource, uv - (off2 / resolution), direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv - (off2 / resolution), direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv - (off2 / resolution), direction.z).w;

		float weight = 0.09447039785044732 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	{
		vec4 sampled = textureLod(colorSource, uv + (off3 / resolution), direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv + (off3 / resolution), direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv + (off3 / resolution), direction.z).w;

		float weight = 0.010381362401148057 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	{
		vec4 sampled = textureLod(colorSource, uv - (off3 / resolution), direction.z);
		vec3 sampledNormal = textureLod(normalSource, uv - (off3 / resolution), direction.z).xyz;
		float sampledDepth = textureLod(normalSource, uv - (off3 / resolution), direction.z).w;

		float weight = 0.010381362401148057 * bilateralWeight2(mainNormal, sampledNormal, mainDepth, sampledDepth);
		color += sampled * weight;
		totalWeight += weight;
	}
	if(totalWeight < 0.00001) {
		return vec4(0);
	}
	vec4 blurred = color / totalWeight;
	return blurred;
}

float gaussian_weight(float offset, float deviation)
{
    float weight = 1.0 / sqrt(2.0 * 3.14159265359 * deviation * deviation);
    weight *= exp(-(offset * offset) / (2.0 * deviation * deviation));
    return weight;
}

vec4 bilateral4x4(vec2 uv) {
	vec4 col = vec4(0.0);
    float accum = 0.0;
    float weight;

	vec2 sourceSize = textureSize(colorSource, int(direction.z));
    vec2 texelSize = vec2(1.0) / sourceSize;

	vec4 lowerNormalDepth = textureLod(normalSource, uv, direction.z + 1);

    float deviation = 1;
	const float c_halfSamplesX = 2.;
	const float c_halfSamplesY = 2.;

    for (float iy = -c_halfSamplesY; iy <= c_halfSamplesY; iy++)
    {
        for (float ix = -c_halfSamplesX; ix <= c_halfSamplesX; ix++)
        {
			if(ix == 0 || iy == 0) continue;

            float fx = gaussian_weight(ix - sign(ix) / 2, deviation);
            float fy = gaussian_weight(iy - sign(iy) / 2, deviation);

            vec2 offset = vec2(ix, iy);

            vec4 normalDepth = textureLod(normalSource, uv + offset * texelSize - sign(offset) * texelSize / 2, direction.z) ;
            weight = fx *fy * bilateralWeight2(lowerNormalDepth.xyz, normalDepth.xyz, lowerNormalDepth.w, normalDepth.w);
            col += textureLod(colorSource, uv + offset * texelSize - sign(offset) * texelSize / 2, direction.z) * weight;
            accum += weight;
        }
    }

    if(accum > 0.0000001) {
		return col / accum;
	}
	else {
		return vec4(0); //todo figure this out
	}
}

void main(void) {
	if(direction.x == 1 ) {
		outFragColor = bilateralBlur(InUv, textureSize(colorSource, int(direction.z)));
		//outFragColor = textureLod(colorSource, InUv, direction.z);
	}
	else if(direction.y == 1) {
		
	
		vec2 uv = InUv;
		vec2 targetSize = textureSize(colorSource, int(direction.z) + 1);
		vec2 texelSize = vec2(1.0) / targetSize;
		vec2 f = fract( uv * targetSize );
		uv += ( .5 - f ) * texelSize;

		outFragColor = bilateral4x4(uv);
	}
	else {
		vec2 uv = InUv;
		vec2 targetSize = textureSize(colorSource, int(direction.z) + 1);
		vec2 texelSize = vec2(1.0) / targetSize;
		vec2 f = fract( uv * targetSize );
		uv += ( .5 - f ) * texelSize;
	
		vec4 tl = textureLod(colorSource, uv + vec2(-texelSize.x, -texelSize.y) / 4, direction.z);
		vec4 tr = textureLod(colorSource, uv + vec2(texelSize.x, -texelSize.y) / 4, direction.z);
		vec4 bl = textureLod(colorSource, uv + vec2(-texelSize.x, texelSize.y) / 4, direction.z);
		vec4 br = textureLod(colorSource, uv + vec2(texelSize.x, texelSize.y) / 4, direction.z);

		vec4 totalColor = vec4(0);
		int count = 0;
		{
			float w1 = 0.25 * bilateralWeight2(tl.xyz, tl.xyz, tl.w, tl.w);
			float w2 = 0.25 * bilateralWeight2(tl.xyz, tr.xyz, tl.w, tr.w);
			float w3 = 0.25 * bilateralWeight2(tl.xyz, bl.xyz, tl.w, bl.w);
			float w4 = 0.25 * bilateralWeight2(tl.xyz, br.xyz, tl.w, br.w);
			if(w1+w2+w3+w4 > 0.00001) {
				totalColor += (w1 * tl + w2 * tr + w3 * bl + w4 * br) / (w1+w2+w3+w4);
				count++;
			}
		}
		
		if(count > 0) {
			outFragColor = totalColor / count;
		}
		else {
			outFragColor = tl;
		}

		//outFragColor =( tl + tr+ bl + br) / 4.0;
	}

	//outFragColor = blur13(source, InUv, textureSize(source, int(direction.z)));
	//outFragColor = textureLod(normalSource, InUv, direction.z);
}