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

	float normalWeight = max(0, pow(dot(n1,n2), 2)) > 0.1 ? 1 : 0;
	float depthWeight = abs(d1-d2) > 0.5 ? 0 : 1;
	float hitDistanceWeight = 1 / (EPSILON + abs(hd1-hd2) / 100.0);

	return normalWeight * depthWeight * hitDistanceWeight;
}

float bilateralWeight2(vec3 n1, vec3 n2, float d1, float d2) {
	if(d1 < 0 && d2 < 0) return 0;
	if(d1 < 0) return 1;

	float normalWeight = max(0, pow(dot(n1,n2), 2)) > 0.5 ? 1 : 0;
	float depthWeight = abs(d1-d2) > 0.5 ? 0 : 1;

	return normalWeight * depthWeight;
}

vec4 bilateralBlur(vec2 uv, vec2 resolution) {
	vec4 color = vec4(0.0);
	vec2 off1 = vec2(1) * direction.xy;
	vec2 off2 = vec2(2) * direction.xy;
	vec2 off3 = vec2(3) * direction.xy;

	vec4 mainColor = textureLod(colorSource, uv, direction.z);
	vec3 mainNormal = textureLod(normalSource, uv, direction.z).xyz;
	float mainDepth = textureLod(normalSource, uv, direction.z).w;

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

void main(void) {
	if(direction.x == 1 ) {
		outFragColor = bilateralBlur(InUv, textureSize(colorSource, int(direction.z)));
	}
	else if(direction.y == 1) {
		vec2 uv = InUv;
		vec2 sourceSize = textureSize(colorSource, int(direction.z));
		vec2 targetSize = textureSize(colorSource, int(direction.z) + 1);
		vec2 texelSize = vec2(1.0) / targetSize;
		vec2 f = fract( uv * targetSize );
		uv += ( .5 - f ) * texelSize;
	
		vec4 tl = bilateralBlur(uv + vec2(-texelSize.x, -texelSize.y) / 4, sourceSize);
		vec4 tr = bilateralBlur(uv + vec2(texelSize.x, -texelSize.y) / 4, sourceSize);
		vec4 bl = bilateralBlur(uv + vec2(-texelSize.x, texelSize.y) / 4, sourceSize);
		vec4 br = bilateralBlur(uv + vec2(texelSize.x, texelSize.y) / 4, sourceSize);

		vec4 ntl = textureLod(normalSource, uv + vec2(-texelSize.x, -texelSize.y) / 4, direction.z);
		vec4 ntr = textureLod(normalSource, uv + vec2(texelSize.x, -texelSize.y) / 4, direction.z);
		vec4 nbl = textureLod(normalSource, uv + vec2(-texelSize.x, texelSize.y) / 4, direction.z);
		vec4 nbr = textureLod(normalSource, uv + vec2(texelSize.x, texelSize.y) / 4, direction.z);

		float w1 = 0.25 * bilateralWeight(ntl.xyz, ntl.xyz, ntl.w, ntl.w, tl.w, tl.w);
		float w2 = 0.25 * bilateralWeight(ntl.xyz, ntr.xyz, ntl.w, ntr.w, tl.w, tr.w);
		float w3 = 0.25 * bilateralWeight(ntl.xyz, nbl.xyz, ntl.w, nbl.w, tl.w, bl.w);
		float w4 = 0.25 * bilateralWeight(ntl.xyz, nbr.xyz, ntl.w, nbr.w, tl.w, br.w);

		if(w1+w2+w3+w4 > 0.00001) {
			outFragColor = (w1 * tl + w2 * tr + w3 * bl + w4 * br) / (w1+w2+w3+w4);
		}
		else {
			outFragColor = tl;
		}
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
		}/*
		{
			float w1 = 0.25 * bilateralWeight(tr.xyz, tl.xyz, tr.w, tl.w);
			float w2 = 0.25 * bilateralWeight(tr.xyz, tr.xyz, tr.w, tr.w);
			float w3 = 0.25 * bilateralWeight(tr.xyz, bl.xyz, tr.w, bl.w);
			float w4 = 0.25 * bilateralWeight(tr.xyz, br.xyz, tr.w, br.w);
			if(w1+w2+w3+w4 > 0.00001) {
				totalColor += (w1 * tl + w2 * tr + w3 * bl + w4 * br) / (w1+w2+w3+w4);
				count++;
			}
		}
		{
			float w1 = 0.25 * bilateralWeight(bl.xyz, tl.xyz, bl.w, tl.w);
			float w2 = 0.25 * bilateralWeight(bl.xyz, tr.xyz, bl.w, tr.w);
			float w3 = 0.25 * bilateralWeight(bl.xyz, bl.xyz, bl.w, bl.w);
			float w4 = 0.25 * bilateralWeight(bl.xyz, br.xyz, bl.w, br.w);
			if(w1+w2+w3+w4 > 0.00001) {
				totalColor += (w1 * tl + w2 * tr + w3 * bl + w4 * br) / (w1+w2+w3+w4);
				count++;
			}
		}
		{
			float w1 = 0.25 * bilateralWeight(br.xyz, tl.xyz, br.w, tl.w);
			float w2 = 0.25 * bilateralWeight(br.xyz, tr.xyz, br.w, tr.w);
			float w3 = 0.25 * bilateralWeight(br.xyz, bl.xyz, br.w, bl.w);
			float w4 = 0.25 * bilateralWeight(br.xyz, br.xyz, br.w, br.w);
			if(w1+w2+w3+w4 > 0.00001) {
				totalColor += (w1 * tl + w2 * tr + w3 * bl + w4 * br) / (w1+w2+w3+w4);
				count++;
			}
		}*/
		
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