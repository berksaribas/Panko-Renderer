#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform _PushConstantRay { vec3 direction; }; //x, y -> direction, z -> miplevel
layout(set = 0, binding = 0) uniform sampler2D source;


vec4 sampleColor(sampler2D image, vec2 uv, vec2 offset) {
	vec4 main = textureLod(source, uv, direction.z);

	vec2 target = uv + offset;
	target = clamp(target, vec2(0), vec2(1));
	vec4 color = textureLod(source, target, direction.z);

	if(int(color.w) != int(main.w)) {
		return main;
	}
	else return color;
}

float bilateralWeight(float w1, float w2) {
return 1;
	if(10000 * abs(w1 - w2) > 1) {
		return 0;
	}
	else {
		return 1 - (10000 * abs(w1 - w2));
	}
}

vec4 blur13(sampler2D image, vec2 uv, vec2 resolution) {
	vec4 color = vec4(0.0);
	vec2 off1 = vec2(1) * direction.xy;
	vec2 off2 = vec2(2) * direction.xy;
	vec2 off3 = vec2(3) * direction.xy;

	float ogw = fract(sampleColor(image, uv, vec2(0)).w);
	float distanceOnlyBlur = 0;
	float totalWeight = 0;
	{
		vec4 sampled = sampleColor(image, uv, vec2(0));
		float weight = 0.1964825501511404 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.1964825501511404;
		totalWeight += weight;
	}
	{
		vec4 sampled = sampleColor(image, uv, (off1 / resolution));
		float weight = 0.2969069646728344 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.2969069646728344;
		totalWeight += weight;
	}
	{
		vec4 sampled = sampleColor(image, uv, -(off1 / resolution));
		float weight = 0.2969069646728344 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.2969069646728344;
		totalWeight += weight;
	}
	{
		vec4 sampled = sampleColor(image, uv, (off2 / resolution));
		float weight = 0.09447039785044732 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.09447039785044732;
		totalWeight += weight;
	}
	{
		vec4 sampled = sampleColor(image, uv, -(off2 / resolution));
		float weight = 0.09447039785044732 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.09447039785044732;
		totalWeight += weight;
	}
	{
		vec4 sampled = sampleColor(image, uv, (off3 / resolution));
		float weight = 0.010381362401148057 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.010381362401148057;
		totalWeight += weight;
	}
	{
		vec4 sampled = sampleColor(image, uv, -(off3 / resolution));
		float weight = 0.010381362401148057 * bilateralWeight(ogw, fract(sampled.w));
		color += sampled * weight;
		distanceOnlyBlur += sampled.w  * 0.010381362401148057;
		totalWeight += weight;
	}

	//color.w = sampleColor(image, uv, vec2(0)).w;
	//color.w += 0.001;
	vec4 blurred = color / totalWeight;
	blurred.w = distanceOnlyBlur;
	return blurred;
}

vec4 customBilinear(vec2 uv, float mip) {
	vec2 originSize = textureSize(source, int(direction.z));
	vec2 targetSize = textureSize(source, int(direction.z) + 1);
    vec2 texelSize = vec2(1.0) / targetSize;

    vec2 f = fract( uv * targetSize );
    uv += ( .5 - f ) * texelSize;    // move uv to texel centre

	// is it /2 or /4? originally /4 was my idea but it seems like that fails?
    vec4 tl = blur13(source, uv + vec2(-texelSize.x, texelSize.y) / 4, originSize);
    vec4 tr = blur13(source, uv + vec2(texelSize.x, texelSize.y) / 4, originSize);
    vec4 bl = blur13(source, uv + vec2(-texelSize.x, -texelSize.y) / 4, originSize);
    vec4 br = blur13(source, uv + vec2(texelSize.x, -texelSize.y) / 4, originSize);

	int nums[32];
	for(int i = 0; i < 32; i++) {
		nums[i] = 99;
	}
	nums[int(tl.w) + 1]--;
	nums[int(tr.w) + 1]--;
	nums[int(bl.w) + 1]--;
	nums[int(br.w) + 1]--;

	int selectedMaterial = -1;
	int minCount = 100;

	for(int i = 1; i < 32; i++) {
		if(nums[i] < minCount) {
			selectedMaterial = i -1;
			minCount = nums[i];
		}
	}

	if(selectedMaterial > -1) {
		vec4 colors = vec4(0);
		int count = 0;
		if(int(tl.w) == selectedMaterial) {
			colors += tl;
			count++;
		}
		if(int(tr.w) == selectedMaterial) {
			colors += tr;
			count++;
		}
		if(int(bl.w) == selectedMaterial) {
			colors += bl;
			count++;
		}
		if(int(br.w) == selectedMaterial) {
			colors += br;
			count++;
		}
		if(count == 0) {
			return vec4(0, 0, 0, -1);
		}
		colors = colors / count;
		//colors.w = selectedMaterial + fract(colors.w);

		return colors;
	}
	else {
		return vec4(0);
	}
}

void main(void) {
	if(direction.x > 0) {
		outFragColor = blur13(source, InUv, textureSize(source, int(direction.z)));
	}
	else {
		outFragColor = customBilinear(InUv, int(direction.z));
	}
	//outFragColor = blur13(source, InUv, textureSize(source, int(direction.z)));
	//outFragColor = sampleColor(source, InUv, vec2(0));
}