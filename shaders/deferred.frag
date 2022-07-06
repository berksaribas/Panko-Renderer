#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : enable

#include "common.glsl"
#include "brdf.glsl"
#include "svgf_edge_functions.glsl"

//shader input
layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };
layout(set = 0, binding = 1) uniform _ShadowMapData { GPUShadowMapData shadowMapData; };

layout(set = 1, binding = 0) uniform sampler2D gbufferAlbedoMetallic;
layout(set = 1, binding = 1) uniform sampler2D gbufferNormalMotion;
layout(set = 1, binding = 2) uniform sampler2D gbufferRoughnessDepthCurvatureMaterial;
layout(set = 1, binding = 3) uniform sampler2D gbufferUV;
layout(set = 1, binding = 4) uniform sampler2D gbufferDepth;

layout(set = 2, binding = 0) uniform sampler2D[] textures;
layout(std140, set = 3, binding = 0) readonly buffer MaterialBuffer{ GPUBasicMaterialData materials[]; };
layout(set = 4, binding = 0) uniform sampler2D shadowMap;
layout(set = 5, binding = 0) uniform sampler2D indirectLightMap;
layout(set = 6, binding = 0) uniform sampler2D glossyReflections;
layout(set = 7, binding = 0) uniform sampler2D brdfLut;
layout(set = 8, binding = 0) uniform sampler2D glossyNormal;

#include "shadow.glsl"

#define CNST_MAX_SPECULAR_EXP 64
float roughnessToSpecularPower(float r)
{
  return 2 / (pow(r,4)) - 2;
}

float specularPowerToConeAngle(float specularPower)
{
    // based on phong distribution model
    if(specularPower >= exp2(CNST_MAX_SPECULAR_EXP))
    {
        return 0.0f;
    }
    const float xi = 0.244f;
    float exponent = 1.0f / (specularPower + 1.0f);
    return acos(pow(xi, exponent));
}

vec4 bilinear(sampler2D tex, vec2 uv, float mip) {
    vec2 tSize = textureSize(glossyReflections, int(mip));
    vec2 texelSize = vec2(1.0) / tSize;

    vec2 f = fract( uv * tSize );
    uv += ( .5 - f ) * texelSize;    // move uv to texel centre

    vec4 tl = textureLod(tex, uv, mip);
    vec4 tr = textureLod(tex, uv + vec2(texelSize.x, 0), mip);
    vec4 bl = textureLod(tex, uv + vec2(0, texelSize.y), mip);
    vec4 br = textureLod(tex, uv + vec2(texelSize.x, texelSize.y), mip);

    vec4 tA = mix( tl, tr, f.x );
    vec4 tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}

const float EPSILON = 0.001;

float bilateralWeight(vec3 n1, vec3 n2, float d1, float d2, float hd1, float hd2 ) {
	if(d1 < 0 && d2 < 0) return 0;
	if(d1 < 0) return 1;

	float normalWeight = max(0, pow(dot(n1,n2), 32));
	float depthWeight = exp(-abs(d1 - d2) / 1);
    
    //float hitDistanceWeight = 1 / (EPSILON + abs(hd1-hd2) / 100.0);

	return normalWeight * depthWeight;
}

mat3 calculateTBN(vec3 dir) {
	const vec3 z = dir;
	const float sign_ = (z.z >= 0.0) ? 1.0 : -1.0;
	const float a = -1.0 / (sign_ + z.z);
	const float b = z.x * z.y * a;

	const vec3 x = vec3(1.0 + sign_ * a * pow(z.x, 2.0), sign_ * b, -sign_ * z.x);
	const vec3 y = vec3(b, sign_ + a * pow(z.y, 2.0), -z.y);

	return mat3(x, y, z);
}

float gaussian_weight(float offset, float deviation)
{
    float weight = 1.0 / sqrt(2.0 * 3.14159265359 * deviation * deviation);
    weight *= exp(-(offset * offset) / (2.0 * deviation * deviation));
    return weight;
}

vec4 bilateral4x4(vec2 uv, int mip, bool bilateral, vec4 indirectColor, float roughnessSquared) {
	vec4 col = vec4(0.0);
    float accum = 0.0;
    float weight;

	vec2 sourceSize = textureSize(glossyReflections, mip);
    vec2 texelSize = vec2(1.0) / sourceSize;

	vec4 lowerNormalDepth = textureLod(glossyNormal, uv, 0);

	const float c_halfSamplesX = 2.;
	const float c_halfSamplesY = 2.;
    float deviation = 1;

    for (float iy = -c_halfSamplesY; iy <= c_halfSamplesY; iy++)
    {
        for (float ix = -c_halfSamplesX; ix <= c_halfSamplesX; ix++)
        {          

            vec2 offset = vec2(ix, iy);
			vec2 newuv = uv + offset * texelSize;
            vec2 f = fract( newuv * sourceSize );
            newuv += ( .5 - f ) * texelSize;
            vec2 dist = newuv - uv;

            float fx = gaussian_weight(dist.x / texelSize.x, deviation);
            float fy = gaussian_weight(dist.y / texelSize.y, deviation);
           
			vec4 normalDepth = textureLod(glossyNormal, newuv, mip);
            vec4 newcol = textureLod(glossyReflections, newuv, mip);
            weight = fx *fy; 
            if(bilateral) {
                weight *= computeWeight(
                    lowerNormalDepth.w, normalDepth.w, 1,
					lowerNormalDepth.xyz, normalDepth.xyz, 32,
                    1, 1, 1);
            }
            //weight += 0.0000002;

            col += newcol * weight;
            accum += weight;
        }
    }

    float w = 0.01 * mip;
    col += indirectColor * w;
    accum += w;

    if(accum > 0.0000001) {
		vec4 result = col / accum;
		return result;
	}
	else {
		return vec4(0); //todo figure this out
	}
}

vec4 bilateral4x4hit(vec2 uv, int mip, bool bilateral) {
	vec4 col = vec4(0.0);
    float accum = 0.0;
    float weight;

	vec2 sourceSize = textureSize(glossyReflections, mip);
    vec2 texelSize = vec2(1.0) / sourceSize;

	vec4 lowerNormalDepth = textureLod(glossyNormal, uv, 0);
    vec4 colorCenter = textureLod(glossyReflections, uv, mip);

	const float c_halfSamplesX = 2.;
	const float c_halfSamplesY = 2.;
    float deviation = 1;

    for (float iy = -c_halfSamplesY; iy <= c_halfSamplesY; iy++)
    {
        for (float ix = -c_halfSamplesX; ix <= c_halfSamplesX; ix++)
        {          

            vec2 offset = vec2(ix, iy);
			vec2 newuv = uv + offset * texelSize;
            vec2 f = fract( newuv * sourceSize );
            newuv += ( .5 - f ) * texelSize;
            vec2 dist = newuv - uv;

            float fx = gaussian_weight(dist.x / texelSize.x, deviation);
            float fy = gaussian_weight(dist.y / texelSize.y, deviation);
           
			vec4 normalDepth = textureLod(glossyNormal, newuv, mip);
            vec4 newcol = textureLod(glossyReflections, newuv, mip);
            weight = fx *fy; 
            if(bilateral) {
                weight *= computeWeight(
                    lowerNormalDepth.w, normalDepth.w, 1,
					lowerNormalDepth.xyz, normalDepth.xyz, 32,
                    1, 1, 1)
                    ;
            }
            weight += 0.0000002;

            col += newcol * weight;
            accum += weight;
        }
    }

    if(accum > 0.0000001) {
		vec4 result = col / accum;
		return result;
	}
	else {
		return vec4(0); //todo figure this out
	}
}

vec3 rotateAxis(vec3 p, vec3 axis, float angle) {
    return mix(dot(axis, p)*axis, p, cos(angle)) + cross(axis,p)*sin(angle);
}

void main()
{

    vec4 gb1 = texture(gbufferAlbedoMetallic, InUv);
    vec4 gb2 = texture(gbufferNormalMotion, InUv);
    vec4 gb3 = texture(gbufferRoughnessDepthCurvatureMaterial, InUv);
    vec4 gb4 = texture(gbufferUV, InUv);

    vec3 inWorldPosition = world_position_from_depth(InUv, texture(gbufferDepth, InUv).r, cameraData.viewprojInverse);
    vec3 inNormal = octohedral_to_direction(gb2.rg);
    int inMaterialId = int(gb3.a);
    vec3 albedo = gb1.rgb;
    float roughness = gb3.r;
    float roughnessSquared = roughness * roughness;
    float metallic = gb1.a;
    
    vec2 inTexCoord = gb4.xy;
    vec2 inLightmapCoord = gb4.zw;

    if(inMaterialId < 0) {
        outFragColor = vec4(cameraData.clearColor);
        return;
    }

    if(materials[inMaterialId].texture > -1) {
        albedo = materials[inMaterialId].base_color.rgb * pow(albedo, vec3(2.2));
    }

    vec3 emissive_color = materials[inMaterialId].emissive_color;

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * vec4(inWorldPosition, 1.0);
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);

    vec3 view = normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz);

    //reflections
    vec4 reflectionColor = vec4(0);
    float tHit = 0;
    float mipChannel = 0;
    float cameraDist = distance(cameraData.cameraPos.xyz, inWorldPosition.xyz);
    float screenSpaceDistance = 0;

    //mat3 TBN = calculateTBN(inNormal);
	//vec3 microfacet1 =  sampleGGXVNDF(transpose(TBN) * view , roughnessSquared, roughnessSquared, 0, 0 ) ;
	//vec3 direction1 = TBN * reflect(-(transpose(TBN) * view), microfacet1);
    //vec3 microfacet2 =  sampleGGXVNDF(transpose(TBN) * view , roughnessSquared, roughnessSquared, 0.9, 0 ) ;
	//vec3 direction2 = TBN * reflect(-(transpose(TBN) * view), microfacet2);


    if(cameraData.useStochasticSpecular == 0)
    {
        if(roughness < 0.04) {
            reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).rgb, 1.0);
        }
        else if(roughness < 11) {
            ivec2 screenSize = textureSize(glossyReflections, 0);

            float coneAngle = acos(sqrt(0.11111f / (roughnessSquared * roughnessSquared + 0.11111f))) ;
            coneAngle = specularPowerToConeAngle(roughnessToSpecularPower(roughness));
            //coneAngle = acos(dot(direction1, direction2)) ;
            vec3 up = abs(inNormal.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
            vec3 tangent = normalize(cross(up, inNormal));
            vec3 bitangent = cross(inNormal, tangent);


            for(int i = 0; i < 2; i++) {
                if(i == 0) {
                    tHit = textureLod(glossyReflections, InUv, 0).w;
                }
                else
                {
                    float channel = mipChannel;
                    vec4 lowerMip = bilateral4x4hit(InUv, int(channel), true);
                    vec4 higherMip = bilateral4x4hit(InUv, int(channel) + 1, true);

                    tHit = mix(lowerMip, higherMip, channel - int(channel)).a;
                }
                
                
                if(i == 1) {
                    tHit += roughnessSquared;
                    tHit = clamp(tHit, 0, 10);
                    tHit /= 10;
                    tHit = pow(1 - tHit,  max(1,  2));
                    tHit = 1 - tHit;
                    tHit *= 10;
                    tHit -=  roughnessSquared;
                    clamp(tHit, 0, 10);
                }
                

                //if(gb3.b > 0.001) {
                //    tHit = 0.25;
                //}
                
                
                //if(i == 0) {
                //}

                float adjacentLength = tHit ;
                float op_len = tan(coneAngle) * adjacentLength;
                float incircleSize = op_len;                
                
                vec3 virtual_pos_1 = cameraData.cameraPos.xyz - (tHit + cameraDist) * view; // 
                vec4 projected_virtual_pos_1 = cameraData.viewproj * vec4(virtual_pos_1, 1.0f);
                projected_virtual_pos_1.xy /= projected_virtual_pos_1.w;
                vec2 hitSS1 = (projected_virtual_pos_1.xy * 0.5f + 0.5f) * screenSize;
                
                vec3 virtual_pos_2 = cameraData.cameraPos.xyz - (cameraDist) * view + tHit  * rotateAxis(-view, cross(tangent, view), coneAngle);
                vec4 projected_virtual_pos_2 = cameraData.viewproj * vec4(virtual_pos_2, 1.0f);
                projected_virtual_pos_2.xy /= projected_virtual_pos_2.w;
                vec2 hitSS2 = (projected_virtual_pos_2.xy * 0.5f + 0.5f) * screenSize;

                vec3 virtual_pos_3 = cameraData.cameraPos.xyz - (cameraDist) * view + tHit  * rotateAxis(-view, cross(bitangent, view), coneAngle);
                vec4 projected_virtual_pos_3 = cameraData.viewproj * vec4(virtual_pos_3, 1.0f);
                projected_virtual_pos_3.xy /= projected_virtual_pos_3.w;
                vec2 hitSS3 = (projected_virtual_pos_3.xy * 0.5f + 0.5f) * screenSize;

                screenSpaceDistance = sqrt(pow(distance(hitSS1, hitSS3),2) + pow(distance(hitSS1, hitSS2),2)) * 2;
                
                /*
                vec3 virtual_pos_2 = virtual_pos_1 + incircleSize * bitangent;
                vec4 projected_virtual_pos_2 = cameraData.viewproj * vec4(virtual_pos_2, 1.0f);
                projected_virtual_pos_2.xy /= projected_virtual_pos_2.w;
                vec2 hitSS2 = (projected_virtual_pos_2.xy * 0.5f + 0.5f) * screenSize;

                vec3 virtual_pos_3 = virtual_pos_1 + incircleSize * tangent ;
                vec4 projected_virtual_pos_3 = cameraData.viewproj * vec4(virtual_pos_3, 1.0f);
                projected_virtual_pos_3.xy /= projected_virtual_pos_3.w;
                vec2 hitSS3 = (projected_virtual_pos_3.xy * 0.5f + 0.5f) * screenSize;
                
                screenSpaceDistance = sqrt(pow(distance(hitSS1, hitSS3),2) + pow(distance(hitSS1, hitSS2),2)) * 2;
                */

                mipChannel = log2(screenSpaceDistance /4);
            }
            float sampleMipChannel = clamp(mipChannel, 0.0f, 7);
            vec4 lowerMip = vec4(0);
            vec4 higherMip = vec4(0);
            lowerMip = bilateral4x4(InUv, int(sampleMipChannel), true, texture(indirectLightMap, inLightmapCoord).rgba, roughnessSquared);
            higherMip = bilateral4x4(InUv, int(sampleMipChannel) + 1, true, texture(indirectLightMap, inLightmapCoord).rgba, roughnessSquared);

            reflectionColor = mix(lowerMip, higherMip, sampleMipChannel - int(sampleMipChannel));

            //reflectionColor = mix(reflectionColor, vec4(texture(indirectLightMap, inLightmapCoord).rgb, 1.0) , clamp((sampleMipChannel - 2) / 8, 0, 1));
            //reflectionColor = mix(reflectionColor, vec4(texture(indirectLightMap, inLightmapCoord).rgb, 1.0) , min(1.0, screenSpaceDistance / textureSize(glossyReflections, 0).x));
            reflectionColor =  /*vec4(mix(vec3(1), albedo, roughness), 1) * */ mix(reflectionColor /* / vec4(mix(vec3(1), albedo, roughness), 1) */, (reflectionColor * 0 + vec4(texture(indirectLightMap, inLightmapCoord).rgb, 1.0) * 0.5) , (pow(16, roughness) - 1) / 15);
            //reflectionColor = vec4(tHit/16);
        }
        else {
            reflectionColor = vec4(texture(indirectLightMap, inLightmapCoord).rgb , 1.0);
        }
    }
    else {
        reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).xyz, 1.0);
        tHit = textureLod(glossyReflections, InUv, 0).r;
    }
    
    vec3 directLight = calculate_direct_lighting(albedo, metallic, roughness, normalize(inNormal), view, normalize(cameraData.lightPos).xyz, cameraData.lightColor.xyz) * shadow;
    
    vec3 diffuseLight = cameraData.indirectDiffuse == 1 ? texture(indirectLightMap, inLightmapCoord).xyz : vec3(0);
    reflectionColor = cameraData.indirectSpecular == 1 ? reflectionColor : vec4(0);
    
    vec3 indirectLight = calculate_indirect_lighting(albedo, metallic, roughness, normalize(inNormal), view, diffuseLight, reflectionColor.rgb, brdfLut, vec3(0));
    
    vec3 outColor = emissive_color + directLight + indirectLight  ;

    //outFragColor = vec4(textureLod(glossyNormal, InUv, 2).xyz, 1.0);
    outFragColor = reflectionColor;
    //outFragColor = vec4(diffuseLight, 1);
    ////outFragColor = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(0)), int(5), inMaterialId);
    tHit /= 10;
    outFragColor = vec4(vec3(outColor), 1.0);
}