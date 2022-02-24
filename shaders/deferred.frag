#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : enable

#include "common.glsl"
#include "brdf.glsl"

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
	float depthWeight = abs(d1-d2) > 0.5 ? 0 : 1;
    
    //float hitDistanceWeight = 1 / (EPSILON + abs(hd1-hd2) / 100.0);

	return normalWeight * depthWeight ;
}


// I probably have all the data needed, just need a better sampling method.
//this first if statements works well, rest needs work.
//the other side of the pixel should be color center, different than usual bilinear
vec4 customBilinear(vec2 uv, int mip, int material) {
    vec2 ogUv = uv;
    vec2 tSize = textureSize(glossyReflections, int(mip));
    vec2 texelSize = vec2(1.0) / tSize;

    vec2 f = fract( uv * tSize );
    uv += ( 0.5 -f ) * texelSize;
    /*
    s1 | s2 | s3
    s4 |  c | s5
    s6 | s7 | s8
    */
    vec4 c = textureLod(glossyReflections, uv, mip);
    vec4 s1 = textureLod(glossyReflections, uv + vec2(-texelSize.x, -texelSize.y), mip);
    vec4 s2 = textureLod(glossyReflections, uv + vec2(0, -texelSize.y), mip);
    vec4 s3 = textureLod(glossyReflections, uv + vec2(texelSize.x, -texelSize.y), mip);
    vec4 s4 = textureLod(glossyReflections, uv + vec2(-texelSize.x, 0), mip);
    vec4 s5 = textureLod(glossyReflections, uv + vec2(texelSize.x, 0), mip);
    vec4 s6 = textureLod(glossyReflections, uv + vec2(-texelSize.x, texelSize.y), mip);
    vec4 s7 = textureLod(glossyReflections, uv + vec2(0, texelSize.y), mip);
    vec4 s8 = textureLod(glossyReflections, uv + vec2(texelSize.x, texelSize.y), mip);

    float defaultHitDistance = textureLod(glossyReflections, ogUv, 0).w;
    vec4 defaultNormal = textureLod(glossyNormal, ogUv, 0);

    vec4 nc = textureLod(glossyNormal, uv, mip);
    vec4 ns1 = textureLod(glossyNormal, uv + vec2(-texelSize.x, -texelSize.y), mip);
    vec4 ns2 = textureLod(glossyNormal, uv + vec2(0, -texelSize.y), mip);
    vec4 ns3 = textureLod(glossyNormal, uv + vec2(texelSize.x, -texelSize.y), mip);
    vec4 ns4 = textureLod(glossyNormal, uv + vec2(-texelSize.x, 0), mip);
    vec4 ns5 = textureLod(glossyNormal, uv + vec2(texelSize.x, 0), mip);
    vec4 ns6 = textureLod(glossyNormal, uv + vec2(-texelSize.x, texelSize.y), mip);
    vec4 ns7 = textureLod(glossyNormal, uv + vec2(0, texelSize.y), mip);
    vec4 ns8 = textureLod(glossyNormal, uv + vec2(texelSize.x, texelSize.y), mip);

    vec4 tl = vec4(0, 0, 0, -1);
    vec4 tr = vec4(0, 0, 0, -1);
    vec4 bl = vec4(0, 0, 0, -1);
    vec4 br = vec4(0, 0, 0, -1);

    vec4 ntl = vec4(0, 0, 0, -1);
    vec4 ntr = vec4(0, 0, 0, -1);
    vec4 nbl = vec4(0, 0, 0, -1);
    vec4 nbr = vec4(0, 0, 0, -1);

    tl = c;
    tr = s5;
    bl = s7;
    br = s8;

    ntl = nc;
    ntr = ns5;
    nbl = ns7;
    nbr = ns8;

    float w1 = (1 - f.x) * (1 - f.y) * bilateralWeight(defaultNormal.xyz, ntl.xyz, defaultNormal.w, ntl.w, defaultHitDistance, tl.w);
	float w2 = (f.x) * (1 - f.y) * bilateralWeight(defaultNormal.xyz, ntr.xyz, defaultNormal.w, ntr.w, defaultHitDistance, tr.w);
	float w3 = (1 - f.x) * (f.y) * bilateralWeight(defaultNormal.xyz, nbl.xyz, defaultNormal.w, nbl.w, defaultHitDistance, bl.w);
	float w4 = (f.x) * (f.y) * bilateralWeight(defaultNormal.xyz, nbr.xyz, defaultNormal.w, nbr.w, defaultHitDistance, br.w);
    

    if(w1+w2+w3+w4 > 0.00001) {
		return (w1 * tl + w2 * tr + w3 * bl + w4 * br) / (w1+w2+w3+w4);
	}
	else {
		return vec4(0,0,0,1);
	}
}

float gaussian_weight(float offset, float deviation)
{
    float weight = 1.0 / sqrt(2.0 * 3.14159265359 * deviation * deviation);
    weight *= exp(-(offset * offset) / (2.0 * deviation * deviation));
    return weight;
}

vec4 bilateral4x4(vec2 uv, int mip) {
	vec4 col = vec4(0.0);
    float accum = 0.0;
    float weight;

	vec2 sourceSize = textureSize(glossyReflections, mip);
    vec2 texelSize = vec2(1.0) / sourceSize;

	vec4 lowerNormalDepth = textureLod(glossyNormal, uv, 0);

    float deviation = 1;
	const float c_halfSamplesX = 2.;
	const float c_halfSamplesY = 2.;

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
            weight = fx *fy * bilateralWeight(lowerNormalDepth.xyz, normalDepth.xyz, lowerNormalDepth.w, normalDepth.w, 0, 0);
            col += textureLod(glossyReflections, newuv, mip) * weight;
            accum += weight;
        }
    }

    //return textureLod(glossyNormal, uv, 6) * 0.5 + vec4(0.5);
    //return textureLod(glossyReflections, uv, 6);

    //return textureLod(glossyReflections, uv, mip);
    if(accum > 0.0000001) {
		return col / accum;
	}
	else {
		return vec4(0); //todo figure this out
	}
}

void main()
{
    vec4 gb1 = texture(gbufferAlbedoMetallic, InUv);
    vec4 gb2 = texture(gbufferNormalMotion, InUv);
    vec4 gb3 = texture(gbufferRoughnessDepthCurvatureMaterial, InUv);
    vec4 gb4 = texture(gbufferUV, InUv);

    vec3 inWorldPosition = world_position_from_depth(InUv, texture(gbufferDepth, InUv).r, cameraData.viewprojInverse);
    vec3 inWorldPosition2 = world_position_from_depth(InUv + vec2(1.0) / textureSize(gbufferDepth, 0), texture(gbufferDepth, InUv).r, cameraData.viewprojInverse);
    vec3 inNormal = octohedral_to_direction(gb2.rg);
    int inMaterialId = int(gb3.a);
    vec3 albedo = gb1.rgb;
    float roughness = gb3.r;
    float metallic = gb1.a;
    
    vec2 inTexCoord = gb4.xy;
    vec2 inLightmapCoord = gb4.zw;

    if(inMaterialId < 0) {
        discard;
    }

    if(materials[inMaterialId].texture > -1) {
        albedo = pow(albedo, vec3(2.2));
    }

    vec3 emissive_color = materials[inMaterialId].emissive_color * 0;

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * vec4(inWorldPosition, 1.0);
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);

    //reflections
    vec4 reflectionColor = vec4(0);
    float tHit = 0;
    if(cameraData.useStochasticSpecular == 0)
    {
        if(roughness < 0.05) {
            reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).rgb, 1.0);
        }
        else if(roughness < 11) {
            float mipChannel = 0;
            vec3 view = normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz);
        
            for(int i = 0; i < 1; i++) {
                tHit = textureLod(glossyReflections, InUv, mipChannel).w;
    	        float coneAngle = specularPowerToConeAngle(roughnessToSpecularPower(roughness)) * 0.5f;
	            float adjacentLength = tHit;
                float op_len = 2.0 * tan(coneAngle) * adjacentLength; // opposite side of iso triangle
                float a = op_len;
                float h = coneAngle;
                float a2 = a * a;
                float fh2 = 4.0f * h * h;
                float incircleSize = (a * (sqrt(a2 + fh2) - a)) / (4.0f * h);

                vec4 hitSS1 = cameraData.viewproj * vec4(inWorldPosition.xyz, 1);
                vec4 hitSS2 = cameraData.viewproj * vec4(inWorldPosition.xyz + normalize(inWorldPosition2 - inWorldPosition) * incircleSize * 7, 1); // i need bitangent or something else
                hitSS1 /= hitSS1.w;
                hitSS2 /= hitSS2.w;

                hitSS1.xy = (hitSS1.xy + vec2(1)) / 2;
                hitSS2.xy = (hitSS2.xy + vec2(1)) / 2;
                hitSS1.xy *= vec2(1600, 900);
                hitSS2.xy *= vec2(1600, 900);

                mipChannel = clamp(log2(distance(hitSS1.xy, hitSS2.xy) / 4), 0.0f, 7);
            }
            bool bilinear = false;
            vec4 lowerMip = vec4(0);
            vec4 higherMip = vec4(0);
            if(bilinear) {
                lowerMip = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(mipChannel)), int(mipChannel), inMaterialId);
                higherMip = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(mipChannel) + 1), int(mipChannel) + 1, inMaterialId);
            }
            else {
                lowerMip = bilateral4x4(InUv, int(mipChannel));
                higherMip = bilateral4x4(InUv, int(mipChannel) + 1);
            }
            reflectionColor = mix(lowerMip, higherMip, mipChannel - int(mipChannel));
            reflectionColor =  mix(reflectionColor, vec4(texture(indirectLightMap, inLightmapCoord).rgb / (mix(albedo, vec3(1), roughness)), 1) , (pow(64, mipChannel / 7) - 1) / 63);
            reflectionColor =  mix(reflectionColor, vec4(texture(indirectLightMap, inLightmapCoord).rgb / (mix(albedo, vec3(1), roughness)), 1) , (pow(16, roughness) - 1) / 15);
        }
        else {
            reflectionColor = vec4(texture(indirectLightMap, inLightmapCoord).rgb , 1.0);
        }
    }
    else {
            reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).xyz, 1.0);
    }
    
    vec3 directLight = calculate_direct_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), normalize(cameraData.lightPos).xyz, cameraData.lightColor.xyz) * shadow;
    
    vec3 diffuseLight = cameraData.indirectDiffuse == 1 ? texture(indirectLightMap, inLightmapCoord).xyz : vec3(0);
    reflectionColor = cameraData.indirectSpecular == 1 ? reflectionColor : vec4(0);
    vec3 indirectLight = calculate_indirect_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), diffuseLight, reflectionColor.rgb, brdfLut);
    //vec3 indirectLight = calculate_indirect_lighting_nospecular(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), diffuseLight, textureLod(glossyReflections, InUv, 0).xyz);
    
    vec3 outColor = emissive_color + directLight + indirectLight  ;

    //outFragColor = vec4(textureLod(glossyNormal, InUv, 2).xyz, 1.0);
    outFragColor = reflectionColor;
    //outFragColor = vec4(diffuseLight, 1);
    //outFragColor = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(0)), int(5), inMaterialId);
    outFragColor = vec4(outColor, 1.0);
}

// can i make the entire thing work with bicubic filters instead?