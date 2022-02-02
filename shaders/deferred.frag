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

layout(set = 1, binding = 0) uniform sampler2D gbufferPositionMaterial;
layout(set = 1, binding = 1) uniform sampler2D gbufferNormal;
layout(set = 1, binding = 2) uniform sampler2D gbufferUV;

layout(set = 2, binding = 0) uniform sampler2D[] textures;
layout(std140, set = 3, binding = 0) readonly buffer MaterialBuffer{ GPUBasicMaterialData materials[]; };
layout(set = 4, binding = 0) uniform sampler2D shadowMap;
layout(set = 5, binding = 0) uniform sampler2D indirectLightMap;
layout(set = 6, binding = 0) uniform sampler2D glossyReflections;
layout(set = 7, binding = 0) uniform sampler2D brdfLut;

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

vec4 bilinear(vec2 uv, float mip, int material) {
    vec2 tSize = textureSize(glossyReflections, int(mip));
    vec2 texelSize = vec2(1.0) / tSize;

    vec2 f = fract( uv * tSize );
    uv += ( .5 - f ) * texelSize;    // move uv to texel centre

    vec4 c = textureLod(glossyReflections, uv, mip);
    vec4 s1 = textureLod(glossyReflections, uv + vec2(-texelSize.x, -texelSize.y), mip);
    vec4 s2 = textureLod(glossyReflections, uv + vec2(0, -texelSize.y), mip);
    vec4 s3 = textureLod(glossyReflections, uv + vec2(texelSize.x, -texelSize.y), mip);
    vec4 s4 = textureLod(glossyReflections, uv + vec2(-texelSize.x, 0), mip);
    vec4 s5 = textureLod(glossyReflections, uv + vec2(texelSize.x, 0), mip);
    vec4 s6 = textureLod(glossyReflections, uv + vec2(-texelSize.x, texelSize.y), mip);
    vec4 s7 = textureLod(glossyReflections, uv + vec2(0, texelSize.y), mip);
    vec4 s8 = textureLod(glossyReflections, uv + vec2(texelSize.x, texelSize.y), mip);

    vec4 tl = vec4(0);
    vec4 tr = vec4(0);
    vec4 bl = vec4(0);
    vec4 br = vec4(0);

    if(int(s5.w) == material && int(s7.w) == material && int(s8.w) == material) {
            tl = c;
            tr = s5;
            bl = s7;
            br = s8;
        }
        else if(int(s5.w) == material) {
            tl = c;
            tr = s5;
            bl = int(s7.w) == material ? s7 : c;
            br = int(s8.w) == material ? s8 : s5;
        }
        else if(int(s7.w) == material) {
            tl = c;
            tr = int(s5.w) == material ? s5 : c;
            bl = s7;
            br = int(s8.w) == material ? s8 : s7;
        }
        else if(int(s1.w) == material) {
            tl = c;
            tr = int(s3.w) == material ? s3 : c;
            bl = c;
            br = c;
        }
        else {
            return c;
        }

    vec4 tA = mix( tl, tr, f.x );
    vec4 tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
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

    vec4 tl = vec4(0, 0, 0, -1);
    vec4 tr = vec4(0, 0, 0, -1);
    vec4 bl = vec4(0, 0, 0, -1);
    vec4 br = vec4(0, 0, 0, -1);

    
    if(int(c.w) != material) {
        if(int(s5.w) == material && int(s4.w) != material && int(s2.w) != material && int(s7.w) != material) {
            return bilinear(vec2((uv + vec2(texelSize.x + 0.001, 0) / 2).x, ogUv.y), mip, material);
        }
        else if(int(s4.w) == material && int(s5.w) != material && int(s2.w) != material && int(s7.w) != material) {
            return bilinear(vec2((uv - vec2(texelSize.x + 0.001, 0) / 2).x, ogUv.y), mip, material);
        }
        else if(int(s2.w) == material && int(s4.w) != material && int(s5.w) != material && int(s7.w) != material) {
            return bilinear(vec2(ogUv.x, (uv - vec2(0, texelSize.y + 0.001) / 2).y), mip, material);
        }
        else if(int(s7.w) == material && int(s4.w) != material && int(s2.w) != material && int(s5.w) != material) {
            return bilinear(vec2(ogUv.x, (uv + vec2(0, texelSize.y + 0.001) / 2).y), mip, material);
        }
        else if(int(s2.w) == material && int(s4.w) == material) {
            tl = bilinear(uv - vec2(texelSize.x + 0.001, texelSize.y - 0.001) / 2, mip, material);
            tr = bilinear(uv + vec2(texelSize.x - 0.001, -texelSize.y - 0.001) / 2, mip, material);
            bl = bilinear(uv + vec2(-texelSize.x - 0.001, texelSize.y - 0.001) / 2, mip, material);
            br = bilinear(uv - vec2(texelSize.x + 0.001, texelSize.y - 0.001) / 2, mip, material);
        }
        else if(int(s2.w) == material && int(s5.w) == material) {
            tl = bilinear(uv + vec2(-texelSize.x + 0.001, -texelSize.y - 0.001) / 2, mip, material);
            tr = bilinear(uv + vec2(texelSize.x - 0.001, -texelSize.y - 0.001) / 2, mip, material);
            bl = bilinear(uv + vec2(texelSize.x - 0.001, -texelSize.y - 0.001) / 2, mip, material);
            br = bilinear(uv + vec2(texelSize.x + 0.001, +texelSize.y - 0.001) / 2, mip, material);
        }
        //these two are not working correctly!
        else if(int(s7.w) == material && int(s4.w) == material) {
            tl = bilinear(uv + vec2(-texelSize.x - 0.001, +texelSize.y - 0.001) / 2, mip, material);
            tr = bilinear(uv + vec2(+texelSize.x - 0.001, +texelSize.y + 0.001) / 2, mip, material);
            bl = bilinear(uv + vec2(-texelSize.x - 0.001, +texelSize.y - 0.001) / 2, mip, material);
            br = bilinear(uv + vec2(+texelSize.x - 0.001, +texelSize.y + 0.001) / 2, mip, material);
        }
        else if(int(s7.w) == material && int(s5.w) == material) {
            tl = bilinear(uv + vec2(texelSize.x + 0.001, +texelSize.y - 0.001) / 2, mip, material);
            tr = bilinear(uv + vec2(texelSize.x + 0.001, -texelSize.y + 0.001) / 2, mip, material);
            bl = bilinear(uv + vec2(-texelSize.x + 0.001, +texelSize.y - 0.001) / 2, mip, material);
            br = bilinear(uv + vec2(texelSize.x + 0.001, texelSize.y - 0.001) / 2, mip, material);
        }
        else {
            return vec4(0, 22, 0, 1);
        }
    }
    else {

        //usual bilinear (c, s5, s7, s8)
        if(int(s5.w) == material && int(s7.w) == material && int(s8.w) == material) {
            tl = c;
            tr = s5;
            bl = s7;
            br = s8;
        }
        else if(int(s5.w) == material) {
        //might have some issues
            tl = c;
            tr = s5;
            bl = int(s7.w) == material ? s7 : c;
            br = int(s8.w) == material ? s8 : s5;
        }
        else if(int(s7.w) == material) {
            tl = c;
            tr = int(s5.w) == material ? s5 : c;
            bl = s7;
            br = int(s8.w) == material ? s8 : s7;
        }
        else if(int(s1.w) == material) {
            tl = c;
            tr = int(s3.w) == material ? s3 : c;
            bl = c;
            br = c;
        }
        else {
            return c;
        }
    }

    vec4 tA = mix( tl, tr, f.x );
    vec4 tB = mix( bl, br, f.x );
    vec4 mixResult = mix( tA, tB, f.y );

    return mixResult;
}

void main()
{
    vec3 inWorldPosition = texture(gbufferPositionMaterial, InUv).xyz;
    int inMaterialId = int(texture(gbufferPositionMaterial, InUv).a);
    vec3 inNormal = texture(gbufferNormal, InUv).xyz;
    vec3 inWorldPosition2 = texture(gbufferPositionMaterial, InUv + vec2(1.0) / textureSize(gbufferPositionMaterial,0)).xyz;
    vec2 inTexCoord = texture(gbufferUV, InUv).xy;
    vec2 inLightmapCoord = texture(gbufferUV, InUv).zw;

    if(inMaterialId < 0) {
        discard;
    }

    vec3 albedo = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materials[inMaterialId].emissive_color * 0;

    float roughness = materials[inMaterialId].roughness_factor;
    float metallic = materials[inMaterialId].metallic_factor;

	if(materials[inMaterialId].texture > -1) {
        albedo = pow(texture(textures[materials[inMaterialId].texture], inTexCoord).xyz, vec3(2.2));
    }
    else {
        albedo = materials[inMaterialId].base_color.xyz;
    }

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * vec4(inWorldPosition, 1.0);
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);

    //reflections
    vec4 reflectionColor = vec4(0);

    if(cameraData.useStochasticSpecular == 0)
    {
        float mipChannel = 0;
        vec3 view = normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz);
        
        float tHit = fract(bilinear(InUv, 0, inMaterialId).a) * 10000.0;
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

        mipChannel = 4;//clamp(log2(distance(hitSS1.xy, hitSS2.xy)), 0.0f, 7);

        vec4 lowerMip = customBilinear(InUv - vec2(0.5) / textureSize(glossyReflections, int(mipChannel)), int(mipChannel), inMaterialId);
        vec4 higherMip = customBilinear(InUv - vec2(0.5) / textureSize(glossyReflections, int(mipChannel) + 1), int(mipChannel) + 1, inMaterialId);
        if(lowerMip.w == -1) lowerMip = texture(indirectLightMap, inLightmapCoord);
        if(higherMip.w == -1) higherMip = texture(indirectLightMap, inLightmapCoord);
        reflectionColor = mix(lowerMip, higherMip, mipChannel - int(mipChannel));
        reflectionColor =  mix(reflectionColor, texture(indirectLightMap, inLightmapCoord), (pow(16, roughness) - 1) / 15);
        //reflectionColor = vec4(textureLod(glossyReflections, InUv, mipChannel).xyz, 1.0);
    }
    else {
        reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).xyz, 1.0);
    }

    if(inMaterialId == 5) {
        //debugPrintfEXT("%f - %f %f %f --- %f %f %f --- %f %f %f\n", tHit, one.x, one.y, one.z, two.x, two.y, two.z, inNormal.x, inNormal.y, inNormal.z);
        //debugPrintfEXT("%f %f vs %f %f\n", screenWorld.x, screenWorld.y, hitWorld.x, hitWorld.y);
    }

    
    vec3 directLight = calculate_direct_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), normalize(cameraData.lightPos).xyz, cameraData.lightColor.xyz) * shadow;
    
    vec3 diffuseLight = cameraData.indirectDiffuse == 1 ? texture(indirectLightMap, inLightmapCoord).xyz : vec3(0);
    reflectionColor = cameraData.indirectSpecular == 1 ? reflectionColor : vec4(0);
    vec3 indirectLight = calculate_indirect_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), diffuseLight, reflectionColor.rgb, brdfLut);
    //vec3 indirectLight = calculate_indirect_lighting_nospecular(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), diffuseLight, textureLod(glossyReflections, InUv, 0).xyz);
    
    vec3 outColor = emissive_color + directLight + indirectLight / PI ;

    outFragColor = vec4(outColor, 1.0f);
}