#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

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

#include "shadow.glsl"

void main()
{
    vec3 inWorldPosition = texture(gbufferPositionMaterial, InUv).xyz;
    int inMaterialId = int(texture(gbufferPositionMaterial, InUv).a);
    vec3 inNormal = texture(gbufferNormal, InUv).xyz;
    vec2 inTexCoord = texture(gbufferUV, InUv).xy;
    vec2 inLightmapCoord = texture(gbufferUV, InUv).zw;

    if(inMaterialId < 0) {
        discard;
    }

    vec3 albedo = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materials[inMaterialId].emissive_color;

    float roughness = materials[inMaterialId].roughness_factor;
    float metallic = materials[inMaterialId].metallic_factor;

    if(emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0) {
        albedo = vec3(1.0f, 1.0f, 1.0f);
    }
    else {
	    if(materials[inMaterialId].texture > -1) {
            albedo = pow(texture(textures[materials[inMaterialId].texture], inTexCoord).xyz, vec3(2.2));
        }
        else {
          albedo = materials[inMaterialId].base_color.xyz;
        }
    }

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * vec4(inWorldPosition, 1.0);
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);

    //if(inMaterialId == 5) {
    //    roughness = 0;
    //    metallic = 1;
    //}
    //else {
    //}
    //if(roughness > 0) {
        roughness = 1;
        metallic = 0;
    //}

    vec3 directLight = calculate_direct_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), normalize(cameraData.lightPos).xyz, cameraData.lightColor.xyz) * shadow;
    vec3 indirectLight = calculate_indirect_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), texture(indirectLightMap, inLightmapCoord).xyz, texture(glossyReflections, InUv).xyz);
    vec3 outColor = directLight + indirectLight;

    outFragColor = vec4(outColor, 1.0f);
}