#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"
#include "brdf.glsl"

//shader input
layout (location = 0) in vec2 inTexCoord;
layout (location = 1) flat in int inMaterialId;
layout (location = 2) in vec2 inLightmapCoord;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec4 inWorldPosition;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };
layout(set = 0, binding = 1) uniform _ShadowMapData { GPUShadowMapData shadowMapData; };

layout(set = 2, binding = 0) uniform sampler2D[] textures;
layout(set = 4, binding = 0) uniform sampler2D shadowMap;
layout(set = 5, binding = 0) uniform sampler2D indirectLightMap;

#include "shadow.glsl"

//all object matrices
layout(std140,set = 3, binding = 0) readonly buffer MaterialBuffer{

	GPUBasicMaterialData materials[];
} materialBuffer;

void main()
{
    vec3 albedo = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materialBuffer.materials[inMaterialId].emissive_color;

    float roughness = materialBuffer.materials[inMaterialId].roughness_factor;
    float metallic = materialBuffer.materials[inMaterialId].metallic_factor;

    if(emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0) {
        albedo = vec3(1.0f, 1.0f, 1.0f);
    }
    else {
	    if(materialBuffer.materials[inMaterialId].texture > -1) {
            albedo = pow(texture(textures[materialBuffer.materials[inMaterialId].texture], inTexCoord).xyz, vec3(2.2));
        }
        else {
          albedo = materialBuffer.materials[inMaterialId].base_color.xyz;
        }
    }

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * inWorldPosition;
    //float shadow = textureProj(shadowPos / shadowPos.w, vec2(0.0));
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);

    roughness = 1;
    metallic = 0;

    vec3 directLight = calculate_direct_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), normalize(cameraData.lightPos).xyz, normalize(cameraData.lightColor).xyz) * shadow;
    vec3 indirectLight = calculate_indirect_lighting(albedo, metallic, roughness, normalize(inNormal), normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz), texture(indirectLightMap, inLightmapCoord).xyz, vec3(0));

    outFragColor = vec4(directLight + indirectLight, 1.0f);
}