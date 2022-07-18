#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../common.glsl"

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
layout(set = 5, binding = 0) uniform sampler2D indirectLightMap;

//all object matrices
layout(std140,set = 3, binding = 0) readonly buffer MaterialBuffer{

	GPUBasicMaterialData materials[];
};

#include "../shadow_mapping/shadow.glsl"

const float PI  = 3.14159265358979323846264;

void main()
{
    vec3 albedo = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materials[material_id].emissive_color ;

	if(materials[material_id].texture > -1) {
        albedo = materials[material_id].base_color.xyz * pow(texture(textures[materials[material_id].texture], texCoord).xyz, vec3(2.2));
    }
    else {
        albedo = materials[material_id].base_color.xyz;
    }

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * inFragPos;
    //float shadow = textureProj(shadowPos / shadowPos.w, vec2(0.0));
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);
    //float shadow = filterPCF(shadowPos / shadowPos.w);

    vec3 N = normalize(inNormal);
    vec3 L = normalize(inLightVec);
    
	vec3 diffuse = (emissive_color +  clamp(dot(N, L), 0.0, 1.0) * inLightColor * albedo / PI * shadow + texture(indirectLightMap, inLightmapCoord).xyz * albedo ) ;

    outFragColor = vec4(diffuse, 1.0f);  
   
}
