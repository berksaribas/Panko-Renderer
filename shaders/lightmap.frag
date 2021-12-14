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

//all object matrices
layout(std140,set = 3, binding = 0) readonly buffer MaterialBuffer{

	GPUBasicMaterialData materials[];
} materialBuffer;

#include "shadow.glsl"

void main()
{
    vec3 color = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materialBuffer.materials[material_id].emissive_color;

    if(emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0) {
        outFragColor = vec4(0);
    }
    else {
        vec4 shadowPos = biasMat * shadowMapData.depthMVP * inFragPos;
        //float shadow = textureProj(shadowPos / shadowPos.w, vec2(0.0));
        float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);
        //float shadow = filterPCF(shadowPos / shadowPos.w);

        vec3 N = normalize(inNormal);
        vec3 L = normalize(inLightVec);
        
	    vec3 diffuse = clamp(dot(N, L), 0.0, 1.0) * inLightColor * color;

        outFragColor = vec4(diffuse * shadow, 1.0f);  
    }
}
