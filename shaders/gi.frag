#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

//shader input
layout (location = 0) in vec2 inTexCoord;
layout (location = 1) flat in int inMaterialId;
layout (location = 2) in vec2 inLightmapCoord;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };

layout(set = 2, binding = 0) uniform sampler2D[] textures;
layout(set = 4, binding = 0) uniform sampler2D lightMap;

//all object matrices
layout(std140,set = 3, binding = 0) readonly buffer MaterialBuffer{

	GPUBasicMaterialData materials[];
} materialBuffer;

void main()
{
    vec3 color = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materialBuffer.materials[inMaterialId].emissive_color;

    if(emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0) {
        color = vec3(1.0f, 1.0f, 1.0f);
    }
    else {
	    if(materialBuffer.materials[inMaterialId].texture > -1) {
            color = texture(textures[materialBuffer.materials[inMaterialId].texture], inTexCoord).xyz;
        }
        else {
            color = materialBuffer.materials[inMaterialId].base_color.xyz;
        }
    }

    vec3 lightmapResult = texture(lightMap, inLightmapCoord).xyz;

    outFragColor = vec4(color * lightmapResult, 1.0f);
}