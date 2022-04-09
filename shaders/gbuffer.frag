#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec4 inOldPosition;
layout (location = 2) flat in int inMaterialId;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec2 inTexCoord;
layout (location = 5) in vec2 inLightmapCoord;
layout (location = 6) in vec3 inTangent;
layout (location = 7) in vec3 inBitangent;

layout(location = 0) out vec4 gbufferAlbedoMetallic;
layout(location = 1) out vec4 gbufferNormalMotion;
layout(location = 2) out vec4 gbufferRoughnessDepthCurvatureMaterial;
layout(location = 3) out vec4 gbufferUV;

layout(set = 2, binding = 0) uniform sampler2D[] textures;
layout(std140, set = 3, binding = 0) readonly buffer MaterialBuffer{ GPUBasicMaterialData materials[]; };

float linearize_depth(float d,float zNear,float zFar)
{
    float z_n = 2.0 * d - 1.0;
    return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
}


vec2 direction_to_octohedral(vec3 normal)
{
    vec2 p = normal.xy * (1.0f / dot(abs(normal), vec3(1.0f)));
    return normal.z > 0.0f ? p : (1.0f - abs(p.yx)) * (step(0.0f, p) * 2.0f - vec2(1.0f));
}

vec2 compute_motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Perspective division, covert clip space positions to NDC.
    vec2 current = (current_pos.xy / current_pos.w);
    vec2 prev    = (prev_pos.xy / prev_pos.w);

    // Remap to [0, 1] range
    current = current * 0.5 + 0.5;
    prev    = prev * 0.5 + 0.5;

    // Calculate velocity (current -> prev)
    return (prev - current);
}

float compute_curvature()
{
    vec3 dx = dFdx(inNormal);
    vec3 dy = dFdy(inNormal);

    float x = dot(dx, dx);
    float y = dot(dy, dy);

    return pow(max(x, y), 0.5f);
}

vec3 getNormal()
{
	// Perturb normal, see http://www.thetenthplanet.de/archives/1180
	vec3 tangentNormal = texture(textures[materials[inMaterialId].normal_texture], inTexCoord).xyz * 2.0 - 1.0;

	mat3 TBN = mat3(normalize(inTangent), normalize(inBitangent), normalize(inNormal));

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    return normalize(TBN * tangentNormal);
}

void main()
{
    vec3 albedo = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materials[inMaterialId].emissive_color * 0;

    float roughness = materials[inMaterialId].roughness_factor;
    float metallic = materials[inMaterialId].metallic_factor;

	if(materials[inMaterialId].texture > -1) {
        vec4 text = texture(textures[materials[inMaterialId].texture], inTexCoord);
        albedo = text.xyz;
        if(text.a < 0.1) {
            discard;
        }
    }
    else {
        albedo = materials[inMaterialId].base_color.xyz;
    }

    if(materials[inMaterialId].metallic_roughness_texture > -1) {
        vec4 values = texture(textures[materials[inMaterialId].metallic_roughness_texture], inTexCoord);
        roughness *= values.g;
        metallic *= values.b;
    }

    gbufferAlbedoMetallic = vec4(albedo, metallic);

    vec3 normal = normalize(inNormal);

    if(materials[inMaterialId].normal_texture > -1) {
        normal = getNormal();
    }

    gbufferNormalMotion = vec4(direction_to_octohedral(normal), compute_motion_vector(inOldPosition, inPosition));
    float linearDepth = gl_FragCoord.z / gl_FragCoord.w;
    float curvature = compute_curvature();
    //gbufferRoughnessDepthCurvatureMaterial = vec4(roughness, linearize_depth(gl_FragCoord.z, 0.1f, 1000.0f), curvature, inMaterialId);
    gbufferRoughnessDepthCurvatureMaterial = vec4(roughness, gl_FragCoord.z / gl_FragCoord.w, curvature, inMaterialId);
    gbufferUV = vec4(inTexCoord, inLightmapCoord);
}