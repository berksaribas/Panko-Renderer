#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_debug_printf : enable

#define RAYTRACING

#include "common.glsl"
#include "brdf.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT ReflectionPayload payload;

layout(set = 0, binding = 1) uniform _SceneDesc { GPUSceneDesc sceneDesc; };
layout(std140, set = 0, binding = 2) readonly buffer _MeshInfo { GPUMeshInfo meshInfos[]; };

layout(set = 1, binding = 0) uniform _CameraBuffer { GPUCameraData cameraData; };
layout(set = 1, binding = 1) uniform _ShadowMapData { GPUShadowMapData shadowMapData; };

layout(std140, set = 2, binding = 0) readonly buffer _ObjectBuffer{ GPUObjectData objects[]; };

layout(set = 3, binding = 0) uniform sampler2D gbufferPositionMaterial;
layout(set = 3, binding = 1) uniform sampler2D gbufferNormal;
layout(set = 3, binding = 2) uniform sampler2D gbufferUV;

layout(set = 4, binding = 0) uniform sampler2D[] textures;

layout(std140, set = 5, binding = 0) readonly buffer MaterialBuffer{ GPUBasicMaterialData materials[]; };

layout(set = 6, binding = 0) uniform sampler2D shadowMap;

layout(set = 7, binding = 0) uniform sampler2D indirectLightMap;


layout(buffer_reference, scalar) readonly buffer Vertices { vec3 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };
layout(buffer_reference, scalar) readonly buffer Normals { vec3 n[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords { vec2 t[]; };
layout(buffer_reference, scalar) readonly buffer LightmapTexCoords { vec2 t[]; };

#include "shadow.glsl"

void main()
{
    // Object data
    GPUMeshInfo meshInfo = meshInfos[gl_InstanceCustomIndexEXT];

    Indices indices = Indices(sceneDesc.indexAddress);
    Vertices vertices = Vertices(sceneDesc.vertexAddress);
    Normals normals = Normals(sceneDesc.normalAddress);
    TexCoords texCoords = TexCoords(sceneDesc.uvAddress);
    LightmapTexCoords lighmapTexCoords = LightmapTexCoords(sceneDesc.lightmapUvAddress);

    uint indexOffset = meshInfo.indexOffset + 3 * gl_PrimitiveID;
  
    const uint ind0 = indices.i[indexOffset + 0];
    const uint ind1 = indices.i[indexOffset + 1];
    const uint ind2 = indices.i[indexOffset + 2];

    const vec3 v0 = vertices.v[ind0 + meshInfo.vertexOffset];
    const vec3 v1 = vertices.v[ind1 + meshInfo.vertexOffset];
    const vec3 v2 = vertices.v[ind2 + meshInfo.vertexOffset];

    const vec3 n0 = normals.n[ind0 + meshInfo.vertexOffset];
    const vec3 n1 = normals.n[ind1 + meshInfo.vertexOffset];
    const vec3 n2 = normals.n[ind2 + meshInfo.vertexOffset];

    const vec2 uv0 = texCoords.t[ind0 + meshInfo.vertexOffset];
    const vec2 uv1 = texCoords.t[ind1 + meshInfo.vertexOffset];
    const vec2 uv2 = texCoords.t[ind2 + meshInfo.vertexOffset];

    const vec2 lightmapUv0 = lighmapTexCoords.t[ind0 + meshInfo.vertexOffset];
    const vec2 lightmapUv1 = lighmapTexCoords.t[ind1 + meshInfo.vertexOffset];
    const vec2 lightmapUv2 = lighmapTexCoords.t[ind2 + meshInfo.vertexOffset];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    const vec3 pos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0)) * 0.3;

    const vec3 nrm = n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z;
    const vec3 worldNrm = normalize(vec3(nrm * gl_WorldToObjectEXT));

    const vec2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
    
    const vec2 lightmapUv = lightmapUv0 * barycentrics.x + lightmapUv1 * barycentrics.y + lightmapUv2 * barycentrics.z;
    
    //calculate lighting
    int inMaterialId = objects[gl_InstanceCustomIndexEXT].material_id;

    vec3 albedo = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive_color = materials[inMaterialId].emissive_color;

    float roughness = materials[inMaterialId].roughness_factor;
    float metallic = materials[inMaterialId].metallic_factor;

    if(emissive_color.r > 0 || emissive_color.g > 0 || emissive_color.b > 0) {
        albedo = vec3(1.0f, 1.0f, 1.0f);
    }
    else {
	    if(materials[inMaterialId].texture > -1) {
            albedo = pow(texture(textures[materials[inMaterialId].texture], uv).xyz, vec3(2.2));
        }
        else {
          albedo = materials[inMaterialId].base_color.xyz;
        }
    }

    vec4 shadowPos = biasMat * shadowMapData.depthMVP * vec4(worldPos.xyz, 1.0);
    float shadow = sample_shadow_map_evsm(shadowPos / shadowPos.w);

    roughness = 1;
    metallic = 0;

    vec3 directLight = calculate_direct_lighting(albedo, metallic, roughness, normalize(worldNrm), normalize(gl_ObjectRayOriginEXT * 0.3  - worldPos.xyz), normalize(cameraData.lightPos).xyz, cameraData.lightColor.xyz) * shadow;
    vec3 indirectLight = calculate_indirect_lighting_nospecular(albedo, metallic, roughness, normalize(worldNrm), normalize(gl_ObjectRayOriginEXT * 0.3  - worldPos.xyz), texture(indirectLightMap, lightmapUv / cameraData.lightmapInputSize).xyz, vec3(0));
        
    payload.color = directLight + indirectLight;
    payload.hitDistance = (gl_RayTminEXT + gl_HitTEXT) * 0.3;
    payload.normal = normalize(worldNrm);
}
