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

float c_x0 = -1.0;
float c_x1 =  0.0;
float c_x2 =  1.0;
float c_x3 =  2.0;
    
//=======================================================================================
vec3 CubicLagrange (vec3 A, vec3 B, vec3 C, vec3 D, float t)
{
    return
        A * 
        (
            (t - c_x1) / (c_x0 - c_x1) * 
            (t - c_x2) / (c_x0 - c_x2) *
            (t - c_x3) / (c_x0 - c_x3)
        ) +
        B * 
        (
            (t - c_x0) / (c_x1 - c_x0) * 
            (t - c_x2) / (c_x1 - c_x2) *
            (t - c_x3) / (c_x1 - c_x3)
        ) +
        C * 
        (
            (t - c_x0) / (c_x2 - c_x0) * 
            (t - c_x1) / (c_x2 - c_x1) *
            (t - c_x3) / (c_x2 - c_x3)
        ) +       
        D * 
        (
            (t - c_x0) / (c_x3 - c_x0) * 
            (t - c_x1) / (c_x3 - c_x1) *
            (t - c_x2) / (c_x3 - c_x2)
        );
}

//=======================================================================================
vec3 BicubicLagrangeTextureSample (vec2 uv, int mip)
{
    vec2 tSize = textureSize(glossyReflections, int(mip));
    vec2 pixel = uv * tSize + 0.5;
    
    vec2 frac = fract(pixel);

    vec2 c_onePixel = vec2(1.0) / tSize;
    vec2 c_twoPixels = vec2(2.0) / tSize;

    pixel = floor(pixel) / tSize - c_onePixel/2.0;
    
    vec3 C00 = textureLod(glossyReflections, pixel + vec2(-c_onePixel.x ,-c_onePixel.y), mip).rgb;
    vec3 C10 = textureLod(glossyReflections, pixel + vec2( 0.0        ,-c_onePixel.y), mip).rgb;
    vec3 C20 = textureLod(glossyReflections, pixel + vec2( c_onePixel.x ,-c_onePixel.y), mip).rgb;
    vec3 C30 = textureLod(glossyReflections, pixel + vec2( c_twoPixels.x,-c_onePixel.y), mip).rgb;
    
    vec3 C01 = textureLod(glossyReflections, pixel + vec2(-c_onePixel.x , 0.0), mip).rgb;
    vec3 C11 = textureLod(glossyReflections, pixel + vec2( 0.0        , 0.0), mip).rgb;
    vec3 C21 = textureLod(glossyReflections, pixel + vec2( c_onePixel.x , 0.0), mip).rgb;
    vec3 C31 = textureLod(glossyReflections, pixel + vec2( c_twoPixels.x, 0.0), mip).rgb;    
    
    vec3 C02 = textureLod(glossyReflections, pixel + vec2(-c_onePixel.x , c_onePixel.y), mip).rgb;
    vec3 C12 = textureLod(glossyReflections, pixel + vec2( 0.0        , c_onePixel.y), mip).rgb;
    vec3 C22 = textureLod(glossyReflections, pixel + vec2( c_onePixel.x , c_onePixel.y), mip).rgb;
    vec3 C32 = textureLod(glossyReflections, pixel + vec2( c_twoPixels.x, c_onePixel.y), mip).rgb;    
    
    vec3 C03 = textureLod(glossyReflections, pixel + vec2(-c_onePixel.x , c_twoPixels.y), mip).rgb;
    vec3 C13 = textureLod(glossyReflections, pixel + vec2( 0.0        , c_twoPixels.y), mip).rgb;
    vec3 C23 = textureLod(glossyReflections, pixel + vec2( c_onePixel.x , c_twoPixels.y), mip).rgb;
    vec3 C33 = textureLod(glossyReflections, pixel + vec2( c_twoPixels.x, c_twoPixels.y), mip).rgb;    
    
    vec3 CP0X = CubicLagrange(C00, C10, C20, C30, frac.x);
    vec3 CP1X = CubicLagrange(C01, C11, C21, C31, frac.x);
    vec3 CP2X = CubicLagrange(C02, C12, C22, C32, frac.x);
    vec3 CP3X = CubicLagrange(C03, C13, C23, C33, frac.x);
    
    return CubicLagrange(CP0X, CP1X, CP2X, CP3X, frac.y);
}

vec4 interpolate_bicubic(vec2 p, vec4 q[16])
{
    vec4 a00 = q[5];
	vec4 a01 = -.5*q[4]+.5*q[6];
	vec4 a02 = q[4]-2.5*q[5]+2.0*q[6]-.5*q[7];
	vec4 a03 = -.5*q[4]+1.5*q[5]-1.5*q[6]+.5*q[7];
	vec4 a10 = -.5*q[1]+.5*q[9];
	vec4 a11 = .25*q[0]-.25*q[2]-.25*q[8]+.25*q[10];
	vec4 a12 = -.5*q[0]+1.25*q[1]-q[2]+.25*q[3]+.5*q[8]-1.25*q[9]+q[10]-.25*q[11];
	vec4 a13 = .25*q[0]-.75*q[1]+.75*q[2]-.25*q[3]-.25*q[8]+.75*q[9]-.75*q[10]
        +.25*q[11];
	vec4 a20 = q[1]-2.5*q[5]+2.0*q[9]-.5*q[13];
	vec4 a21 = -.5*q[0]+0.5*q[2]+1.25*q[4]-1.25*q[6]-q[8]+q[10]+.25*q[12]-.25*q[14];
	vec4 a22 = q[0]-2.5*q[1]+2.0*q[2]-.5*q[3]-2.5*q[4]+6.25*q[5]-5.0*q[6]+1.25*q[7]
        +2.0*q[8]-5.0*q[9]+4.0*q[10]-q[11]-.5*q[12]+1.25*q[13]-q[14]+.25*q[15];
	vec4 a23 = -.5*q[0]+1.5*q[1]-1.5*q[2]+.5*q[3]+1.25*q[4]-3.75*q[5]+3.75*q[6]
        - 1.25*q[7]-q[8]+3.0*q[9]-3.0*q[10]+q[11]+.25*q[12]-.75*q[13]+.75*q[14]-.25*q[15];
	vec4 a30 = -.5*q[1]+1.5*q[5]-1.5*q[9]+.5*q[13];
	vec4 a31 = .25*q[0]-0.25*q[2]-0.75*q[4]+.75*q[6]+.75*q[8]-.75*q[10]-.25*q[12]+.25*q[14];
	vec4 a32 = -.5*q[0]+1.25*q[1]-q[2]+.25*q[3]+1.5*q[4]-3.75*q[5]+3.0*q[6]-.75*q[7]-1.5*q[8]
        + 3.75*q[9]-3.0*q[10]+.75*q[11]+.5*q[12]-1.25*q[13]+q[14]-.25*q[15];
	vec4 a33 = .25*q[0]-0.75*q[1]+0.75*q[2]-.25*q[3]-.75*q[4]+2.25*q[5]-2.25*q[6]+.75*q[7]
        +.75*q[8]-2.25*q[9]+2.25*q[10]-.75*q[11]-.25*q[12]+.75*q[13]-.75*q[14] 
        +.25*q[15];
    float x2 = p.x*p.x;
    float x3 = x2*p.x;
    float y2 = p.y*p.y;
    float y3 = y2*p.y;
   	return (a00+a01*p.y+a02*y2+a03*y3)+
		   (a10+a11*p.y+a12*y2+a13*y3)*p.x+
		   (a20+a21*p.y+a22*y2+a23*y3)*x2+
		   (a30+a31*p.y+a32*y2+a33*y3)*x3;
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

const float EPSILON = 0.001;

float bilateralWeight(vec3 n1, vec3 n2, float d1, float d2, float hd1, float hd2 ) {
	if(d1 < 0 && d2 < 0) return 0;
	if(d1 < 0) return 1;

	float normalWeight = max(0, pow(dot(n1,n2), 2)) > 0.5 ? 1 : 0;
	float depthWeight = abs(d1-d2) > 0.5 ? 0 : 1;
    
    float hitDistanceWeight = 1 / (EPSILON + abs(hd1-hd2) / 100.0);

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
    float tHit = 0;
    if(cameraData.useStochasticSpecular == 0)
    {
        float mipChannel = 0;
        vec3 view = normalize(cameraData.cameraPos.xyz - inWorldPosition.xyz);
        
        tHit = textureLod(glossyReflections, InUv, 0).w;
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

        mipChannel = clamp(log2(distance(hitSS1.xy, hitSS2.xy)), 0.0f, 7);

        bool bicubic = false;
        vec4 lowerMip = vec4(0);
        vec4 higherMip = vec4(0);
        if(bicubic) {
            lowerMip = vec4(BicubicLagrangeTextureSample(InUv ,int(mipChannel)), 1.0);//customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(mipChannel)), int(mipChannel), inMaterialId);
            higherMip = vec4(BicubicLagrangeTextureSample(InUv ,int(mipChannel) + 1), 1.0);//customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(mipChannel) + 1), int(mipChannel) + 1, inMaterialId);
        }
        else {
            lowerMip = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(mipChannel)), int(mipChannel), inMaterialId);
            higherMip = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(mipChannel) + 1), int(mipChannel) + 1, inMaterialId);
        }
        reflectionColor = mix(lowerMip, higherMip, mipChannel - int(mipChannel));
        //somewhat experimental
        reflectionColor =  mix(reflectionColor, texture(indirectLightMap, inLightmapCoord), (pow(64, mipChannel / 7) - 1) / 63);
        reflectionColor =  mix(reflectionColor, texture(indirectLightMap, inLightmapCoord), (pow(16, roughness) - 1) / 15);
    }
    else {
        if(roughness > 0.05) {
            reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).xyz, 1.0);
        }
        else {
            reflectionColor = vec4(textureLod(glossyReflections, InUv, 0).xyz, 1.0);
        }
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

    //outFragColor = vec4(textureLod(glossyReflections, InUv, 0).xyz, 1.0);
    //outFragColor = vec4(textureLod(glossyNormal, InUv, 5).xyz, 1.0);
    //outFragColor = reflectionColor;
    //outFragColor = customBilinear(InUv - vec2(0) / textureSize(glossyReflections, int(0)), int(5), inMaterialId);
    outFragColor = vec4(outColor, 1.0);
}

// can i make the entire thing work with bicubic filters instead?