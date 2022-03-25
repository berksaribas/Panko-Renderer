const float PI  = 3.14159265358979323846264;

float luminance(vec3 c) {
    return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 calculate_direct_lighting(vec3 albedo, float metallic, float roughness, vec3 normal, vec3 view, vec3 lightDir, vec3 lightColor) {
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);
	           
    // reflectance equation
    vec3 Lo = vec3(0.0);

    // calculate per-light radiance
    vec3 halfway = normalize(view + lightDir);
    vec3 radiance = lightColor;        
        
    // cook-torrance brdf
    float NDF = DistributionGGX(normal, halfway, roughness);        
    float G   = GeometrySmith(normal, view, lightDir, roughness);      
    vec3 F    = fresnelSchlick(max(dot(halfway, view), 0.0), F0);       
        
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;	  
        
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, view), 0.0) * max(dot(normal, lightDir), 0.0) + 0.0001;
    vec3 specular     = numerator / denominator;  
            
    // add to outgoing radiance Lo
    float NdotL = max(dot(normal, lightDir), 0.0);   
    
    vec3 c_diffuse = mix(albedo * (vec3(1.0f) - F0), vec3(0.0f), metallic);

    Lo += (kD * albedo / PI + specular) * radiance * NdotL; //does it make sense to divide the albedo by PI?
  
    return Lo;
}

vec3 calculate_indirect_lighting(vec3 albedo, float metallic, float roughness, vec3 normal, vec3 view, vec3 diffuseIrradiance, vec3 glossyIrradiance, sampler2D brdfLUT, vec3 directLight) {
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    vec3 F = fresnel_schlick_roughness(max(dot(normal, view), 0.0), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;	  

    vec3 diffuse = diffuseIrradiance * albedo;

    vec2 envBRDF  = texture(brdfLUT, vec2(max(dot(normal, view), 0.0), roughness)).rg;
    //vec3 specular = roughness < 0.75 ? (glossyIrradiance * (F * envBRDF.x + envBRDF.y)) : (diffuseIrradiance * (F * envBRDF.x + envBRDF.y));
    vec3 specular = glossyIrradiance * (F * envBRDF.x + envBRDF.y);
    
    
    return kD * diffuse + specular ;
}

vec3 calculate_indirect_lighting_nospecular(vec3 albedo, float metallic, float roughness, vec3 normal, vec3 view, vec3 diffuseIrradiance, vec3 glossyIrradiance) {
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    vec3 F = fresnel_schlick_roughness(max(dot(normal, view), 0.0), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;	  
    vec3 diffuse = diffuseIrradiance * albedo;

    return kD * diffuse + glossyIrradiance;
}

//[Heitz18] – Eric Heitz, Sampling the GGX Distribution of Visible Normals, JCGT 2018
vec3 sampleGGXVNDF(vec3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    vec3 Vh = normalize(vec3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1,0,0);
    vec3 T2 = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    float r = sqrt(U1);
    float phi = 2.0 * PI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    // Section 4.3: reprojection onto hemisphere
    vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    vec3 Ne = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

vec4 importance_sample_ggx(vec2 E, vec3 N, float Roughness)
{
    float a  = Roughness * Roughness;
    float m2 = a * a;

    float phi      = 2.0f * PI * E.x;
    float cosTheta = sqrt((1.0f - E.y) / (1.0f + (m2 - 1.0f) * E.y));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates - halfway vector
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    float d = (cosTheta * m2 - cosTheta) * cosTheta + 1;
    float D = m2 / (PI * d * d);

    float PDF = D * cosTheta;

    // from tangent-space H vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return vec4(normalize(sampleVec), PDF);
}