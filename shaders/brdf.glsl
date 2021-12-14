const float PI  = 3.14159265358979323846264;

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
    Lo += (kD * albedo + specular) * radiance * NdotL; //does it make sense to divide the albedo by PI?
  
    return Lo;
}

vec3 calculate_indirect_lighting(vec3 albedo, float metallic, float roughness, vec3 normal, vec3 view, vec3 diffuseIrradiance, vec3 glossyIrradiance) {
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 c_diffuse = mix(albedo * (vec3(1.0) - F0), vec3(0.0), metallic);
    vec3 F = fresnel_schlick_roughness(max(dot(normal, view), 0.0), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 diffuse = diffuseIrradiance * c_diffuse;
    vec3 specular = vec3(0); //TODO

    return (kD * diffuse + specular);
}