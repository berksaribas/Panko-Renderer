#version 460

layout (location = 0) in vec2 InUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D source;

//
float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x)
{
   return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
//

void main(void) {
    vec3 texColor  = texture(source, InUv).rgb;

   texColor  *= 1;  // Hardcoded Exposure Adjustment
   
   float ExposureBias = 2.0f;
   vec3 curr = Uncharted2Tonemap(ExposureBias*texColor );
   
   vec3 whiteScale = vec3(1.0) / Uncharted2Tonemap(vec3(W));
   vec3 color = curr * whiteScale;
      
   vec3 retColor = pow(color, vec3(1/2.2));

    outFragColor = vec4(retColor, 1.0);
}