#version 460
layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vColor;

layout (location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform  CameraBuffer{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 lightPos;
	vec4 lightColor;
} cameraData;

void main()
{
	gl_PointSize = 10.0f;
	gl_Position = cameraData.viewproj * vec4(vPosition, 1.0);
	outColor = vColor;
}