float luminance(vec3 c) {
    return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

float normal_edge_stopping_weight(vec3 center_normal, vec3 sample_normal, float power)
{
    return pow(clamp(dot(center_normal, sample_normal), 0.0f, 1.0f), power);
}

// ------------------------------------------------------------------------

float depth_edge_stopping_weight(float center_depth, float sample_depth, float phi)
{
    return -abs(center_depth - sample_depth) / phi;
}

// ------------------------------------------------------------------

float luma_edge_stopping_weight(float center_luma, float sample_luma, float phi)
{
    return abs(center_luma - sample_luma) / phi;
}

float computeWeight(
	float depthCenter, float depthP, float phiDepth,
	vec3 normalCenter, vec3 normalP, float normPower,
	float luminanceIndirectCenter, float luminanceIndirectP, float phiIndirect)
{
    float wZ      = depth_edge_stopping_weight(depthCenter, depthP, phiDepth);
    float wNormal = normal_edge_stopping_weight(normalCenter, normalP, normPower);
    float wL      = luma_edge_stopping_weight(luminanceIndirectCenter, luminanceIndirectP, phiIndirect);
    float w = exp(0.0 - max(wL, 0.0) - max(wZ, 0.0)) * wNormal;

    return w;
}
