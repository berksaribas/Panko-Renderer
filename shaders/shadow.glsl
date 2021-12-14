float linstep(float minv, float maxv, float v)
{
    return maxv == minv ? 1.0 : clamp((v - minv) / (maxv - minv), 0.0, 1.0);
}

vec2 warp_depth(float depth, vec2 exponents)
{
    // Rescale depth into [-1, 1]
    depth = 2.0f * depth - 1.0f;
    float pos =  exp( exponents.x * depth);
    float neg = -exp(-exponents.y * depth);
    return vec2(pos, neg);
}

float reduce_light_bleeding(float pMax, float amount)
{
  // Remove the [0, amount] tail and linearly rescale (amount, 1].
   return linstep(amount, 1.0f, pMax);
}

float chebyshev_upper_bound(vec2 moments, float mean, float minVariance,
                          float lightBleedingReduction)
{
    // Compute variance
    float variance = moments.y - (moments.x * moments.x);
    variance = max(variance, minVariance);

    // Compute probabilistic upper bound
    float d = mean - moments.x;
    float pMax = variance / (variance + (d * d));

    pMax = reduce_light_bleeding(pMax, lightBleedingReduction);

    // One-tailed Chebyshev
    return (mean <= moments.x ? 1.0f : pMax);
}

float sample_shadow_map_evsm(in vec4 shadowPos)
{
    vec2 exponents = vec2(shadowMapData.positiveExponent, shadowMapData.negativeExponent);
    vec2 warpedDepth = warp_depth(shadowPos.z, exponents);

    vec4 occluder = texture(shadowMap, shadowPos.st);

    // Derivative of warping at depth
    vec2 depthScale = shadowMapData.VSMBias * 0.01f * exponents * warpedDepth;
    vec2 minVariance = depthScale * depthScale;

    float posContrib = chebyshev_upper_bound(occluder.xz, warpedDepth.x, minVariance.x, shadowMapData.LightBleedingReduction);
    float negContrib = chebyshev_upper_bound(occluder.yw, warpedDepth.y, minVariance.y, shadowMapData.LightBleedingReduction);
    return min(posContrib, negContrib);
}

float textureProj(vec4 shadowCoord, vec2 off)
{
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
		float dist = texture( shadowMap, shadowCoord.st + off ).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z ) 
		{
			shadow = 0;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc)
{
    ivec2 texDim = textureSize(shadowMap, 0);
    float scale = 1.5;
    float dx = scale * 1.0 / float(texDim.x);
    float dy = scale * 1.0 / float(texDim.y);

    float shadowFactor = 0.0;
    int count = 0;
    int range = 1;
    
    for (int x = -range; x <= range; x++)
    {
        for (int y = -range; y <= range; y++)
        {
            shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
            count++;
        }
    
    }
    return shadowFactor / count;
}

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );