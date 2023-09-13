// Configure ML
#define DISTANCE_ML_MAX_N 1024
#define DISTANCE_ML_MIN_ALPHA .01

DistanceMCState distance_mc_state_new() {
    DistanceMCState r = {0.0, 0, vec2(0)};
    return r;
}

// returns (mu, sigma)
vec2 distance_mc_state_get_normal_dist(const DistanceMCState distance_mc_state) {
    const vec2 moments = distance_mc_state.moments / (distance_mc_state.sum_w > 0.0 ? distance_mc_state.sum_w : 1.0);
    float sigma = sqrt(max(moments.y - moments.x * moments.x, 0.0));
    sigma = (distance_mc_state.N * distance_mc_state.N * sigma) / (distance_mc_state.N * distance_mc_state.N + 1.);
    return vec2(moments.x, sigma);
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void distance_mc_state_add_sample(inout DistanceMCState mc_state,
                         const float distance,
                         const float w) {
    mc_state.N = min(mc_state.N + 1, DISTANCE_ML_MAX_N);
    const float alpha = max(1.0 / mc_state.N, DISTANCE_ML_MIN_ALPHA);

    mc_state.sum_w   = mix(mc_state.sum_w,   w,          alpha);
    mc_state.moments = mix(mc_state.moments, w * vec2(distance, distance * distance), alpha);
}

#define distance_mc_buf_idx(index, resolution) (index.x + (resolution.x + 1) * index.y)
