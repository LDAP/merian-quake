#define MC_MAX_GRID_WIDTH 100
#define MC_MIN_GRID_WIDTH .1
#define MC_LEVELS 5
// Set the target for light cache resolution
#define MC_TAN_ALPHA_HALF 0.005

// Configure ML
#define ML_MAX_N 128
#define ML_MIN_ALPHA .01

uint mc_level_for_pos(const vec3 pos, inout uint rng_state) {
    const float target_grid_width = clamp(2 * MC_TAN_ALPHA_HALF * distance(pos, params.cam_x.xyz), MC_MIN_GRID_WIDTH, MC_MAX_GRID_WIDTH);
    const float level = MC_LEVELS * pow((target_grid_width - MC_MIN_GRID_WIDTH) / (MC_MAX_GRID_WIDTH - MC_MIN_GRID_WIDTH), 1 / 9.);
    return uint(round(level));
}

ivec3 mc_grid_idx_for_level_closest(const uint level, const vec3 pos, inout uint rng_state) {
    const float grid_width = pow(level / float(MC_LEVELS), 9.) * (MC_MAX_GRID_WIDTH - MC_MIN_GRID_WIDTH) + MC_MIN_GRID_WIDTH;
    return grid_idx_closest(pos, grid_width);
}

ivec3 mc_grid_idx_for_level_interpolate(const uint level, const vec3 pos, inout uint rng_state) {
    const float grid_width = pow(level / float(MC_LEVELS), 9.) * (MC_MAX_GRID_WIDTH - MC_MIN_GRID_WIDTH) + MC_MIN_GRID_WIDTH;
    return grid_idx_interpolate(pos, grid_width, XorShift32(rng_state));
}

MCState mc_state_new() {
    MCState r = {vec3(0.0), 0.0, 0, 0.0, ivec3(0), 0};
    return r;
}

// return normalized direction (from pos)
vec3 mc_state_dir(const MCState mc_state, const vec3 pos) {
    const vec3 tgt = mc_state.sum_tgt / (mc_state.sum_w > 0.0 ? mc_state.sum_w : 1.0);
    return normalize(tgt - pos);
}

// returns the vmf lobe vec4(direction, kappa) for a position
vec4 mc_state_get_vmf(const MCState mc_state, const vec3 pos) {
    float r = mc_state.sum_len / mc_state.sum_w; // = mean cosine in [0,1]

    // Jo
    // const vec3 tgt = mc_state.sum_tgt / (mc_state.sum_w > 0.0 ? mc_state.sum_w : 1.0);
    // const float d = length(tgt - pos);
    // const float rp = 1.0 -  1.0 / clamp(50.0 * d, 0.0, 6500.0);

    // r = (mc_state.N * mc_state.N * r + ML_PRIOR_N * rp) / (mc_state.N * mc_state.N + ML_PRIOR_N);
    // Addis
    r = (mc_state.N * mc_state.N * r) / (mc_state.N * mc_state.N + ml_prior());
    return vec4(mc_state_dir(mc_state, pos), (3.0 * r - r * r * r) / (1.0 - r * r));
}

MCState mc_state_load(const vec3 pos, const vec3 normal, inout uint rng_state) {
    const float rand = XorShift32(rng_state);
    const uint level = clamp(mc_level_for_pos(pos, rng_state) + (rand < .2 ? 1 : (rand > .8 ? 2 : 0)), 0, MC_LEVELS - 1);
    const ivec3 grid_idx = mc_grid_idx_for_level_interpolate(level, pos, rng_state);
    const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, MC_BUFFER_SIZE);

    MCState state = mc_states[buf_idx].state;
    state.sum_w *= float(grid_idx == state.grid_idx && level == state.level);
    return state;
}

// return true if a valid state was found
bool mc_state_load_resample(out MCState mc_state, const vec3 pos, const vec3 normal, inout uint rng_state) {
    float score_sum = 0;
    [[unroll]]
    for (int i = 0; i < 5; i++) {
        MCState state = mc_state_load(pos, normal, rng_state);
        const float candidate_score = state.sum_w;
        // candidate_score *= exp(- abs(yuv_luminance(light_cache_get(pos, normal, rng_state).rgb) - state.sum_w));
        // candidate_score *= (1. - (level + 1) / (ML_MAX_N + 1));
        // candidate_score *= max(dot(state.normal, normal), 0.);

        score_sum += candidate_score;
        if (XorShift32(rng_state) < candidate_score / score_sum) {
            // we use here that comparison with NaN is false, that happens if candidate_score == 0 and sum == 0; 
            mc_state = state;
        }
    }

    return score_sum > 0.0;
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void mc_state_add_sample(inout MCState mc_state,
                         const vec3 pos,         // position where the ray started
                         const float w,          // goodness
                         const vec3 target) {    // ray hit point
    mc_state.N = min(mc_state.N + 1, ML_MAX_N);
    const float alpha = max(1.0 / mc_state.N, ML_MIN_ALPHA);

    mc_state.sum_w   = mix(mc_state.sum_w,   w,          alpha);
    mc_state.sum_tgt = mix(mc_state.sum_tgt, w * target, alpha);
    // is this the same?
    mc_state.sum_len = mix(mc_state.sum_len, w * dot(normalize(target - pos), mc_state_dir(mc_state, pos)), alpha);

    mc_state.sum_len = max(mc_state.sum_len, 0);
}

void mc_state_save(in MCState mc_state, const vec3 pos, const vec3 normal, inout uint rng_state) {
    // update state that was used for sampling
    {
        const uint buf_idx = hash_grid_normal_level(mc_state.grid_idx, normal, mc_state.level, MC_BUFFER_SIZE);
        // const uint old = atomicExchange(mc_states[buf_idx].lock, params.frame);
        // if (old != params.frame)
            mc_states[buf_idx].state = mc_state;
    }

    // update other levels
    [[unroll]]
    for (uint i = 0; i < 1; i++) {
        const float rand = XorShift32(rng_state);
        mc_state.level = clamp(mc_level_for_pos(pos, rng_state) + (rand < .2 ? 1 : (rand > .8 ? 2 : 0)), 0, MC_LEVELS - 1);
        mc_state.grid_idx = mc_grid_idx_for_level_interpolate(mc_state.level, pos, rng_state);
        const uint buf_idx = hash_grid_normal_level(mc_state.grid_idx, normal, mc_state.level, MC_BUFFER_SIZE);

        mc_states[buf_idx].state = mc_state;
    }
}

bool mc_state_valid(const MCState mc_state) {
    return mc_state.sum_w > 0.0;
}
