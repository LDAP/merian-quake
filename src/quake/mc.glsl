// ADAPTIVE GRID (sclaes with distance to camera)
#define MC_ADAPTIVE_GRID_MAX_WIDTH 100
#define MC_ADAPTIVE_GRID_MIN_WIDTH .1
#define MC_ADAPTIVE_GRID_POWER 4.
#define MC_ADAPTIVE_GRID_LEVELS 10
// Set the target for light cache resolution
#define MC_ADAPTIVE_GRID_TAN_ALPHA_HALF 0.002

// STATIC GRID (does not scale, for state exchange)
#define MC_STATIC_GRID_WIDTH 25.3

// Configure ML
#define ML_MAX_N 1024
#define ML_MIN_ALPHA .01


// ADAPTIVE GRID

uint mc_adaptive_level_for_pos(const vec3 pos, inout uint rng_state) {
    const float target_grid_width = clamp(2 * MC_ADAPTIVE_GRID_TAN_ALPHA_HALF * distance(pos, params.cam_x.xyz), MC_ADAPTIVE_GRID_MIN_WIDTH, MC_ADAPTIVE_GRID_MAX_WIDTH);
    const float level = MC_ADAPTIVE_GRID_LEVELS * pow((target_grid_width - MC_ADAPTIVE_GRID_MIN_WIDTH) / (MC_ADAPTIVE_GRID_MAX_WIDTH - MC_ADAPTIVE_GRID_MIN_WIDTH), 1 / MC_ADAPTIVE_GRID_POWER);
    return uint(round(level));
}

ivec3 mc_adpative_grid_idx_for_level_closest(const uint level, const vec3 pos, inout uint rng_state) {
    const float grid_width = pow(level / float(MC_ADAPTIVE_GRID_LEVELS), MC_ADAPTIVE_GRID_POWER) * (MC_ADAPTIVE_GRID_MAX_WIDTH - MC_ADAPTIVE_GRID_MIN_WIDTH) + MC_ADAPTIVE_GRID_MIN_WIDTH;
    return grid_idx_closest(pos, grid_width);
}

ivec3 mc_adaptive_grid_idx_for_level_interpolate(const uint level, const vec3 pos, inout uint rng_state) {
    const float grid_width = pow(level / float(MC_ADAPTIVE_GRID_LEVELS), MC_ADAPTIVE_GRID_POWER) * (MC_ADAPTIVE_GRID_MAX_WIDTH - MC_ADAPTIVE_GRID_MIN_WIDTH) + MC_ADAPTIVE_GRID_MIN_WIDTH;
    return grid_idx_interpolate(pos, grid_width, XorShift32(rng_state));
}

MCState mc_adaptive_load(const vec3 pos, const vec3 normal, inout uint rng_state) {
    const float rand = XorShift32(rng_state);
    const uint level = clamp(mc_adaptive_level_for_pos(pos, rng_state) + (rand < .2 ? 1 : (rand > .8 ? 2 : 0)), 0, MC_ADAPTIVE_GRID_LEVELS - 1);
    const ivec3 grid_idx = mc_adaptive_grid_idx_for_level_interpolate(level, pos, rng_state);
    const uint buf_idx = hash_grid_level(grid_idx, level, MC_ADAPTIVE_BUFFER_SIZE);

    MCState state = mc_states_adaptive[buf_idx].state;
    state.sum_w *= float(hash2_grid_level(grid_idx, level) == state.hash);
    return state;
}

void mc_adaptive_save(in MCState mc_state, const vec3 pos, const vec3 normal, inout uint rng_state) {
    const float rand = XorShift32(rng_state);
    const uint level = clamp(mc_adaptive_level_for_pos(pos, rng_state) + (rand < .2 ? 1 : (rand > .8 ? 2 : 0)), 0, MC_ADAPTIVE_GRID_LEVELS - 1);
    const ivec3 grid_idx = mc_adaptive_grid_idx_for_level_interpolate(level, pos, rng_state);
    const uint buf_idx = hash_grid_level(grid_idx, level, MC_ADAPTIVE_BUFFER_SIZE);

    mc_state.hash = hash2_grid_level(grid_idx, level);
    mc_states_adaptive[buf_idx].state = mc_state;
}


// STATIC GRID

MCState mc_static_load(const vec3 pos, const vec3 normal, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, MC_STATIC_GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, MC_STATIC_BUFFER_SIZE);
    const uint state_idx = uint(XorShift32(rng_state) * MC_STATIC_VERTEX_STATE_COUNT);
    
    MCState state = mc_states_static[buf_idx].states[state_idx];
    state.sum_w *= float(hash2_grid(grid_idx) == state.hash);

    return state;
}

void mc_static_save(in MCState mc_state, const vec3 pos, const vec3 normal, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, MC_STATIC_GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, MC_STATIC_BUFFER_SIZE);
    const uint state_idx = uint(XorShift32(rng_state) * MC_STATIC_VERTEX_STATE_COUNT);

    mc_state.hash = hash2_grid(grid_idx);
    mc_states_static[buf_idx].states[state_idx] = mc_state;
}


// GENERAL

MCState mc_state_new(const vec3 pos, const vec3 normal, inout uint rng_state) {
    MCState r = {vec3(0.0), 0.0, 0, 0.0, 0};
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

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void mc_state_add_sample(inout MCState mc_state,
                         const vec3 pos,         // position where the ray started
                         const float w,          // goodness
                         const vec3 target) {    // ray hit point
    mc_state.N = min(mc_state.N + 1, ML_MAX_N);
    const float alpha = max(1.0 / mc_state.N, ML_MIN_ALPHA);

    mc_state.sum_w   = mix(mc_state.sum_w,   w,          alpha);
    mc_state.sum_tgt = mix(mc_state.sum_tgt, w * target, alpha);
    mc_state.sum_len = mix(mc_state.sum_len, w * max(0, dot(normalize(target - pos), mc_state_dir(mc_state, pos))), alpha);
}

bool mc_state_valid(const MCState mc_state) {
    return mc_state.sum_w > 0.0;
}
