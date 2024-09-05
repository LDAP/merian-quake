// ADAPTIVE GRID (sclaes with distance to camera)
#define MC_ADAPTIVE_GRID_MAX_WIDTH 100
#define MC_ADAPTIVE_GRID_MIN_WIDTH .01
#define MC_ADAPTIVE_GRID_POWER 4.

// Configure ML
#define ML_MAX_N 1024
#define ML_MIN_ALPHA .01

// GENERAL

MCState mc_state_new(const vec3 pos, const vec3 normal) {
    MCState r = {vec3(0.0), 0.0, 0, 0.0, vec3(0), 0.0, 0};
    return r;
}

// return normalized direction (from pos)
#define mc_state_dir(mc_state, pos) normalize((mc_state.w_tgt / (mc_state.sum_w > 0.0 ? mc_state.sum_w : 1.0)) - pos)

#define mc_state_pos(mc_state) (mc_state.w_tgt / (mc_state.sum_w > 0.0 ? mc_state.sum_w : 1.0))

float mc_state_mean_cos(const MCState mc_state) {
    return (mc_state.N * mc_state.N * (mc_state.w_cos / mc_state.sum_w)) / (mc_state.N * mc_state.N + DIR_GUIDE_PRIOR);
}

float mc_state_kappa(const MCState mc_state) {
    const float r = (mc_state.N * mc_state.N * (mc_state.w_cos / mc_state.sum_w)) / (mc_state.N * mc_state.N + DIR_GUIDE_PRIOR);
    return (3.0 * r - r * r * r) / (1.0 - r * r);
}

// returns the vmf lobe vec4(direction, kappa) for a position
vec4 mc_state_get_vmf(const MCState mc_state, const vec3 pos) {
    const float r = (mc_state.N * mc_state.N * (mc_state.w_cos / mc_state.sum_w)) / (mc_state.N * mc_state.N + DIR_GUIDE_PRIOR);
    return vec4(mc_state_dir(mc_state, pos), (3.0 * r - r * r * r) / (1.0 - r * r));
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void mc_state_add_sample(inout MCState mc_state,
                         const vec3 pos,         // position where the ray started
                         const float w,          // goodness
                         const vec3 target, const vec3 target_mv) {    // ray hit point
    mc_state.N = min(mc_state.N + 1, ML_MAX_N);
    const float alpha = max(1.0 / mc_state.N, ML_MIN_ALPHA);

    mc_state.sum_w = mix(mc_state.sum_w, w,          alpha);
    mc_state.w_tgt = mix(mc_state.w_tgt, w * target, alpha);
    mc_state.w_cos = mix(mc_state.w_cos, w * max(0, dot(normalize(target - pos), mc_state_dir(mc_state, pos))), alpha);
    //mc_state.w_cos = length(mix(mc_state.w_cos * mc_state_dir(mc_state, pos), w * normalize(target - pos), alpha));

    mc_state.mv = target_mv;
    mc_state.T = params.cl_time;
}

void mc_state_reweight(inout MCState mc_state, const float factor) {
    mc_state.sum_w *= factor;
    mc_state.w_tgt *= factor;
    mc_state.w_cos *= factor;
}
#define mc_state_valid(mc_state) (mc_state.sum_w > 0.0)

// ADAPTIVE GRID

uint mc_adaptive_level_for_pos(const vec3 pos) {
    const float target_grid_width = clamp(2 * MC_ADAPTIVE_GRID_TAN_ALPHA_HALF * distance(pos, params.cam_x.xyz), MC_ADAPTIVE_GRID_MIN_WIDTH, MC_ADAPTIVE_GRID_MAX_WIDTH);
    const float level = MC_ADAPTIVE_GRID_LEVELS * pow((target_grid_width - MC_ADAPTIVE_GRID_MIN_WIDTH) / (MC_ADAPTIVE_GRID_MAX_WIDTH - MC_ADAPTIVE_GRID_MIN_WIDTH), 1 / MC_ADAPTIVE_GRID_POWER);
    return uint(round(level));
}

ivec3 mc_adpative_grid_idx_for_level_closest(const uint level, const vec3 pos) {
    const float grid_width = pow(level / float(MC_ADAPTIVE_GRID_LEVELS), MC_ADAPTIVE_GRID_POWER) * (MC_ADAPTIVE_GRID_MAX_WIDTH - MC_ADAPTIVE_GRID_MIN_WIDTH) + MC_ADAPTIVE_GRID_MIN_WIDTH;
    return grid_idx_closest(pos, grid_width);
}

ivec3 mc_adaptive_grid_idx_for_level_interpolate(const uint level, const vec3 pos) {
    const float grid_width = pow(level / float(MC_ADAPTIVE_GRID_LEVELS), MC_ADAPTIVE_GRID_POWER) * (MC_ADAPTIVE_GRID_MAX_WIDTH - MC_ADAPTIVE_GRID_MIN_WIDTH) + MC_ADAPTIVE_GRID_MIN_WIDTH;
    return grid_idx_interpolate(pos, grid_width, XorShift32(rng_state));
}

// returns (buffer_index, hash)
void mc_adaptive_buffer_index(const vec3 pos, const vec3 normal, out uint buffer_index, out uint hash) {
    const float rand = XorShift32(rng_state);
    const uint level = clamp(mc_adaptive_level_for_pos(pos) + (rand < .2 ? 1 : (rand > .8 ? 2 : 0)), 0, MC_ADAPTIVE_GRID_LEVELS - 1);
    const ivec3 grid_idx = mc_adaptive_grid_idx_for_level_interpolate(level, pos);
    buffer_index = hash_grid_normal_level(grid_idx, normal, level, MC_ADAPTIVE_BUFFER_SIZE);
    hash = hash2_grid_level(grid_idx, level);
}

void mc_adaptive_finalize_load(inout MCState mc_state, const uint hash) {
    mc_state.sum_w *= float(hash == mc_state.hash);
    mc_state.w_tgt += mc_state.sum_w * (params.cl_time - mc_state.T) * mc_state.mv;
}

void mc_adaptive_load(out MCState mc_state, const vec3 pos, const vec3 normal) {
    uint buffer_index, hash;
    mc_adaptive_buffer_index(pos, normal, buffer_index, hash);
    mc_state = mc_states[buffer_index];
    mc_adaptive_finalize_load(mc_state, hash);
}

void mc_adaptive_save(in MCState mc_state, const vec3 pos, const vec3 normal) {
    uint buffer_index, hash;
    mc_adaptive_buffer_index(pos, normal, buffer_index, hash);

    mc_state.hash = hash;
    mc_states[buffer_index] = mc_state;
}


// STATIC GRID

// returns (buffer_index, hash)
void mc_static_buffer_index(const vec3 pos, out uint buffer_index, out uint hash) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, MC_STATIC_GRID_WIDTH, XorShift32(rng_state));
    buffer_index = hash_grid(grid_idx, MC_STATIC_BUFFER_SIZE) + MC_ADAPTIVE_BUFFER_SIZE;
    hash = hash2_grid(grid_idx);
}

void mc_static_finalize_load(inout MCState mc_state, const uint hash) {
    mc_state.sum_w *= float(hash == mc_state.hash);
    mc_state.w_tgt += mc_state.sum_w * (params.cl_time - mc_state.T) * mc_state.mv;
}

void mc_static_finalize_load(inout MCState mc_state, const uint hash, const vec3 pos, const vec3 normal) {
    mc_state.sum_w *= float(hash == mc_state.hash);
    mc_state.sum_w *= float(dot(normal, mc_state_dir(mc_state, pos)) > 0.);
    mc_state.w_tgt += mc_state.sum_w * (params.cl_time - mc_state.T) * mc_state.mv;
}

void mc_static_load(out MCState mc_state, const vec3 pos) {
    uint buffer_index, hash;
    mc_static_buffer_index(pos, buffer_index, hash);
    mc_state = mc_states[buffer_index];
    mc_static_finalize_load(mc_state, hash);
}

void mc_static_load(out MCState mc_state, const vec3 pos, const vec3 normal) {
    uint buffer_index, hash;
    mc_static_buffer_index(pos, buffer_index, hash);
    mc_state = mc_states[buffer_index];
    mc_static_finalize_load(mc_state, hash, pos, normal);
}

void mc_static_save(in MCState mc_state, const vec3 pos, const vec3 normal) {
    uint buffer_index, hash;
    mc_static_buffer_index(pos, buffer_index, hash);

    mc_state.hash = hash;
    mc_states[buffer_index] = mc_state;
}
