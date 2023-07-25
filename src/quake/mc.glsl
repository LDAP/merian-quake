// Configure ML
#define ML_MAX_N 1024
#define ML_MIN_ALPHA 0.01

MCState mc_state_new() {
    MCState r = {vec3(0.0), 0.0, 0, 0.0, 0.0, ivec3(0.0)};
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

MCState mc_state_load(const vec3 pos, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, MC_GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, MC_BUFFER_SIZE);
    const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
    const MCState state = cells[buf_idx].states[uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)))];
    if (grid_idx == state.grid_idx) {
        return state;
    } else {
        // const ivec3 grid_idx = grid_idx_interpolate(pos, MC_GRID_WIDTH, XorShift32(rng_state));
        // const uint buf_idx = hash_grid(grid_idx, MC_BUFFER_SIZE);
        // const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
        // const MCState state = cells[buf_idx].states[uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)))];
        // if (grid_idx == state.grid_idx) {
        //     return state;
        // } else {
            return mc_state_new();
        // }
    }
}

// return true if a valid state was found
bool mc_state_load_resample(out MCState mc_state, const vec3 pos, inout uint rng_state) {
    float score_sum = 0;
    for (int i = 0; i < 5; i++) {
        const MCState candidate = mc_state_load(pos, rng_state);
        const float candidate_score = candidate.sum_w;  // * dot(vmf.xyz, normal)
        // why not best?
        score_sum += candidate_score;
        if (XorShift32(rng_state) < candidate_score / score_sum) {
            // we use here that comparison with NaN is false, that happens if candidate_score == 0 and sum == 0; 
            mc_state = candidate;
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

void mc_state_save(MCState mc_state, const vec3 pos, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, MC_GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, MC_BUFFER_SIZE);
    const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
    mc_state.grid_idx = grid_idx;
    cells[buf_idx].states[state_idx] = mc_state;
}

bool mc_state_valid(const MCState mc_state) {
    return mc_state.sum_w > 0.0;
}
