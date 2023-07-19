// Configure ML
#define ML_PRIOR_N .20 // cannot be zero or else mean cos -> kappa blows up
#define ML_MAX_N 1024
#define ML_MIN_ALPHA 0.01

MCState mc_state_new() {
    MCState r = {vec3(0.0), 0.0, 0, 0.0, 0.0};
    return r;
}

// return true if a valid state was found
// in case of false the state is reset to zero
bool mc_state_load(out MCState mc_state, const vec3 pos, inout uint rng_state, const ivec2 pixel) {
    // todo: Jo used a sum here instead of the "best", better/why?
    float best_score = 0;
    for (int i = 0; i < 5; i++) {
        const ivec3 grid_idx = grid_idx_interpolate(pos, GRID_WIDTH, XorShift32(rng_state));
        const uint buf_idx = hash_grid(grid_idx, BUFFER_SIZE);
        const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
        // if (cells[buf_idx].grid_idx != grid_idx) {
        //     // hash grid collision
        //     continue;
        // }
        const MCState candidate = cells[buf_idx].states[uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)))];
        const float candidate_score = candidate.f;
        if (XorShift32(rng_state) < candidate_score / (candidate_score + best_score)) {
            // we use here that comparison with NaN is false, that happens if candidate_score == 0 and sum == 0; 
            mc_state = candidate;
            best_score = candidate_score;
        }
    }

    return best_score > 0.0;
}

// return normalized direction (from pos)
vec3 mc_state_dir(const MCState mc_state, const vec3 pos) {
    const vec3 tgt = mc_state.sum_tgt / (mc_state.sum_w > 0.0 ? mc_state.sum_w : 1.0);
    return normalize(tgt - pos);
}

// returns the vmf lobe vec4(direction, kappa) for a position
vec4 mc_state_get_vmf(const MCState mc_state, const vec3 pos) {
    float r = mc_state.sum_len / mc_state.sum_w; // = mean cosine in [0,1]

    const vec3 tgt = mc_state.sum_tgt / (mc_state.sum_w > 0.0 ? mc_state.sum_w : 1.0);
    const float d = length(tgt - pos);
    const float rp = 1.0 -  1.0 / clamp(50.0 * d, 0.0, 6500.0);

    r = (mc_state.N * mc_state.N * r + ML_PRIOR_N * rp) / (mc_state.N * mc_state.N + ML_PRIOR_N);
    return vec4(mc_state_dir(mc_state, pos), (3.0 * r - r * r * r) / (1.0 - r * r));
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void mc_state_add_sample(inout MCState mc_state,
                         const vec3 pos,         // position where the ray started
                         const float w,          // goodness
                         const vec3 dir,         // raydir
                         const vec3 light_pos) { // ray hit point
    mc_state.N = min(mc_state.N + 1, ML_MAX_N);
    const float alpha = max(1.0 / mc_state.N, ML_MIN_ALPHA);
    mc_state.sum_w   = mix(mc_state.sum_w,   w,             alpha);
    mc_state.sum_tgt = mix(mc_state.sum_tgt, w * light_pos, alpha);

    vec3 to = mc_state.sum_len * mc_state_dir(mc_state, pos);
    to = mix(to, w * dir, alpha);
    mc_state.sum_len = length(to);
}

void mc_state_save(const MCState mc_state, const vec3 pos, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, BUFFER_SIZE);
    const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
    cells[buf_idx].states[state_idx] = mc_state;
    cells[buf_idx].grid_idx = grid_idx;
}

bool mc_state_valid(const MCState mc_state) {
    return mc_state.sum_w > 0.0;
}
