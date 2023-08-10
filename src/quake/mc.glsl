#define MC_MAX_GRID_WIDTH 25
#define MC_MIN_GRID_WIDTH .1
#define MC_MAX_LEVEL 4

// Configure ML
#define ML_MAX_N 1024
#define ML_MIN_ALPHA .1

ivec3 mc_grid_idx_for_level_interpolate(const uint level, const vec3 pos, const vec3 normal, inout uint rng_state) {
    float grid_width = cell_width_for_level_poly(level, MC_MAX_LEVEL, MC_MIN_GRID_WIDTH, MC_MAX_GRID_WIDTH, 2);
    return grid_idx_interpolate(pos, grid_width, XorShift32(rng_state));
}

ivec3 mc_grid_idx_for_level_closest(const uint level, const vec3 pos, const vec3 normal, inout uint rng_state) {
    float grid_width = cell_width_for_level_poly(level, MC_MAX_LEVEL, MC_MIN_GRID_WIDTH, MC_MAX_GRID_WIDTH, 8);
    return grid_idx_closest(pos, grid_width);
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

// MCState mc_state_load(const vec3 pos, const vec3 normal, inout uint rng_state) {
//     const ivec3 grid_idx = grid_idx_interpolate(pos, MC_GRID_WIDTH, XorShift32(rng_state));
//     const uint buf_idx = hash_grid_normal(grid_idx, normal, MC_BUFFER_SIZE);
//     const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
//     const MCState state = mc_states[buf_idx].states[uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)))];
//     if (grid_idx == state.grid_idx) {
//         return state;
//     } else {
//         // const ivec3 grid_idx = grid_idx_interpolate(pos, MC_GRID_WIDTH, XorShift32(rng_state));
//         // const uint buf_idx = hash_grid(grid_idx, MC_BUFFER_SIZE);
//         // const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
//         // const MCState state = mc_states[buf_idx].states[uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)))];
//         // if (grid_idx == state.grid_idx) {
//         //     return state;
//         // } else {
//             return mc_state_new();
//         // }
//     }
// }

// return true if a valid state was found
bool mc_state_load_resample(out MCState mc_state, const vec3 pos, const vec3 normal, inout uint rng_state) {
    float score_sum = 0;
    for (int level = 0; level <= MC_MAX_LEVEL; level++) {
        //int level = clamp(int(round(XorShift32(rng_state) * MC_MAX_LEVEL)), 0, MC_MAX_LEVEL);
        const ivec3 grid_idx = mc_grid_idx_for_level_interpolate(level, pos, normal, rng_state);
        const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, MC_BUFFER_SIZE);
        MCVertex vtx = mc_states[buf_idx];

        const float candidate_score = vtx.state.sum_w * float(grid_idx == vtx.state.grid_idx && level == vtx.state.level);  // * smoothstep(0.8, 1.0, dot(vtx.normal, normal))

        score_sum += candidate_score;
        if (XorShift32(rng_state) < candidate_score / score_sum) {
            // we use here that comparison with NaN is false, that happens if candidate_score == 0 and sum == 0; 
            mc_state = vtx.state;
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

void mc_state_save(MCState mc_state, const vec3 pos, const vec3 normal, inout uint rng_state) {
    // const ivec3 grid_idx = grid_idx_interpolate(pos, MC_GRID_WIDTH, XorShift32(rng_state));
    // const uint buf_idx = hash_grid_normal(grid_idx, normal, MC_BUFFER_SIZE);
    // const uint state_idx = uint(round(XorShift32(rng_state) * (STATES_PER_CELL - 1)));
    // mc_state.grid_idx = grid_idx;
    // //mc_state.normal = normal;
    // mc_states[buf_idx].states[state_idx] = mc_state;
    
    // update state that was used for sampling
    {
        const uint buf_idx = hash_grid_normal_level(mc_state.grid_idx, normal, mc_state.level, MC_BUFFER_SIZE);
        // const uint old = atomicExchange(mc_states[buf_idx].lock, params.frame);
        // if (old != params.frame)
            mc_states[buf_idx].state = mc_state;
            mc_states[buf_idx].avg_frame = mix(mc_states[buf_idx].avg_frame, params.frame, .5);
    }

    // resample to coarser level
    float sum = 0;
    for (uint i = 0; i < 5; i++) {
        mc_state.level = int(round(XorShift32(rng_state) * MC_MAX_LEVEL));
        mc_state.grid_idx = mc_grid_idx_for_level_interpolate(mc_state.level, pos, normal, rng_state);
        const uint buf_idx = hash_grid_normal_level(mc_state.grid_idx, normal, mc_state.level, MC_BUFFER_SIZE);
        sum += mc_states[buf_idx].state.sum_w;
        //if (XorShift32(rng_state) < mc_state.sum_w / mc_states[buf_idx].state.sum_w) {
            // const uint old = atomicExchange(mc_states[buf_idx].lock, params.frame);
            // if (old == params.frame)
            //     // did not get lock
            //     continue;

            mc_states[buf_idx].state = mc_state;
            mc_states[buf_idx].avg_frame = mix(mc_states[buf_idx].avg_frame, params.frame, .5);
        //}
    }

    // for (int level = MC_MAX_LEVEL; level >= 0; level--) {
    //     const ivec3 grid_idx = mc_grid_idx_for_level_interpolate(level, pos, normal, rng_state);
    //     const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, MC_BUFFER_SIZE);

    //     const uint old = atomicExchange(mc_states[buf_idx].lock, params.frame);
    //     if (old == params.frame)
    //         // did not get lock
    //         continue;

    //     MCVertex vtx = mc_states[buf_idx];

    //     if (old == 0) {
    //         vtx.grid_idx = grid_idx;
    //         vtx.level = level;
    //         vtx.avg_frame = 0;
    //     } else if (vtx.grid_idx == grid_idx && vtx.level == level) {
            
    //     } else if (XorShift32(rng_state) < level / vtx.level * .1) { // < level / vtx.level * .1
    //         vtx.grid_idx = grid_idx;
    //         vtx.level = level;
    //         vtx.avg_frame = 0;
    //     } else {
    //         continue;
    //     }

    //     vtx.state = mc_state;
    //     vtx.avg_frame = mix(vtx.avg_frame, params.frame, .5);

    //     mc_states[buf_idx] = vtx;
    //     return;   
    // }
}

bool mc_state_valid(const MCState mc_state) {
    return mc_state.sum_w > 0.0;
}
