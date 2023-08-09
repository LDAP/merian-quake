
#define LIGHT_CACHE_MAX_GRID_WIDTH 25
#define LIGHT_CACHE_MIN_GRID_WIDTH .1
#define LIGHT_CACHE_MAX_LEVEL 4

#define LIGHT_CACHE_MAX_N 128
#define LIGHT_CACHE_MIN_ALPHA .1

ivec3 grid_idx_for_level_interpolate(const uint level, const vec3 pos, const vec3 normal, inout uint rng_state) {
    float grid_width = cell_width_for_level_poly(level, LIGHT_CACHE_MAX_LEVEL, LIGHT_CACHE_MIN_GRID_WIDTH, LIGHT_CACHE_MAX_GRID_WIDTH, 8);
    return grid_idx_interpolate(pos, grid_width, XorShift32(rng_state));
}

ivec3 grid_idx_for_level_closest(const uint level, const vec3 pos, const vec3 normal, inout uint rng_state) {
    float grid_width = cell_width_for_level_poly(level, LIGHT_CACHE_MAX_LEVEL, LIGHT_CACHE_MIN_GRID_WIDTH, LIGHT_CACHE_MAX_GRID_WIDTH, 8);
    return grid_idx_closest(pos, grid_width);
}

vec4 light_cache_get(const vec3 pos, const vec3 normal, inout uint rng_state) {
    // const uint l = 5;
    // const ivec3 grid_idx = grid_idx_for_level(l, pos, normal, rng_state);
    // uint buf_idx = hash_grid_normal_level(grid_idx, normal, l, LIGHT_CACHE_BUFFER_SIZE);
    // return vec4(XorShift32(buf_idx), XorShift32(buf_idx), XorShift32(buf_idx), XorShift32(buf_idx));

    vec4 irr_N = vec4(0);
    float sum_w = 0;
    for (int level = 0; level <= LIGHT_CACHE_MAX_LEVEL; level++) {
        const ivec3 grid_idx = grid_idx_for_level_interpolate(level, pos, normal, rng_state);
        const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, LIGHT_CACHE_BUFFER_SIZE);
        LightCacheVertex vtx = light_cache[buf_idx];
        if (grid_idx == vtx.grid_idx        // detect conflict
            && level == vtx.level
            && !any(isinf(vtx.irr_N))       // make sure information is not damaged
            && !any(isnan(vtx.irr_N))
            //&& vtx.avg_frame >= params.frame - 1 // make sure information is somewhat recent
            //&& vtx.irr_N.a > 4              // make sure information is somewhat trustworthy
        ) {           
            float w = 1.;
            w *= smoothstep(4, LIGHT_CACHE_MAX_N, vtx.irr_N.a);
            w *= exp(vtx.avg_frame - params.frame);
            const float a = 100;
            w *= /*1. - smoothstep(0., LIGHT_CACHE_MAX_LEVEL + 1, level);*/(pow(a, 1. - smoothstep(0., LIGHT_CACHE_MAX_LEVEL + 1, level)) - 1) / (a - 1);

            irr_N += w * vtx.irr_N;
            sum_w += w;
        }
    }

    return sum_w > 0 ? irr_N / sum_w : vec4(0);
}

void light_cache_update(const vec3 pos, const vec3 normal, const vec3 irr, inout uint rng_state) {
    // const ivec3 grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
    // const uint buf_idx = hash_grid_normal(grid_idx, normal, LIGHT_CACHE_BUFFER_SIZE);

    // int i = 0;
    // LightCacheEntry entry;
    // bool valid = false;
    // for (; i < LIGHT_CACHE_ENTRIES_PER_VERTEX; i++) {
    //     entry = light_cache[buf_idx].entries[i];
    //     if (grid_idx == entry.grid_idx && !any(isinf(entry.irr_N)) && !any(isnan(entry.irr_N))) {
    //         valid = true;
    //         break;
    //     }
    // }

    // if (!valid) {
    //     entry.irr_N = vec4(0);
    //     entry.grid_idx = grid_idx;
    //     i = int(round(XorShift32(rng_state) * (LIGHT_CACHE_ENTRIES_PER_VERTEX - 1)));
    // }

    // entry.irr_N.a = min(entry.irr_N.a + 1, LIGHT_CACHE_MAX_N);
    // entry.irr_N.rgb = mix(entry.irr_N.rgb, irr, max(1. / entry.irr_N.a, LIGHT_CACHE_MIN_ALPHA));

    // light_cache[buf_idx].entries[i] = entry;

    for (int level = int(round(XorShift32(rng_state) * LIGHT_CACHE_MAX_LEVEL)); level >= 0; level--) {
        const ivec3 grid_idx = grid_idx_for_level_interpolate(level, pos, normal, rng_state);
        const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, LIGHT_CACHE_BUFFER_SIZE);

        const uint old = atomicExchange(light_cache[buf_idx].lock, params.frame);
        if (old == params.frame)
            // did not get lock
            continue;

        LightCacheVertex vtx = light_cache[buf_idx];

        if (any(isinf(vtx.irr_N)) || any(isnan(vtx.irr_N)) || old == 0) {
            vtx.grid_idx = grid_idx;
            vtx.level = level;
            vtx.irr_N = vec4(0);
            vtx.avg_frame = 0;
        } else if (vtx.grid_idx == grid_idx && vtx.level == level) {
            
        } else if (XorShift32(rng_state) < level / vtx.level * .1) { // < level / vtx.level * .1
            vtx.grid_idx = grid_idx;
            vtx.level = level;
            vtx.irr_N = vec4(0);
            vtx.avg_frame = 0;
        } else {
            continue;
        }

        vtx.irr_N.a = min(vtx.irr_N.a + 1, LIGHT_CACHE_MAX_N);
        vtx.irr_N.rgb = mix(vtx.irr_N.rgb, irr, max(1. / vtx.irr_N.a, LIGHT_CACHE_MIN_ALPHA));
        vtx.avg_frame = mix(vtx.avg_frame, params.frame, .5);

        light_cache[buf_idx] = vtx;

        return;   
    }
}
