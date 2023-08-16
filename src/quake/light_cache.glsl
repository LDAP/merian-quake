#define LIGHT_CACHE_MAX_GRID_WIDTH 100.
#define LIGHT_CACHE_MIN_GRID_WIDTH .01
#define LIGHT_CACHE_LEVELS 32.
// Set the target for light cache resolution
#define LIGHT_CACHE_TAN_ALPHA_HALF 0.002

#define LIGHT_CACHE_MAX_N 128
#define LIGHT_CACHE_MIN_ALPHA .01

uint lc_level_for_pos(const vec3 pos, inout uint rng_state) {
    const float target_grid_width = clamp(2 * LIGHT_CACHE_TAN_ALPHA_HALF * distance(pos, params.cam_x.xyz), LIGHT_CACHE_MIN_GRID_WIDTH, LIGHT_CACHE_MAX_GRID_WIDTH);
    const float level = LIGHT_CACHE_LEVELS * pow((target_grid_width - LIGHT_CACHE_MIN_GRID_WIDTH) / (LIGHT_CACHE_MAX_GRID_WIDTH - LIGHT_CACHE_MIN_GRID_WIDTH), 1 / 9.);
    return uint(round(level));

    // const float frac = fract(level);
    // if (frac < .4 || frac > .6) {
    //     return uint(round(level));
    // }

    // const uint lower = uint(level);
    // return XorShift32(rng_state) < frac ? lower + 1 : lower;
}

ivec3 lc_grid_idx_for_level_closest(const uint level, const vec3 pos, inout uint rng_state) {
    const float grid_width = pow(level / LIGHT_CACHE_LEVELS, 9.) * (LIGHT_CACHE_MAX_GRID_WIDTH - LIGHT_CACHE_MIN_GRID_WIDTH) + LIGHT_CACHE_MIN_GRID_WIDTH;
    return grid_idx_closest(pos, grid_width);
}

ivec3 lc_grid_idx_for_level_interpolate(const uint level, const vec3 pos, inout uint rng_state) {
    const float grid_width = pow(level / LIGHT_CACHE_LEVELS, 9.) * (LIGHT_CACHE_MAX_GRID_WIDTH - LIGHT_CACHE_MIN_GRID_WIDTH) + LIGHT_CACHE_MIN_GRID_WIDTH;
    return grid_idx_interpolate(pos, grid_width, XorShift32(rng_state));
}

vec4 light_cache_get_level(const uint level, const vec3 pos, const vec3 normal, inout uint rng_state) {
    const ivec3 grid_idx = lc_grid_idx_for_level_interpolate(level, pos, rng_state);
    const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, LIGHT_CACHE_BUFFER_SIZE);
    const LightCacheVertex vtx = light_cache[buf_idx];

    if (vtx.hash == hash_level(grid_idx, level)
        && !any(isinf(vtx.irr_N))
        && !any(isnan(vtx.irr_N))) {
        return vtx.irr_N;
    }

    return vec4(0);
}

vec4 light_cache_get(const vec3 pos, const vec3 normal, inout uint rng_state) {
    const uint level = lc_level_for_pos(pos, rng_state);
    return light_cache_get_level(level, pos, normal, rng_state);
}

void light_cache_update(const vec3 pos, const vec3 normal, const vec3 irr, inout uint rng_state) {
    const uint level = lc_level_for_pos(pos, rng_state);
    const ivec3 grid_idx = lc_grid_idx_for_level_closest(level, pos, rng_state);
    const uint buf_idx = hash_grid_normal_level(grid_idx, normal, level, LIGHT_CACHE_BUFFER_SIZE);
    
    const uint old = atomicExchange(light_cache[buf_idx].lock, params.frame);
    if (old == params.frame)
        // did not get lock
        return;

    LightCacheVertex vtx = light_cache[buf_idx];

    if (vtx.hash != hash_level(grid_idx, level)
        || any(isinf(vtx.irr_N))
        || any(isnan(vtx.irr_N))) {

        // attempt to get from coarser level
        vtx.irr_N = light_cache_get_level(level + 1, pos, normal, rng_state);
        vtx.hash = hash_level(grid_idx, level);
    }

    vtx.irr_N.a = min(vtx.irr_N.a + 1, LIGHT_CACHE_MAX_N);
    vtx.irr_N.rgb = mix(vtx.irr_N.rgb, irr, max(1. / vtx.irr_N.a, LIGHT_CACHE_MIN_ALPHA));

    light_cache[buf_idx] = vtx;

    light_cache[buf_idx].lock = 0;
}
