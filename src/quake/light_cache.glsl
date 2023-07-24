
#define LIGHT_CACHE_MAX_N 1e6
#define LIGHT_CACHE_MIN_ALPHA 0

vec4 light_cache_get(const vec3 pos, const vec3 normal, inout uint rng_state) {
    ivec3 grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
    uint buf_idx = hash_grid(grid_idx, LIGHT_CACHE_BUFFER_SIZE);
    LightCacheEntry entry = light_cache[buf_idx].entry;

    if (grid_idx != entry.grid_idx || dot(normal, entry.n) <= 0.7) {
        grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
        buf_idx = hash_grid(grid_idx, LIGHT_CACHE_BUFFER_SIZE);
        entry = light_cache[buf_idx].entry;
    }

    const vec4 irr = entry.irr_N;
    if (any(isinf(irr)) || any(isnan(irr)) || grid_idx != entry.grid_idx || dot(normal, entry.n) <= 0.7) {
        return vec4(0);
    }

    return irr;
}

void light_cache_update(const vec3 pos, const vec3 normal, const vec3 irr, inout uint rng_state) {
    ivec3 grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
    uint buf_idx = hash_grid(grid_idx, LIGHT_CACHE_BUFFER_SIZE);
    LightCacheEntry entry = light_cache[buf_idx].entry;

    if (grid_idx != entry.grid_idx || dot(normal, entry.n) <= 0.7) {
        grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
        buf_idx = hash_grid(grid_idx, LIGHT_CACHE_BUFFER_SIZE);
        entry = light_cache[buf_idx].entry;
    }

    if (grid_idx != entry.grid_idx) {
        if (XorShift32(rng_state) < 0.001) {
            entry.grid_idx = grid_idx;
            entry.n = normal;
            entry.irr_N = vec4(0);
        }
        else {
            return;
        }
    }

    vec4 irr_N = entry.irr_N;
    vec3 n = entry.n;
    if (any(isinf(n)) || any(isnan(n)) || any(isinf(irr_N)) || any(isnan(irr_N))) {
        irr_N = vec4(0);
        n = normal;
    }

    if (dot(normal, n) > .9) {
        irr_N.a = min(irr_N.a + 1, LIGHT_CACHE_MAX_N);
        irr_N.rgb = mix(irr_N.rgb, irr, max(1. / irr_N.a, LIGHT_CACHE_MIN_ALPHA));
        entry.irr_N = irr_N;
    }
    entry.n = normalize(mix(n, normal, 0.001));

    light_cache[buf_idx].entry = entry;
}
