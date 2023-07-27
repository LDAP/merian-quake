
#define LIGHT_CACHE_MAX_N 1024
#define LIGHT_CACHE_MIN_ALPHA 0.00

vec4 light_cache_get(const vec3 pos, const vec3 normal, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, LIGHT_CACHE_BUFFER_SIZE);

    for (int i = 0; i < LIGHT_CACHE_ENTRIES_PER_VERTEX; i++) {
        LightCacheEntry entry = light_cache[buf_idx].entries[i];
        if (grid_idx == entry.grid_idx && dot(normal, entry.n) >= 0.7 && !any(isinf(entry.irr_N)) && !any(isnan(entry.irr_N))) {
            return entry.irr_N;
        }
    }

    return vec4(0);
}

void light_cache_update(const vec3 pos, const vec3 normal, const vec3 irr, inout uint rng_state) {
    const ivec3 grid_idx = grid_idx_interpolate(pos, LIGHT_CACHE_GRID_WIDTH, XorShift32(rng_state));
    const uint buf_idx = hash_grid(grid_idx, LIGHT_CACHE_BUFFER_SIZE);

    // TODO: Remove dirty fix
    if (any(greaterThanEqual(irr, vec3(1e7))))
        return;

    int i = 0;
    LightCacheEntry entry;
    bool valid = false;
    for (; i < LIGHT_CACHE_ENTRIES_PER_VERTEX; i++) {
        entry = light_cache[buf_idx].entries[i];
        if (grid_idx == entry.grid_idx && dot(normal, entry.n) >= 0.7 && !any(isinf(entry.irr_N)) && !any(isnan(entry.irr_N))) {
            valid = true;
            break;
        }
    }

    if (!valid) {
        entry.irr_N = vec4(0);
        entry.n = vec3(0);
        entry.grid_idx = grid_idx;
        i = int(round(XorShift32(rng_state) * (LIGHT_CACHE_ENTRIES_PER_VERTEX - 1)));
    }

    entry.n = normalize(mix(entry.n, normal, 0.001));
    entry.irr_N.a = min(entry.irr_N.a + 1, LIGHT_CACHE_MAX_N);
    entry.irr_N.rgb = mix(entry.irr_N.rgb, irr, max(1. / entry.irr_N.a, LIGHT_CACHE_MIN_ALPHA));

    light_cache[buf_idx].entries[i] = entry;
}
