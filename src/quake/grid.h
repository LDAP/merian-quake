#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
using vec4 = glm::vec4;

#endif

#define STATES_PER_CELL 20
#define MC_BUFFER_SIZE 200000
#define MC_GRID_WIDTH 15.3

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;
    float f;
    ivec3 grid_idx;
};

struct MCVertex {
    MCState states[STATES_PER_CELL];
};

#define LIGHT_CACHE_BUFFER_SIZE 100000
#define LIGHT_CACHE_GRID_WIDTH 20.3
#define LIGHT_CACHE_ENTRIES_PER_VERTEX 6

struct LightCacheEntry {
    ivec3 grid_idx;
    vec4 irr_N;
    vec3 n;
};

struct LightCacheVertex {
    LightCacheEntry entries[LIGHT_CACHE_ENTRIES_PER_VERTEX];
};
