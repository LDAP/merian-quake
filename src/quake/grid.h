#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
using vec4 = glm::vec4;
using float16_t = uint16_t;

#endif

#define MC_ADAPTIVE_BUFFER_SIZE 4000000
#define MC_STATIC_BUFFER_SIZE 200000
#define MC_STATIC_VERTEX_STATE_COUNT 25

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;

    uint hash; // grid_idx and level
};

struct MCAdaptiveVertex {
    MCState state;
};

struct MCStaticVertex {
    MCState states[MC_STATIC_VERTEX_STATE_COUNT];
};

#define LIGHT_CACHE_BUFFER_SIZE 4000000

struct LightCacheVertex {
    uint hash; // grid_idx and level
    uint lock;
    vec4 irr_N;
};
