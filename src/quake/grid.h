#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
using vec4 = glm::vec4;

#endif

#define MC_BUFFER_SIZE 4000000

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;

    ivec3 grid_idx;
    uint level;
};

struct MCVertex {
    MCState state;
};

#define LIGHT_CACHE_BUFFER_SIZE 4000000

struct LightCacheVertex {
    uint hash; // grid_idx and level
    uint lock;
    vec4 irr_N;
};
