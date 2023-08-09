#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
using vec4 = glm::vec4;

#endif

#define STATES_PER_CELL 50
#define MC_BUFFER_SIZE 200000
#define MC_GRID_WIDTH 15.3

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;
    float f;
    ivec3 grid_idx;
    //vec3 normal;
};

struct MCVertex {
    MCState states[STATES_PER_CELL];
};

#define LIGHT_CACHE_BUFFER_SIZE 40000000

struct LightCacheVertex {
    ivec3 grid_idx;
    float avg_frame;
    uint level;
    uint lock;
    vec4 irr_N;
    //vec3 n;
};
