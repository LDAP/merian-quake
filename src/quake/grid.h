#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;
using vec4 = glm::vec4;
using float16_t = uint16_t;

#endif

#define MC_BUFFER_SIZE 4000000
#define MC_EXCHANGE_BUFFER_SIZE 40000

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;

    uint buf_idx;
    uint hash; // grid_idx and level
};

struct MCVertex {
    MCState state;
};

struct MCExchangeVertex {
    MCState state;
};

#define LIGHT_CACHE_BUFFER_SIZE 4000000

struct LightCacheVertex {
    uint hash; // grid_idx and level
    uint lock;
    vec4 irr_N;
};
