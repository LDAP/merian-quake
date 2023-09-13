#include "merian-nodes/common/types.glsl.h"

#define MC_ADAPTIVE_BUFFER_SIZE 32777259
#define MC_STATIC_BUFFER_SIZE 800009
#define MC_STATIC_VERTEX_STATE_COUNT 23

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;

    uint hash; // grid_idx and level
};

#define LIGHT_CACHE_BUFFER_SIZE 4000000

struct LightCacheVertex {
    uint hash; // grid_idx and level
    uint lock;
    f16vec3 irr;
    uint16_t N;
};

struct DistanceMCState {
    float sum_w;
    uint N;
    vec2 moments;
};
