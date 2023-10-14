#include "merian-nodes/common/types.glsl.h"

#define MC_STATIC_VERTEX_STATE_COUNT 23

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;

    vec3 mv;
    uint frame;

    uint hash; // grid_idx and level
};

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

#define DISTANCE_MC_VERTEX_STATE_COUNT 10

struct DistanceMCVertex {
    DistanceMCState states[DISTANCE_MC_VERTEX_STATE_COUNT];
};
