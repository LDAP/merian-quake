#include "merian-shaders/types.glsl.h"


struct SSMCState {
    vec3  sum_tgt;
    float sum_w;
    uint  N;
    float sum_len;
    float f;
};
