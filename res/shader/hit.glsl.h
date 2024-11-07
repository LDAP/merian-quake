#ifndef _HIT_H_
#define _HIT_H_

#include "merian-shaders/types.glsl.h"

struct Hit {
    vec3 pos;
    vec3 prev_pos;
    vec3 wi;
    vec3 normal;
    uint enc_geonormal;

    // Material
    f16vec3 albedo;
    float16_t roughness;
};

#endif // _HIT_H_
