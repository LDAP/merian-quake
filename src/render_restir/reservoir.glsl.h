#ifndef _RESERVOIR_H_
#define _RESERVOIR_H_

#include "merian-shaders/types.glsl.h"

struct Reservoir {
    vec3 pos;
    float f;
    float w;
};

#ifdef GL_core_profile // is glsl

#include "merian-shaders/random.glsl"

Reservoir reservoir_init() {
    Reservoir reservoir = {vec3(0), 0., 0.};
    return reservoir;
}

void reservoir_add_sample(inout Reservoir reservoir, inout uint rng_state, const vec3 pos, const float f) {
    reservoir.w += f;
    if (XorShift32(rng_state) < f / reservoir.w) {
        reservoir.f = f;
        reservoir.pos = pos;
    }
}

#endif

#endif // _RESERVOIR_H_
