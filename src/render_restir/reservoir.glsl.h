#ifndef _RESERVOIR_H_
#define _RESERVOIR_H_

#include "merian-shaders/types.glsl.h"

struct ReSTIRDIReservoir {
    uint M;      // number of samples that went into the reservoir
    float w_sum; // sum of weights (w_i = target_p / p_sample).
    
    float p_target;     // the target function for the current sample (not really a PDF)
    vec3 y;             // current sample

    // => the current one sample estimator is <L> = (f(y) / p_target) * (w_sum / M)
    //    the second factor corrects that the distribution of y only approximates p_target

    // float f;
};

#ifdef GL_core_profile // is glsl

#include "merian-shaders/random.glsl"

ReSTIRDIReservoir reservoir_init() {
    ReSTIRDIReservoir reservoir = {0, 0., 0., vec3(0)};
    return reservoir;
}

// Add a sample to the reservoir.
// Note: p_sample can also be the effective PDF after MIS. (ReSTIR DI, page 3)
// 
// (this is called update() in the ReSTIR paper)
void reservoir_add_sample(inout ReSTIRDIReservoir reservoir, inout uint rng_state, const vec3 x, const float p_sample, const float p_target) {
    const float w = p_target / p_sample;
    reservoir.w_sum += w;
    reservoir.M += 1;

    // Weighted Reservoir Sampling (Chao, 1982):
    // selects a sample x_i with probability w_i / sum(w_i),
    // while only storing the current sample.
    if (XorShift32(rng_state) < w / reservoir.w_sum) {
        reservoir.p_target = p_target;
        reservoir.y = x;
    }
}

// The weight / inverse PDF for the current sample.
// returns W = 1 / p_ris = (1 / p_target) * (w_sum / M)
float reservoir_W(const ReSTIRDIReservoir reservoir) {
    return reservoir.w_sum / reservoir.p_target / reservoir.M;
}

#endif

#endif // _RESERVOIR_H_
