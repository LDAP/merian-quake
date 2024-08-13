#ifndef _MERIAN_RESTIR_DI_RESERVOIR_H_
#define _MERIAN_RESTIR_DI_RESERVOIR_H_

#include "merian-shaders/types.glsl.h"

struct ReSTIRDIReservoir {
    uint M;      // number of samples that went into the reservoir
    float w_sum; // sum of weights (w_i = target_p / p_sample).

    float p_target; // the target function for the current sample (not really a PDF)
    vec3 y;         // current sample

    // => the current one sample estimator is <L> = (f(y) / p_target) * (w_sum / M)
    //    the second factor corrects that the distribution of y only approximates p_target

    // float f;
};

#endif // _MERIAN_RESTIR_DI_RESERVOIR_H_
