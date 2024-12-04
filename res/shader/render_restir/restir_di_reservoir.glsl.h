#ifndef _MERIAN_RESTIR_DI_RESERVOIR_H_
#define _MERIAN_RESTIR_DI_RESERVOIR_H_

#include "merian-shaders/types.glsl.h"

#define ReSTIRDISample_Flags_Valid 1

struct ReSTIRDISample {
    vec3 pos;
    vec3 normal;
    f16vec3 radiance;
    uint flags;
};

struct ReSTIRDIReservoir {
    uint M;      // number of samples that went into the reservoir
    float w_sum_or_W; // sum of RIS weights (w_i = target_p / p_sample)
                      // or the reservoir_weight W (eq. 6 in paper) after resampling.

    float p_target;          // the target function for the current sample (not really a PDF)
    ReSTIRDISample y;        // current sample

    // => the current one sample estimator is <L> = (f(y) / p_target) * (w_sum_or_W / M)
    //    the second factor corrects that the distribution of y only approximates p_target
};

// singalize that w_sum_or_W is now W and not the RIS weight
#define ReSTIRDIReservoirFinalized ReSTIRDIReservoir

#endif // _MERIAN_RESTIR_DI_RESERVOIR_H_
