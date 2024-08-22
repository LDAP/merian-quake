#ifndef _MERIAN_RESTIR_DI_H_
#define _MERIAN_RESTIR_DI_H_

#include "restir_di_reservoir.glsl.h"

#include "merian-shaders/random.glsl"

// +------------------------+
// |  ReSTIR DI Pseudocode  |
// +------------------------+
//
// For each pixel p:
//
// Reservoir r
// for n samples x ~ p:
//  restir_di_reservoir_add_sample(r, rng_state, x, p(x), f_p(x))
//
// if f_p was unshadowed contribution:
//  if (shadowed(r.y))
//    discard r
//
// // Temporal reuse
// Reservoir o = get_reprojected_reservoir_from_previous_frame()
// restir_di_reservoir_combine(r, rng_state, o, f_p(o.y))
//
// // write to share with neighbors (or use shared memory, subgroup operations,...)
// share(r);
//
// // Spatial reuse
// for n neighbors:
//   Reservoir o = get_reservoir_from_neighbor()
//   restir_di_reservoir_combine(r, rng_state, o, f_p(o.y))
//   (barrier() // get new information next iteraiton)
//
// // write to temporal buffer for next iteration
// store(r)
//
// // shade pixel...
//

ReSTIRDIReservoir restir_di_reservoir_init() {
    ReSTIRDIReservoir reservoir = {0, 0., 0., vec3(0)};
    return reservoir;
}

ReSTIRDIReservoir restir_di_reservoir_init(const vec3 x,
                                           const float p_sample,
                                           const float p_target) {
    ReSTIRDIReservoir reservoir = {1, p_target / p_sample, p_target, x};
    return reservoir;
}

// The weight / inverse PDF for the current sample.
// returns W = 1 / p_ris = (1 / p_target) * (w_sum / M)
float restir_di_reservoir_W(const ReSTIRDIReservoir reservoir) {
    return reservoir.w_sum / reservoir.p_target / reservoir.M;
}

// Add a sample to the reservoir.
// Note: p_sample can also be the effective PDF after MIS. (ReSTIR DI, page 3)
//
// (this is called update() in the ReSTIR paper / StreamSample in RTXDI)
bool restir_di_reservoir_add_sample(inout ReSTIRDIReservoir reservoir,
                                    inout uint rng_state,
                                    const vec3 x,
                                    const float p_sample,
                                    const float p_target) {
    const float w = p_target / p_sample;
    reservoir.w_sum += w;
    reservoir.M += 1;

    // Weighted Reservoir Sampling (Chao, 1982):
    // selects a sample x_i with probability w_i / sum(w_i),
    // while only storing the current sample.
    if (XorShift32(rng_state) * reservoir.w_sum < w) {
        reservoir.p_target = p_target;
        reservoir.y = x;
        return true;
    }
    return false;
}

// Combine two reservoirs.
//
// p_target_x_y = p_target_x(other.y) i.e. the target function of the current pixel evaluated with
// the sample of the other reservoir (needed for reweighting).
//
// Note: Naively combining reservoirs of neighbors is biased since each pixel uses a different
// integration domain and target distribution! (ReSTRI DI, page 4: ReSTIR DI. chapter 4).
// Quote: If all candidate PDFs are non-zero wherever the target function is non-zero, then |Z (y)| = M,
// and the RIS weight becomes an unbiased estimator of the inverse RIS PDF. If, however,
// some of the PDFs are zero for part of the integrand, then |Z (y)| M < 1,
// and the inverse PDF is consistently underestimated. 
//
// Note: visibility resuse means:
// For each pixel check if y is occluded and if so, discard the reservoir before sharing with
// neighbors.
bool restir_di_reservoir_combine(inout ReSTIRDIReservoir reservoir,
                                 inout uint rng_state,
                                 const ReSTIRDIReservoir other,
                                 const float p_target_x_y) {
    reservoir.M += other.M;

    if (other.w_sum == 0.)
        return false;

    // Account for the fact that samples from neighbor are resampled from a different target
    // distribution by reweighting the samples with p_target_x(other.y) / p_target_y(other.y)
    const float w = other.w_sum * p_target_x_y / other.p_target;
    reservoir.w_sum += w;
    if (XorShift32(rng_state) * reservoir.w_sum < w) {
        reservoir.p_target = other.p_target;
        reservoir.y = other.y;
        return true;
    }
    return false;
}

#endif // _MERIAN_RESTIR_DI_H_
