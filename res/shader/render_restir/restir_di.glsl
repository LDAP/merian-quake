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
    ReSTIRDIReservoir reservoir = {0, 0., 0., ReSTIRDISample(vec3(0), f16vec3(0.hf), 0)};
    return reservoir;
}

ReSTIRDIReservoir restir_di_reservoir_init(const ReSTIRDISample x,
                                           const float p_sample,
                                           const float p_target) {
    ReSTIRDIReservoir reservoir = {1, p_target / p_sample, p_target, x};
    return reservoir;
}

void restir_di_reservoir_discard(inout ReSTIRDIReservoir reservoir) {
    reservoir.w_sum_or_W = 0;
    reservoir.y.flags = 0;
    reservoir.y.radiance = f16vec3(0);
}

bool restir_di_light_sample_valid(const ReSTIRDISample y) {
    return (y.flags & ReSTIRDISample_Flags_Valid) > 0;
}

// Add a sample to the reservoir.
// Note: p_sample can also be the effective PDF after MIS. (ReSTIR DI, page 3)
//
// (this is called update() in the ReSTIR paper / StreamSample in RTXDI)
bool restir_di_reservoir_add_sample(inout ReSTIRDIReservoir reservoir,
                                    inout uint rng_state,
                                    const ReSTIRDISample x,
                                    const float p_sample,
                                    const float p_target) {
    const float w = p_target / p_sample;
    reservoir.w_sum_or_W += w;
    reservoir.M += 1;

    // Weighted Reservoir Sampling (Chao, 1982):
    // selects a sample x_i with probability w_i / sum(w_i),
    // while only storing the current sample.
    if (XorShift32(rng_state) * reservoir.w_sum_or_W < w) {
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

    if (other.p_target == 0) {
        return false;
    }

    // Account for the fact that samples from neighbor are resampled from a different target
    // distribution by reweighting the samples with p_target_x(other.y) / p_target_y(other.y)
    const float w = p_target_x_y * other.w_sum_or_W / other.p_target;
    reservoir.w_sum_or_W += w;
    if (XorShift32(rng_state) * reservoir.w_sum_or_W < w) {
        reservoir.p_target = p_target_x_y;
        reservoir.y = other.y;
        return true;
    }
    return false;
}

// like restir_di_reservoir_combine but for other Reservoirs that were finalized (the RIS weight sum is replaced with W).
bool restir_di_reservoir_combine_finalized(inout ReSTIRDIReservoir reservoir,
                                           inout uint rng_state,
                                           const ReSTIRDIReservoirFinalized other,
                                           const float p_target_x_y) {
    reservoir.M += other.M;

    // Account for the fact that samples from neighbor are resampled from a different target
    // distribution by reweighting the samples.
    const float w = p_target_x_y * other.w_sum_or_W * other.M;
    reservoir.w_sum_or_W += w;
    if (XorShift32(rng_state) * reservoir.w_sum_or_W < w) {
        reservoir.p_target = p_target_x_y;
        reservoir.y = other.y;
        return true;
    }
    return false;
}

// performs the normalization after resampling. After this method w_sum_or_W holds W.
// equation 6 in the ReSTIR DI paper
// Uses 1 / M
void restir_di_reservoir_finalize(inout ReSTIRDIReservoir reservoir) {
    const float denominator = reservoir.M * reservoir.p_target;
    reservoir.w_sum_or_W = denominator > 0. ? reservoir.w_sum_or_W / denominator : 0.0;
}

// performs the normalization after resampling. After this method w_sum_or_W holds W.
// equation 6 in the ReSTIR DI paper
void restir_di_reservoir_finalize_custom(inout ReSTIRDIReservoir reservoir, const float numerator, float denominator) {
    denominator *= reservoir.p_target;
    reservoir.w_sum_or_W = denominator > 0. ? reservoir.w_sum_or_W * numerator / denominator : 0.0;
}

#endif // _MERIAN_RESTIR_DI_H_
