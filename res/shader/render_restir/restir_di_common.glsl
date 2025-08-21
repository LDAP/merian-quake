#ifndef _MERIAN_QUAKE_RESTIR_DI_COMMOON_
#define _MERIAN_QUAKE_RESTIR_DI_COMMOON_

#include "merian-shaders/bsdf_ggx.glsl"
#include "merian-shaders/colors/colors_yuv.glsl"

float restir_di_target_pdf(const ReSTIRDISample y, const Hit surface) {
    const vec3 wo = normalize(y.pos - surface.pos);
    const float wodotn = dot(wo, surface.normal);

    if (wodotn <= 0) {
        return 0;
    }

    const float bsdf = bsdf_ggx_diffuse_mix_times_wodotn(surface.wi, wo, surface.normal, bsdf_ggx_roughness_to_alpha(surface.roughness), 0.02);

    return max(dot(y.normal, -wo), 0) / pow(distance(y.pos, surface.pos), 2) * bsdf * yuv_luminance_f16(y.radiance);
}

#endif // _MERIAN_QUAKE_RESTIR_DI_COMMOON_
