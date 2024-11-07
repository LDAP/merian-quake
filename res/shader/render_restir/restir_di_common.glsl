#ifndef _MERIAN_QUAKE_RESTIR_DI_COMMOON_
#define _MERIAN_QUAKE_RESTIR_DI_COMMOON_

#include "merian-shaders/bsdf_microfacet.glsl"
#include "merian-shaders/frames.glsl"
#include "merian-shaders/color/colors_yuv.glsl"

float restir_di_target_pdf(const ReSTIRDISample y, const Hit surface) {
    const vec3 wo = normalize(y.pos - surface.pos);
    const float wodotn = dot(wo, surface.normal);

    if (wodotn <= 0) {
        return 0;
    }

    const float bsdf = bsdf_microfacet_eval(-surface.wi, make_frame(surface.normal), wo, vec2(surface.roughness)) * wodotn;

    return yuv_luminance(bsdf * y.radiance);
}

#endif // _MERIAN_QUAKE_RESTIR_DI_COMMOON_
