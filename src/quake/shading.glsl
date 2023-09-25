#include "common/von_mises_fisher.glsl"
#include "common/cubemap.glsl"

// assert(alpha != 0)
#define decode_alpha(enc_alpha) (float16_t(alpha - 1) / 14.hf)

f16vec3 get_sky(const vec3 w) {
    f16vec3 emm = f16vec3(0);

    {
        const vec3 sundir = normalize(vec3(SUN_W_X, SUN_W_Y, SUN_W_Z));
        const f16vec3 suncolor = f16vec3(SUN_COLOR_R, SUN_COLOR_G, SUN_COLOR_B);

        emm += 0.5hf * pow(0.5hf * (1.0hf + float16_t(dot(sundir, w))), 4.0hf);
        emm += 5.0hf * float16_t(vmf_pdf(3000.0, dot(sundir, w)));
        emm *= suncolor;
    }


    if((params.sky_lf_ft & 0xffff) == 0xffff) {
        // classic quake sky
        const vec2 st = 0.5 + 1. * vec2(w.x , w.y) / abs(w.z);
        const vec2 t = params.cl_time * vec2(0.12, 0.12);
        const vec4 bck = texture(img_tex[nonuniformEXT(min(params.sky_rt_bk & 0xffff, MAX_GLTEXTURES - 1))], st + 0.5 * t);
        const vec4 fnt = texture(img_tex[nonuniformEXT(min(params.sky_rt_bk >> 16   , MAX_GLTEXTURES - 1))], st + t);
        const vec3 tex = mix(bck.rgb, fnt.rgb, fnt.a);
        emm = 10.0hf * (exp2(3.5hf * f16vec3(tex)) - 1.0hf);
    } else {
        // Evaluate cubemap
        uint side = 0;
        vec2 st;
        switch(cubemap_side(w)) {
            case 0: { side = params.sky_rt_bk & 0xffff; st = 0.5 + 0.5 * vec2(-w.y, -w.z) / abs(w.x); break; } // rt
            case 1: { side = params.sky_lf_ft & 0xffff; st = 0.5 + 0.5 * vec2( w.y, -w.z) / abs(w.x); break; } // lf
            case 2: { side = params.sky_rt_bk >> 16   ; st = 0.5 + 0.5 * vec2( w.x, -w.z) / abs(w.y); break; } // bk
            case 3: { side = params.sky_lf_ft >> 16   ; st = 0.5 + 0.5 * vec2(-w.x, -w.z) / abs(w.y); break; } // ft
            case 4: { side = params.sky_up_dn & 0xffff; st = 0.5 + 0.5 * vec2(-w.y,  w.x) / abs(w.z); break; } // up
            case 5: { side = params.sky_up_dn >> 16   ; st = 0.5 + 0.5 * vec2(-w.y, -w.x) / abs(w.z); break; } // dn
        }
        if (side < MAX_GLTEXTURES)
            emm += f16vec3(texture(img_tex[nonuniformEXT(side)], st).rgb);
    }

    return emm;
}
