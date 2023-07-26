#include "common/von_mises_fisher.glsl"
#include "common/cubemap.glsl"

vec3 envmap(in vec3 w) {
    if((params.sky_lf_ft & 0xffff) == 0xffff) {
        // classic quake sky
        const vec2 st = 0.5 + 0.5 * vec2(-w.y,w.x) / abs(w.z);
        const vec2 t = params.cl_time * vec2(0.12, 0.06);
        const vec4 bck = texture(img_tex[nonuniformEXT(params.sky_rt_bk & 0xffff)], st + 0.1 * t);
        const vec4 fnt = texture(img_tex[nonuniformEXT(params.sky_rt_bk >> 16   )], st + t);
        const vec3 tex = mix(bck.rgb, fnt.rgb, fnt.a);
        return 50 * tex;
    } else {
        // Add a custom sun using vmf lobe
        // vec3 sundir = normalize(vec3(1, 1, 1)); // this where the moon is in ad_azad
        // vec3 sundir = normalize(vec3(1, -1, 1)); // this comes in more nicely through the windows for debugging
        vec3 sundir = normalize(vec3(1, -1, 1)); // ad_tears
        
        const float k0 = 4.0, k1 = 30.0, k2 = 4.0, k3 = 3000.0;
        vec3 emcol = vec3(0.0);
        emcol += vec3(0.50, 0.50, 0.50) * /*(k0+1.0)/(2.0*M_PI)*/ pow(0.5*(1.0+dot(sundir, w)), k0);
        emcol += vec3(1.00, 0.70, 0.30) * /*(k1+1.0)/(2.0*M_PI)*/ pow(0.5*(1.0+dot(sundir, w)), k1);
        emcol += 30.0*vec3(1.1, 1.0, 0.9)*vmf_pdf(k3, dot(sundir, w));
        emcol += vec3(0.20, 0.08, 0.02) * /*(k2+1.0)/(2.0*M_PI)*/ pow(0.5*(1.0-w.z), k2);
        
        // Evaluate cubemap
        // cubemap: gfx/env/*{rt,bk,lf,ft,up,dn}
        uint side = 0;
        vec2 st;
        switch(cubemap_side(w)) {
            case 0: { side = params.sky_rt_bk & 0xffff; st = 0.5 + 0.5*vec2(-w.y, -w.z) / abs(w.x); break; } // rt
            case 1: { side = params.sky_lf_ft & 0xffff; st = 0.5 + 0.5*vec2( w.y, -w.z) / abs(w.x); break; } // lf
            case 2: { side = params.sky_rt_bk >> 16   ; st = 0.5 + 0.5*vec2( w.x, -w.z) / abs(w.y); break; } // bk
            case 3: { side = params.sky_lf_ft >> 16   ; st = 0.5 + 0.5*vec2(-w.x, -w.z) / abs(w.y); break; } // ft
            case 4: { side = params.sky_up_dn & 0xffff; st = 0.5 + 0.5*vec2(-w.y,  w.x) / abs(w.z); break; } // up
            case 5: { side = params.sky_up_dn >> 16   ; st = 0.5 + 0.5*vec2(-w.y, -w.x) / abs(w.z); break; } // dn
        }
        emcol += texture(img_tex[nonuniformEXT(side)], st).rgb;
        return emcol;
    }
}
