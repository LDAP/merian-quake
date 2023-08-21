#include "common/von_mises_fisher.glsl"
#include "common/cubemap.glsl"

struct ShadingMaterial {
    vec4 albedo;
    vec3 emission;
    vec3 normal; // normalized
    vec3 geo_normal;
    vec3 pos;
    float16_t gloss;
};

// assert(alpha != 0)
float decode_alpha(const uint alpha) {
    return float(alpha - 1) / 14.;
}

// Get warped texture coordinate
void warp(inout vec2 st) {
    st += vec2(0.1 * sin(st.y * 1.5 + params.cl_time * 1.0),
              0.1 * sin(st.x * 1.4 + params.cl_time * 1.0));
    st -= vec2(.1, 0) * cos(st.x * 5 + params.cl_time * 2) * pow(max(sin(st.x * 5 + params.cl_time * 2), 0), 5);
    st -= vec2(.07, 0) * cos(-st.x * 5 + -st.y * 3 + params.cl_time * 3) * pow(max(sin(-st.x * 5 + -st.y * 3 + params.cl_time * 3), 0), 5);
}

vec3 get_emission(const uint texnum_fb, const vec2 st, const vec3 albedo, const uint16_t flags) {
    vec3 emission;

    if (flags == MAT_FLAGS_LAVA)
        return 20.0 * albedo;
    else if (flags == MAT_FLAGS_SLIME)
        return 0.5 * albedo;
    else if (flags == MAT_FLAGS_TELE)
        return 5.0 * albedo;
    else if (flags == MAT_FLAGS_WATERFALL)
        return albedo;
    else if (flags == MAT_FLAGS_SPRITE)
        emission = albedo;
    else if (texnum_fb > 0 && texnum_fb < MAX_GLTEXTURES) {
        emission = texture(img_tex[nonuniformEXT(texnum_fb)], st).rgb;
    } else {
        return vec3(0);
    }

    const float sum = emission.x + emission.y + emission.z;
    if (sum > 0) {
        emission /= sum;
        emission *= 10.0 * (exp2(3.5 * sum) - 1.0);
    }
    return emission;
}

void get_verts_pos_geonormal(out mat3 verts,
                             out vec3 pos,
                             out vec3 geo_normal,
                             const IntersectionInfo info) {
    const uvec3 prim_indexes = buf_idx[nonuniformEXT(info.instance_id)].i[info.primitive_index];
    verts = mat3(buf_vtx[nonuniformEXT(info.instance_id)].v[prim_indexes.x],
                 buf_vtx[nonuniformEXT(info.instance_id)].v[prim_indexes.y],
                 buf_vtx[nonuniformEXT(info.instance_id)].v[prim_indexes.z]);
    pos = verts * info.barycentrics;
    geo_normal = normalize(cross(verts[2] - verts[0], verts[1] - verts[0]));
}

vec3 apply_normalmap(const vec3 v0,
                     const vec3 v1,
                     const vec3 v2,
                     const f16vec2 st0,
                     const f16vec2 st1,
                     const f16vec2 st2,
                     const vec3 geo_normal,
                     const vec3 tangent_normal) {
    vec3 du = v2 - v0, dv = v1 - v0;
    const f16vec2 duv1 = st2 - st0, duv2 = st1 - st0;
    const float16_t det = duv1.x * duv2.y - duv2.x * duv1.y;
    if(abs(det) > 1e-8) {
      const vec3 du2 =  normalize(( duv2.y * du - duv1.y * dv) / det);
      dv = -normalize((-duv2.x * du + duv1.x * dv) / det);
      du = du2;
    }
    vec3 n = normalize(mat3(du, dv, geo_normal) * ((tangent_normal - vec3(0.5)) * vec3(2)));
    // if (dot(geo_normal, n) <= 0) n -= geo_normal * dot(geo_normal, n);
    return n;
}

void get_sky(const vec3 pos, const vec3 w, out ShadingMaterial mat) {
    if((params.sky_lf_ft & 0xffff) == 0xffff) {
        // classic quake sky
        const vec2 st = 0.5 + 0.5 * vec2(-w.y,w.x) / abs(w.z);
        const vec2 t = params.cl_time * vec2(0.12, 0.06);
        const vec4 bck = texture(img_tex[nonuniformEXT(min(params.sky_rt_bk & 0xffff, MAX_GLTEXTURES - 1))], st + 0.1 * t);
        const vec4 fnt = texture(img_tex[nonuniformEXT(min(params.sky_rt_bk >> 16   , MAX_GLTEXTURES - 1))], st + t);
        const vec3 tex = mix(bck.rgb, fnt.rgb, fnt.a);
        mat.emission = 10.0 * (exp2(3.5 * tex) - 1.0);
    } else {
        const vec3 sundir = normalize(vec3(SUN_W_X, SUN_W_Y, SUN_W_Z));
        const vec3 suncolor = vec3(SUN_COLOR_R, SUN_COLOR_G, SUN_COLOR_B);
        
        mat.emission = vec3(0.0);
        mat.emission += 0.5 * suncolor * pow(0.5 * (1.0 + dot(sundir, w)), 4.0);
        mat.emission += 5. * suncolor * vmf_pdf(3000.0, dot(sundir, w));
        
        // Evaluate cubemap
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
        if (side < MAX_GLTEXTURES)
            mat.emission += texture(img_tex[nonuniformEXT(side)], st).rgb;
    }

    mat.albedo = vec4(0, 0, 0, 1);
    mat.normal = -w;
    mat.geo_normal = -w;
    mat.pos = pos + T_MAX * w;
    mat.gloss = float16_t(0);
}

void get_shading_material(const IntersectionInfo info,
                          const vec3 ray_origin,
                          const vec3 ray_dir,
                          out ShadingMaterial mat) {
    const VertexExtraData extra_data =  buf_ext[nonuniformEXT(info.instance_id)].v[info.primitive_index];
    const uint16_t flags = extra_data.texnum_fb_flags >> 12;

    if (flags == MAT_FLAGS_SKY) {
        get_sky(ray_origin, ray_dir, mat);
        return;
    }

    vec2 st = extra_data.st_0 * info.barycentrics.x
            + extra_data.st_1 * info.barycentrics.y
            + extra_data.st_2 * info.barycentrics.z;

    if (flags > 0 && flags < 6) {
        warp(st);
    }

    {
        // Load albedo
        // const uint texnum = extra_data.texnum_alpha & 0xfff;
        // Clamp to 1e-3 (nothing is really 100% black)
        mat.albedo = max(texture(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st), vec4(vec3(1e-3), 1));
        const uint16_t alpha = extra_data.texnum_alpha >> 12;
        if (alpha != 0)
            mat.albedo.a = decode_alpha(alpha);
    }
    {
        // Load emission
        // const uint texnum_fb = extra_data.texnum_fb_flags & 0xfff;
        mat.emission = get_emission(extra_data.texnum_fb_flags & 0xfff, st, mat.albedo.rgb, flags);
    }

    mat3 verts;
    get_verts_pos_geonormal(verts, mat.pos, mat.geo_normal, info);
    mat.normal = mat.geo_normal;

    if (extra_data.n1_brush == 0xffffffff) { // is brush model
        // can have normal and gloss map (else has geo normals)
        const uint16_t texnum_gloss = uint16_t(extra_data.n0_gloss_norm & 0xffff);
        const uint16_t texnum_normal = uint16_t(extra_data.n0_gloss_norm >> 16);
        
        if (texnum_normal > 0 && texnum_normal < MAX_GLTEXTURES) {
            // overwrite geo normal with normal map normal
            const vec3 tangent_normal = texture(img_tex[nonuniformEXT(texnum_normal)], st).rgb;
            mat.normal = apply_normalmap(verts[0], verts[1], verts[2], extra_data.st_0, extra_data.st_1, extra_data.st_2, mat.normal, tangent_normal);
        }
        if (texnum_gloss > 0 && texnum_gloss < MAX_GLTEXTURES) {
            mat.gloss = float16_t(texture(img_tex[nonuniformEXT(texnum_gloss)], st).r);
        }
    } else {
        const vec3 n0 = geo_decode_normal(extra_data.n0_gloss_norm);
        const vec3 n1 = geo_decode_normal(extra_data.n1_brush);
        const vec3 n2 = geo_decode_normal(extra_data.n2);
        mat.normal = normalize(mat3(n0, n1, n2) * info.barycentrics);
        mat.gloss = float16_t(0);
    }
}
