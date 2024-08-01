#include "common/normal_encode.glsl"
#include "common/transmittance.glsl"
#include "common/von_mises_fisher.glsl"
#include "common/cubemap.glsl"
#include "common/linalg.glsl"
#include "common/ray_differential.glsl"
#include "common/raytrace.glsl"

// assert(alpha != 0)
#define decode_alpha(enc_alpha) (float16_t(alpha - 1) / 14.hf)

f16vec3 get_sky(const vec3 w) {
    f16vec3 emm = f16vec3(0);

    {
        const vec3 sundir = vec3(SUN_W_X, SUN_W_Y, SUN_W_Z);
        const f16vec3 suncolor = f16vec3(SUN_COLOR_R, SUN_COLOR_G, SUN_COLOR_B);

        emm += 0.5hf * pow(0.5hf * (1.0hf + float16_t(dot(sundir, w))), 4.0hf);
        emm += 5.0hf * float16_t(vmf_pdf(3000.0, dot(sundir, w)));
        emm *= suncolor;
    }


    if((params.sky_lf_ft & 0xffff) == 0xffff) {
        // classic quake sky
        const vec2 st = 0.5 + 1. * vec2(w.x , w.y) / abs(w.z);
        const vec2 t = params.cl_time * vec2(0.12, 0.12);
        const vec4 bck = textureLod(img_tex[nonuniformEXT(min(params.sky_rt_bk & 0xffff, MAX_GLTEXTURES - 1))], st + 0.5 * t, 0);
        const vec4 fnt = textureLod(img_tex[nonuniformEXT(min(params.sky_rt_bk >> 16   , MAX_GLTEXTURES - 1))], st + t, 0);
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
            emm += f16vec3(textureLod(img_tex[nonuniformEXT(side)], st, 0).rgb);
    }

    return emm;
}

f16vec3 ldr_to_hdr(f16vec3 color) {
    const float16_t sum = color.x + color.y + color.z;
    return (sum > 0.hf) ? color.rgb / sum * 10.0hf * (exp2(3.hf * sum) - 1.0hf) : f16vec3(0);
}

// adapted from gl_warp.c
#define WARPCALC(st) 2 * (st + 0.125 * sin(vec2(3 * st.yx + 0.5 * params.cl_time)))
#define WATER_WAVES(st) (-vec2(.1, 0) * cos(st.x * 5 + params.cl_time * 2) * pow(max(sin(st.x * 5 + params.cl_time * 2), 0), 5) \
                         -vec2(.07, 0) * cos(-st.x * 5 + -st.y * 3 + params.cl_time * 3) * pow(max(sin(-st.x * 5 + -st.y * 3 + params.cl_time * 3), 0), 5))

// Initialize pos and wi with ray origin and ray direction and
// throughput, contribution as needed
// 
// Returns the troughput along the ray (without hit)
// and contribution (with hit) multiplied with throuhput.
// (allows for volumetric effects)
#ifdef MERIAN_QUAKE_FIRST_HIT
void trace_ray(inout f16vec3 throughput, inout f16vec3 contribution, inout Hit hit, const vec3 r_x, const vec3 r_y) {
#else
void trace_ray(inout f16vec3 throughput, inout f16vec3 contribution, inout Hit hit) {
#endif
    uint16_t intersection = 0s;

    do {
        rayQueryEXT ray_query;
        VertexExtraData extra_data;
        uint16_t flags;
        vec2 st;

        // FIND NEXT HIT
        rayQueryInitializeEXT(ray_query,
                              tlas,
                              gl_RayFlagsCullBackFacingTrianglesEXT,  // We need to cull backfaces
                                                                      // else we get z-fighting in Quake
                              0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                              hit.pos,
                              0,                     // Minimum t-value (we set it here to 0 and pull back the ray)
                                                     // such that the ray cannot escape in corners.
                              hit.wi,
                              T_MAX);                // Maximum t-value
        while(rayQueryProceedEXT(ray_query)) {
            // We are only interessted in triangles. "false" means candidates too, not only commited
            if (rayQueryGetIntersectionTypeEXT(ray_query, false) != gl_RayQueryCandidateIntersectionTriangleEXT) {
                continue;
            } else {
                bool confirm;
                extra_data = buf_ext[nonuniformEXT(rq_instance_id_uc(ray_query))].v[rq_primitive_index_uc(ray_query)];
                flags = extra_data.texnum_fb_flags >> 12;
                const uint16_t alpha = extra_data.texnum_alpha >> 12;
                if (flags > 0 && flags < 7) {
                    // treat sky, lava, slime,... not transparent for now
                    confirm = true;
                } else if (alpha != 0) { // 0 means use texture
                    confirm = decode_alpha(alpha) >= ALPHA_THRESHOLD;
                } else {
                    // We covered the flags above, this surface cannot warp
                    st = extra_data.st * rq_barycentrics_uc(ray_query);
                    confirm = textureGather(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, 3).r >= ALPHA_THRESHOLD;
                }

                if (confirm)
                    rayQueryConfirmIntersectionEXT(ray_query);
            }
        }

        throughput *= float16_t(transmittance3(rq_get_t(ray_query), MU_T, VOLUME_MAX_T));
        hit.roughness = 0.02hf;

        // NO HIT this should not happen in Quake, but it does -> treat that as sky.
        if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
            const f16vec3 sky = get_sky(hit.wi);
            contribution = throughput * sky;
            hit.albedo = sky;

            hit.prev_pos = hit.pos = hit.pos + hit.wi * T_MAX;
            hit.enc_geonormal = geo_encode_normal(-hit.wi);
            hit.normal = -hit.wi;
            break;
        }

        // HIT
        extra_data = buf_ext[nonuniformEXT(rq_instance_id(ray_query))].v[rq_primitive_index(ray_query)];
        flags = extra_data.texnum_fb_flags >> 12;

        if (flags == MAT_FLAGS_SKY) {
            const f16vec3 sky = get_sky(hit.wi);
            contribution += throughput * sky;
            hit.albedo = sky;

            hit.prev_pos = hit.pos = hit.pos + hit.wi * T_MAX;
            hit.enc_geonormal = geo_encode_normal(-hit.wi);
            hit.normal = -hit.wi;
            break;
        }

        st = extra_data.st * rq_barycentrics(ray_query);
        if (flags > 0 && flags < 5) {
            st = WARPCALC(st);
            if (flags == MAT_FLAGS_WATER)
                st += WATER_WAVES(st);
        }

        // NORMALS AND GLOSS
        mat2x3 dudv;
        const f16mat2 st_dudv = f16mat2(extra_data.st[2] - extra_data.st[0],
                                        extra_data.st[1] - extra_data.st[0]);
        {
            vec3 verts[3];
            rayQueryGetIntersectionTriangleVertexPositionsEXT(ray_query, true, verts);
            hit.pos = mat3(verts[0], verts[1], verts[2]) * rq_barycentrics(ray_query);
            dudv[0] = verts[2] - verts[0];
            dudv[1] = verts[1] - verts[0];
            hit.normal = normalize(cross(dudv[0], dudv[1]));
            hit.enc_geonormal = geo_encode_normal(hit.normal);

            const uvec3 prim_indexes = buf_idx[nonuniformEXT(rq_instance_id(ray_query))].i[rq_primitive_index(ray_query)];
            hit.prev_pos = mat3(buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.x],
                                buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.y],
                                buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.z])
                         * rq_barycentrics(ray_query);
        }

#ifdef MERIAN_QUAKE_FIRST_HIT
    f16vec4 albedo_texture;
    if (ENABLE_MIPMAP) {
        //vec3 proj_x = dFdx(hit.pos);
        // const mat2 ATA = transpose(dudv) * dudv;
        // const mat2 ATA_inv = inverse(ATA);
        // const mat3x2 A_T = transpose(dudv);
        // const mat3x2 A_pseudo_inverse = ATA_inv * A_T;
        RayDifferential rd = ray_diff_create(vec3(0), vec3(0), r_x, r_y);
        ray_diff_propagate(rd, hit.wi, rq_get_t(ray_query), hit.normal);
        const mat3x2 pinv = pseudoinverse(dudv);

        const vec2 grad_x = st_dudv * (pinv * rd.dOdx);
        const vec2 grad_y = st_dudv * (pinv * rd.dOdy);

        albedo_texture = f16vec4(textureGrad(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, grad_x, grad_y));
    } else {
        albedo_texture = f16vec4(textureLod(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, 0));
    }
#else
    const f16vec4 albedo_texture = f16vec4(textureLod(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, 0));
#endif



        if (extra_data.n1_brush == 0xffffffff) {
            const uint16_t texnum_normal = uint16_t(extra_data.n0_gloss_norm >> 16);
            const uint16_t texnum_gloss = uint16_t(extra_data.n0_gloss_norm & 0xffff);

            if (texnum_normal > 0 && texnum_normal < MAX_GLTEXTURES) {
                const vec3 tangent_normal = textureLod(img_tex[nonuniformEXT(texnum_normal)], st, 0).rgb;
                const float16_t st_det = st_dudv[0].x * st_dudv[1].y - st_dudv[1].x * st_dudv[0].y;
                if (abs(st_det) > 1e-8) {
                    const vec3 du2 =  normalize(( st_dudv[1].y * dudv[0] - st_dudv[0].y * dudv[1]) / st_det);
                    dudv[1] = -normalize((-st_dudv[1].x * dudv[0] + st_dudv[0].x * dudv[1]) / st_det);
                    dudv[0] = du2;
                }

                const vec3 geo_normal = hit.normal;
                hit.normal = normalize(mat3(dudv[0], dudv[1], hit.normal) * ((tangent_normal - vec3(0.5)) * vec3(2)));

                // Keller et al. [2017] workaround for artifacts
                const vec3 r = reflect(hit.wi, hit.normal);
                if (dot(r, geo_normal) < 0) {
                    hit.normal = normalize(-hit.wi + normalize(r - geo_normal * dot(geo_normal, r)));
                } 
            }

            if (texnum_gloss > 0 && texnum_gloss < MAX_GLTEXTURES) {
                hit.roughness = mix(hit.roughness, 0.0001hf, float16_t(textureLod(img_tex[nonuniformEXT(texnum_gloss)], st, 0).r));
            }
        } else if (flags == MAT_FLAGS_SOLID) {
            hit.albedo = f16vec3(unpack8(extra_data.n0_gloss_norm).rgb) / 255.hf;
            contribution += throughput * ldr_to_hdr(f16vec3(unpack8(extra_data.n1_brush).rgb) / 255.hf);
            break;
        } else {
            // Only for alias models. Disabled for now, results in artifacts.

            // hit.normal = normalize(mat3(geo_decode_normal(extra_data.n0_gloss_norm),
            //                             geo_decode_normal(extra_data.n1_brush),
            //                             geo_decode_normal(extra_data.n2)) * rq_barycentrics(ray_query));
        }

        // MATERIAL
        if (flags == MAT_FLAGS_WATERFALL) {
            hit.albedo = albedo_texture.rgb;
            contribution += throughput * hit.albedo;
        } else if (flags == MAT_FLAGS_SPRITE || flags == MAT_FLAGS_TELE) {
            hit.albedo = ldr_to_hdr(albedo_texture.rgb);
            contribution += throughput * hit.albedo;
        } else {
            const uint16_t texnum_fb = extra_data.texnum_fb_flags & 0xfffs;
            hit.albedo = albedo_texture.rgb;
            if (texnum_fb > 0 && texnum_fb < MAX_GLTEXTURES) {
                const f16vec3 emission = ldr_to_hdr(f16vec3(textureLod(img_tex[nonuniformEXT(texnum_fb)], st, 0).rgb));
                if (any(greaterThan(emission, f16vec3(0)))) {
                    contribution += throughput * emission;
                    hit.albedo = emission;
                }
            }
        }

        break;
    } while (intersection++ < MAX_INTERSECTIONS);
}
