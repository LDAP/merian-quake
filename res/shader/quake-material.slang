// Quake material model: registered with merian::MaterialSystem from QuakeScene.
// Payload layout mirrors QuakeMaterialPayload in src/game/quake_material.hpp.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;

#include "shader/quake-material.slangh"

namespace merian {

// this took way to long to get right...
float3 ldr_to_hdr(float3 color) {
    const float l = clamp(pow((color.r + color.g + color.b) / 3, .1), 0, 0.99);
    return sqrt(color) * 2 * (l / (1.0 - l));
}

float2 quake_warp(const float2 st, const float time) {
    return 2 * ((st) + 0.125 * sin(float2(3 * (st).yx + 0.5 * (time))));
}

float2 quake_waves(const float2 st, const float time) {
    return (-float2(.1, 0) * cos((st).x * 5 + (time) * 2) *
                pow(max(sin((st).x * 5 + (time) * 2), 0), 5) -
            float2(.07, 0) * cos(-(st).x * 5 + -(st).y * 3 + (time) * 3) *
                pow(max(sin(-(st).x * 5 + -(st).y * 3 + (time) * 3), 0), 5));
}

//     // HIT
//     const VertexExtraData extra_data =
//     buf_ext[nonuniformEXT(rq_instance_id(ray_query))].v[rq_primitive_index(ray_query)]; const
//     uint16_t flags = extra_data.texnum_fb_flags >> 12;

//     if (flags == MAT_TYPE_SKY) {
//         const f16vec3 sky = get_sky(hit.wi);
//         contribution += throughput * sky;
//         hit.albedo = sky;

//         hit.prev_pos = hit.pos = hit.pos + hit.wi * T_MAX;
//         hit.enc_geonormal = geo_encode_normal(-hit.wi);
//         hit.normal = -hit.wi;
//         return;
//     }

//     const vec3 bary = rq_barycentrics(ray_query);
//     vec2 st = extra_data.st * bary;
//     if (flags > 0 && flags < 5) {
//         st = MERIAN_TEXTUREEFFECT_QUAKE_WARPCALC(st, params.cl_time);
//         if (flags == MAT_TYPE_WATER) {
//             st += MERIAN_TEXTUREEFFECT_WAVES(st, params.cl_time);
//             hit.roughness = 0.4hf;
//         }
//     }

//     // NORMALS AND GLOSS
//     mat2x3 dudv;
//     const f16mat2 st_dudv = f16mat2(extra_data.st[2] - extra_data.st[0],
//                                     extra_data.st[1] - extra_data.st[0]);
//     {
//         const uvec3 prim_indexes =
//         buf_idx[nonuniformEXT(rq_instance_id(ray_query))].i[rq_primitive_index(ray_query)];
//         vec3 verts[3];
// #ifdef MERIAN_SPIRV_CAP_SUPPORTED_RayQueryPositionFetchKHR
//         rayQueryGetIntersectionTriangleVertexPositionsEXT(ray_query, true, verts);
// #else
//         verts[0] = buf_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.x];
//         verts[1] = buf_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.y];
//         verts[2] = buf_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.z];
// #endif
//         hit.pos = verts[0] * bary.x  + verts[1] * bary.y + verts[2] * bary.z;
//         dudv[0] = verts[2] - verts[0];
//         dudv[1] = verts[1] - verts[0];
//         hit.normal = normalize(cross(dudv[0], dudv[1]));
//         hit.enc_geonormal = geo_encode_normal(hit.normal);

//         hit.prev_pos =
//         buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.x] * bary.x
//                      +
//                      buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.y]
//                      * bary.y
//                      +
//                      buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.z]
//                      * bary.z;
//     }

// #if defined(MERIAN_QUAKE_FIRST_HIT) && (ENABLE_ALBEDO_MIPMAP || ENABLE_EMISSION_MIPMAP)
//         RayDifferential rd = ray_diff_create(vec3(0), vec3(0), r_x, r_y);
//         ray_diff_propagate(rd, hit.wi, rq_get_t(ray_query), hit.normal);
//         const mat3x2 pinv = pseudoinverse(dudv);

//         const vec2 grad_x = st_dudv * (pinv * rd.dOdx);
//         const vec2 grad_y = st_dudv * (pinv * rd.dOdy);
// #endif

// #if defined(MERIAN_QUAKE_FIRST_HIT) && ENABLE_ALBEDO_MIPMAP
//         const f16vec4 albedo_texture =
//         pow(f16vec4(textureGrad(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff,
//         MAX_GLTEXTURES - 1))], st, grad_x, grad_y)), f16vec4(1. / 1.2));
// #else
//         const f16vec4 albedo_texture =
//         pow(f16vec4(textureLod(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff,
//         MAX_GLTEXTURES - 1))], st, 0)), f16vec4(1. / 1.2));
// #endif

//     if (extra_data.n1_brush == 0xffffffff) {
//         const uint16_t texnum_normal = uint16_t(extra_data.n0_gloss_norm >> 16);
//         const uint16_t texnum_gloss = uint16_t(extra_data.n0_gloss_norm & 0xffff);

//         if (texnum_normal > 0 && texnum_normal < MAX_GLTEXTURES) {
//             const vec3 tangent_normal = (textureLod(img_tex[nonuniformEXT(texnum_normal)],
//             st, 0).rgb - 0.5) * 2; const float16_t st_det = st_dudv[0].x * st_dudv[1].y -
//             st_dudv[1].x * st_dudv[0].y; if (abs(st_det) > 0.0hf) {
//                 const vec3 du2 =  normalize(( st_dudv[1].y * dudv[0] - st_dudv[0].y *
//                 dudv[1]) / st_det); dudv[1] = -normalize((-st_dudv[1].x * dudv[0] +
//                 st_dudv[0].x * dudv[1]) / st_det); dudv[0] = du2;
//             }

//             const vec3 geo_normal = hit.normal;
//             hit.normal = normalize(dudv[0] * tangent_normal.x + dudv[1] * tangent_normal.y +
//             hit.normal * tangent_normal.z);

//             // Keller et al. [2017] workaround for artifacts
//             const vec3 r = reflect(hit.wi, hit.normal);
//             if (dot(r, geo_normal) < 0) {
//                 hit.normal = normalize(-hit.wi + normalize(r - geo_normal * dot(geo_normal,
//                 r)));
//             }
//         }

//         if (texnum_gloss > 0 && texnum_gloss < MAX_GLTEXTURES) {
//             hit.roughness = float16_t(textureLod(img_tex[nonuniformEXT(texnum_gloss)], st,
//             0).r);;
//         }
//     } else if (flags == MAT_TYPE_SOLID) {
//         hit.albedo = f16vec3(unpack8(extra_data.n0_gloss_norm).rgb) / 255.hf;
//         contribution += throughput * ldr_to_hdr(f16vec3(unpack8(extra_data.n1_brush).rgb) /
//         255.hf); return;
//     } else {
//         // Only for alias models. Disabled for now, results in artifacts.

//         // hit.normal = normalize(mat3(geo_decode_normal(extra_data.n0_gloss_norm),
//         //                             geo_decode_normal(extra_data.n1_brush),
//         //                             geo_decode_normal(extra_data.n2)) *
//         rq_barycentrics(ray_query));
//     }

//     // MATERIAL
//     if (flags == MAT_TYPE_WATERFALL) {
//         hit.albedo = albedo_texture.rgb;
//         contribution += throughput * hit.albedo;
//     } else if (flags == MAT_TYPE_SPRITE || flags == MAT_TYPE_TELE) {
//         hit.albedo = ldr_to_hdr(albedo_texture.rgb);
//         contribution += throughput * hit.albedo;
//     } else {
//         const uint16_t texnum_fb = extra_data.texnum_fb_flags & 0xfffs;
//         hit.albedo = albedo_texture.rgb;
//         if (texnum_fb > 0 && texnum_fb < MAX_GLTEXTURES) {

// #if defined(MERIAN_QUAKE_FIRST_HIT) && ENABLE_EMISSION_MIPMAP
//             const f16vec3 emission =
//             ldr_to_hdr(f16vec3(textureGrad(img_tex[nonuniformEXT(texnum_fb)], st, grad_x,
//             grad_y).rgb));
// #else
//             const f16vec3 emission =
//             ldr_to_hdr(f16vec3(textureLod(img_tex[nonuniformEXT(texnum_fb)], st, 0).rgb));
// #endif

//             if (any(greaterThan(emission, f16vec3(0)))) {
//                 contribution += throughput * emission;
//                 hit.albedo = emission;
//             }
//         }
//     }

struct QuakeMaterial : MaterialModel {
    typealias MaterialSampleImpl = QuakeMaterialSample;

    MaterialHeader header;

    // Compact payload (must match QuakeMaterialPayload C++-side layout). The
    // albedo texture lives in header.alpha_texture_id; only fullbright/normal/
    // gloss go in the payload.
    TextureID fullbright_tex;
    TextureID normal_tex;
    TextureID gloss_tex;
    uint16_t flags;     // MAT_TYPE_*
    uint8_t alpha_mode; // 0: use texture alpha; [1,15] map to [0,1] opaque
    uint8_t _pad;

    uint16_t get_flags() {
        return flags;
    }

    QuakeMaterialSample
    get_sample(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        // Normal
        QuakeMaterialSample s;
        if (normal_tex != TextureID(-1)) {
            const float3 n = (tm.get_sampler2D(normal_tex).SampleLevel(sd.uv, 0).rgb - 0.5) * 2;
            // TODO: n = clamp_to_geo(n, n_geo, wi);
            s.frame = sd.frame.apply_normal_map(n);
        } else {
            s.frame = sd.frame;
        }

        // Gloss / Roughness
        if (gloss_tex != TextureID(-1)) {
            // hit.roughness = tm.get_sampler2D(gloss_tex).SampleLevel(sd.uv, 0).rg;
        }

        // Albedo / Emission
        float2 uv = sd.uv;
        if ((flags & MAT_TYPE_WARP) != 0u) {
            uv = quake_warp(uv, sd.time);
            if (flags == MAT_TYPE_WATER) {
                uv += quake_waves(uv, sd.time);
                // hit.roughness = 0.4hf;
            }
        }

        float3 albedo = float3(1, 0, 1);
        if (let albedo_tex_id = header.get_alpha_texture_id()) {
            albedo = lod.sample(tm.get_sampler2D(albedo_tex_id), uv).rgb;
            // desaturate for a better look
            albedo = pow(albedo, 1. / 1.2);
        }

        if (get_flags() == MAT_TYPE_WATERFALL) {
            s.emission = albedo;
            // } else if (get_flags() == MAT_TYPE_TELE || get_flags() == MAT_TYPE_SPRITE) {
            //     TODO: Fix on host side!
            //     s.emission = ldr_to_hdr(albedo);
        } else if (fullbright_tex != TextureID(-1)) {
            s.emission = ldr_to_hdr(lod.sample(tm.get_sampler2D(fullbright_tex), uv).rgb);
        } else {
            s.emission = float3(0);
        }

        s.brdf = LambertDiffuseBRDF(albedo);
        return s;
    }
}

struct QuakeMaterialSample : MaterialSample {
    typealias BSDFImpl = LambertDiffuseBRDF;

    LambertDiffuseBRDF brdf;
    OrthonormalFrame frame;
    float3 emission;

    BSDFImpl get_bsdf() {
        return brdf;
    }
    property OrthonormalFrame shading_frame {
        get { return frame; }
    }
    property float3 shading_normal {
        get { return frame.get_normal(); }
    }
}

}
