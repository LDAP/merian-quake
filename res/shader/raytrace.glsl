#ifndef _MERIAN_QUAKE_RT_
#define _MERIAN_QUAKE_RT_

#extension GL_EXT_ray_tracing                       : enable
#extension GL_EXT_ray_query                         : enable
#extension GL_EXT_nonuniform_qualifier              : enable

#ifdef MERIAN_CONTEXT_EXT_ENABLED_ExtensionVkRayTracingPositionFetch
#extension GL_EXT_ray_tracing_position_fetch        : enable
#endif

#include "merian-shaders/normal_encode.glsl"
#include "merian-shaders/transmittance.glsl"
#include "merian-shaders/von_mises_fisher.glsl"
#include "merian-shaders/cubemap.glsl"
#include "merian-shaders/linalg.glsl"
#include "merian-shaders/ray_differential.glsl"
#include "merian-shaders/raytrace.glsl"
#include "merian-shaders/textures.glsl"

// Water shader source: https://www.shadertoy.com/view/MdXyzX
// afl_ext MIT-License

#define DRAG_MULT 0.28 // changes how much waves pull on the water
#define WATER_DEPTH 1. // how deep is the water
#define ITERATIONS_NORMAL 25 // waves iterations when calculating normals

// Calculates wave value and its derivative, 
// for the wave direction, position in space, wave frequency and time
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
  float x = dot(direction, position) * frequency + timeshift;
  float wave = exp(sin(x) - 1.0);
  float dx = wave * cos(x);
  return vec2(wave, -dx);
}

// Calculates waves by summing octaves of various waves with various parameters
float getwaves(vec2 position, int iterations) {
  float iter = 0.0; // this will help generating well distributed wave directions
  float frequency = 1.0; // frequency of the wave, this will change every iteration
  float timeMultiplier = 1.0; // time multiplier for the wave, this will change every iteration
  float weight = 1.0;// weight in final sum for the wave, this will change every iteration
  float sumOfValues = 0.0; // will store final sum of values
  float sumOfWeights = 0.0; // will store final sum of weights
  for(int i=0; i < iterations; i++) {
    // generate some wave direction that looks kind of random
    vec2 p = vec2(sin(iter), cos(iter));
    // calculate wave data
    vec2 res = wavedx(position, p, frequency, params.cl_time * timeMultiplier);

    // shift position around according to wave drag and derivative of the wave
    position += p * res.y * weight * DRAG_MULT;

    // add the results to sums
    sumOfValues += res.x * weight;
    sumOfWeights += weight;

    // modify next octave parameters
    weight *= 0.82;
    frequency *= 1.18;
    timeMultiplier *= 1.07;

    // add some kind of random value to make next wave look random too
    iter += 1232.399963;
  }
  // calculate and return
  return sumOfValues / sumOfWeights;
}

// Calculate normal at point by calculating the height at the pos and 2 additional points very close to pos
vec3 water_normal(vec2 pos, float e, float depth) {
  vec2 ex = vec2(e, 0);
  float H = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
  vec3 a = vec3(pos.x, H, pos.y);
  return normalize(
    cross(
      a - vec3(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y), 
      a - vec3(pos.x, getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
    )
  );
}


// assert(alpha != 0)
#define decode_alpha(enc_alpha) (float16_t(enc_alpha - 1) / 14.hf)

f16vec3 get_sky(const vec3 w) {
    f16vec3 emm = f16vec3(0);

    {
        emm += 0.5hf * pow(0.5hf * (1.0hf + float16_t(dot(vec3(SUN_W_X, SUN_W_Y, SUN_W_Z), w))), 4.0hf);
        emm += 5.0hf * float16_t(vmf_pdf(3000.0, dot(vec3(SUN_W_X, SUN_W_Y, SUN_W_Z), w)));
        emm *= f16vec3(SUN_COLOR_R, SUN_COLOR_G, SUN_COLOR_B);
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

// Sets T_MAX and T_MIN accordingly, avoiding unnecessary intersection tests.
// If this ray hits noting, the surface can be considered visible
void trace_visibility_ray_init(accelerationStructureEXT tlas, rayQueryEXT ray_query, const vec3 from, const vec3 to, const float offset /* = 1e-3 */) {
    const vec3 wo = to - from;
    rayQueryInitializeEXT(ray_query,
                          tlas,
                          gl_RayFlagsCullBackFacingTrianglesEXT,  // We need to cull backfaces
                                                                  // else we get z-fighting in Quake
                          0xFF,                 // 8-bit instance mask, here saying "trace against all instances"
                          from,
                          offset,                                 // T_MIN
                          normalize(wo),
                          max(offset, length(wo) - 2 * offset));  // T_MAX
}

void trace_ray_init(rayQueryEXT ray_query, const vec3 direction, const vec3 position) {
    rayQueryInitializeEXT(ray_query,
                          tlas,
                          gl_RayFlagsCullBackFacingTrianglesEXT,  // We need to cull backfaces
                                                                  // else we get z-fighting in Quake
                          0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                          position,
                          0,                     // Minimum t-value (we set it here to 0 and pull back the ray)
                                                 // such that the ray cannot escape in corners.
                          direction,
                          T_MAX);                // Maximum t-value
}

void trace_ray(rayQueryEXT ray_query) {
    VertexExtraData extra_data;
    uint16_t flags;
    vec2 st;

    while(rayQueryProceedEXT(ray_query)) {
        extra_data = buf_ext[nonuniformEXT(rq_instance_id_uc(ray_query))].v[rq_primitive_index_uc(ray_query)];
        flags = extra_data.texnum_fb_flags >> 12;
        const uint16_t alpha = extra_data.texnum_alpha >> 12;
        if (flags > 0 && flags < 7) {
            // treat sky, lava, slime,... not transparent for now
            rayQueryConfirmIntersectionEXT(ray_query);
        } else if (alpha != 0) { // 0 means use texture
            if (decode_alpha(alpha) >= ALPHA_THRESHOLD) {
                rayQueryConfirmIntersectionEXT(ray_query);
            }
        } else {
            // We covered the flags above, this surface cannot warp
            st = extra_data.st * rq_barycentrics_uc(ray_query);
            if (textureGather(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, 3).r >= ALPHA_THRESHOLD) {
                rayQueryConfirmIntersectionEXT(ray_query);
            }
        }
    }
}

// special case: intersections with skybox do not count...
bool trace_visibility(accelerationStructureEXT tlas, const vec3 from, const vec3 to) {
    rayQueryEXT ray_query;

    trace_visibility_ray_init(tlas, ray_query, from, to, 1e-3);

    trace_ray(ray_query);

    if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        // Nothing hit. Surface should be visible!
        return true;// f16vec3(transmittance3(distance(from, to), MU_T, VOLUME_MAX_T));
    }

    // Need this for restir, because with artificially move the position in trace ray when we hit the skybox...
    const uint16_t flags = buf_ext[nonuniformEXT(rq_instance_id(ray_query))].v[rq_primitive_index(ray_query)].texnum_fb_flags >> 12;
    if (flags == MAT_FLAGS_SKY) {
        return true;
    }

    return false;
}

bool trace_visibility(const vec3 from, const vec3 to) {
    return trace_visibility(tlas, from, to);
}

// Initialize pos and wi with ray origin and ray direction and
// throughput, contribution as needed
// 
// Returns the throughput along the ray (without hit)
// and contribution (with hit) multiplied with throughput.
// (allows for volumetric effects)
#ifdef MERIAN_QUAKE_FIRST_HIT
void trace_ray(inout f16vec3 throughput, inout f16vec3 contribution, inout Hit hit, const vec3 r_x, const vec3 r_y) {
#else
void trace_ray(inout f16vec3 throughput, inout f16vec3 contribution, inout Hit hit) {
#endif

    rayQueryEXT ray_query;

    bool was_water = false;
    while(true) {

    // FIND NEXT HIT
    trace_ray_init(ray_query, hit.wi, hit.pos);

    trace_ray(ray_query);

    throughput *= float16_t(transmittance3(rq_get_t(ray_query), MU_T, VOLUME_MAX_T));
    hit.roughness = 0.6hf;

    // NO HIT this should not happen in Quake, but it does -> treat that as sky.
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        const f16vec3 sky = get_sky(hit.wi);
        contribution = throughput * sky;
        hit.albedo = sky;

        if (!was_water)
            hit.prev_pos = hit.pos = hit.pos + hit.wi * T_MAX;
        hit.enc_geonormal = geo_encode_normal(-hit.wi);
        hit.normal = -hit.wi;
        return;
    }

    // HIT
    const VertexExtraData extra_data = buf_ext[nonuniformEXT(rq_instance_id(ray_query))].v[rq_primitive_index(ray_query)];
    const uint16_t flags = extra_data.texnum_fb_flags >> 12;

    if (flags == MAT_FLAGS_SKY) {
        const f16vec3 sky = get_sky(hit.wi);
        contribution += throughput * sky;
        hit.albedo = sky;

        if (!was_water)
            hit.prev_pos = hit.pos = hit.pos + hit.wi * T_MAX;
        hit.enc_geonormal = geo_encode_normal(-hit.wi);
        hit.normal = -hit.wi;
        return;
    }

    const vec3 bary = rq_barycentrics(ray_query);
    vec2 st = extra_data.st * bary;
    if (flags > 0 && flags < 5) {
        st = MERIAN_TEXTUREEFFECT_QUAKE_WARPCALC(st, params.cl_time);
        if (flags == MAT_FLAGS_WATER) {
            st += MERIAN_TEXTUREEFFECT_WAVES(st, params.cl_time);
            hit.roughness = 0.4hf;
        }
    }

    // NORMALS AND GLOSS
    mat2x3 dudv;
    const f16mat2 st_dudv = f16mat2(extra_data.st[2] - extra_data.st[0],
                                    extra_data.st[1] - extra_data.st[0]);
    {
        const uvec3 prim_indexes = buf_idx[nonuniformEXT(rq_instance_id(ray_query))].i[rq_primitive_index(ray_query)];
        vec3 verts[3];
        #ifdef MERIAN_CONTEXT_EXT_ENABLED_ExtensionVkRayTracingPositionFetch
        rayQueryGetIntersectionTriangleVertexPositionsEXT(ray_query, true, verts);
        #else
        verts[0] = buf_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.x];
        verts[1] = buf_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.y];
        verts[2] = buf_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.z];
        #endif
        hit.pos = verts[0] * bary.x  + verts[1] * bary.y + verts[2] * bary.z;
        dudv[0] = verts[2] - verts[0];
        dudv[1] = verts[1] - verts[0];
        hit.normal = normalize(cross(dudv[0], dudv[1]));
        hit.enc_geonormal = geo_encode_normal(hit.normal);

        hit.prev_pos = buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.x] * bary.x
        + buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.y] * bary.y
        + buf_prev_vtx[nonuniformEXT(rq_instance_id(ray_query))].v[prim_indexes.z] * bary.z;
    }

    #ifdef MERIAN_QUAKE_FIRST_HIT
    f16vec4 albedo_texture;
    if (ENABLE_MIPMAP) {
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
                const vec3 tangent_normal = (textureLod(img_tex[nonuniformEXT(texnum_normal)], st, 0).rgb - 0.5) * 2;
                const float16_t st_det = st_dudv[0].x * st_dudv[1].y - st_dudv[1].x * st_dudv[0].y;
                if (abs(st_det) > 1e-8) {
                    const vec3 du2 =  normalize(( st_dudv[1].y * dudv[0] - st_dudv[0].y * dudv[1]) / st_det);
                    dudv[1] = -normalize((-st_dudv[1].x * dudv[0] + st_dudv[0].x * dudv[1]) / st_det);
                    dudv[0] = du2;
                }

                const vec3 geo_normal = hit.normal;
                hit.normal = normalize(dudv[0] * tangent_normal.x + dudv[1] * tangent_normal.y + hit.normal * tangent_normal.z);

            // Keller et al. [2017] workaround for artifacts
            const vec3 r = reflect(hit.wi, hit.normal);
            if (dot(r, geo_normal) < 0) {
                hit.normal = normalize(-hit.wi + normalize(r - geo_normal * dot(geo_normal, r)));
            } 
        }

        if (texnum_gloss > 0 && texnum_gloss < MAX_GLTEXTURES) {
            hit.roughness = float16_t(textureLod(img_tex[nonuniformEXT(texnum_gloss)], st, 0).r);;
        }
        } else if (flags == MAT_FLAGS_SOLID) {
            hit.albedo = f16vec3(unpack8(extra_data.n0_gloss_norm).rgb) / 255.hf;
            contribution += throughput * ldr_to_hdr(f16vec3(unpack8(extra_data.n1_brush).rgb) / 255.hf);
            return;
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
    } else if (!was_water && flags == MAT_FLAGS_WATER) {
        hit.albedo = albedo_texture.rgb;

        #define WATER_IOR 1.33
        #define CRIT_ANGLE 0.02
        const bool underwater = true;
        hit.normal = make_frame(hit.normal) * water_normal(hit.pos.xy / 50, 0.1, WATER_DEPTH).xzy;
        float fresnel;
        if (underwater) fresnel = CRIT_ANGLE + (1.0-CRIT_ANGLE) * pow(clamp(WATER_IOR*WATER_IOR*(1.0 - dot(-hit.wi, hit.normal)*dot(-hit.wi, hit.normal)), 0.0, 1.0), 5.0);
        else            fresnel = CRIT_ANGLE + (1.0-CRIT_ANGLE) * pow(clamp(1.0 - dot(-hit.wi, hit.normal), 0.0, 1.0), 5.0);

        vec3 w = hit.wi;
        if (fresnel < 1.) {
            // refract
            if (underwater) hit.wi = refract(hit.wi, hit.normal, WATER_IOR);
            else            hit.wi = refract(hit.wi, hit.normal, 1.0 / WATER_IOR);

            //throughput /= uint16_t(1. - fresnel);
        } else {
            // reflect
            hit.wi = reflect(hit.wi, hit.normal);
            //throughput /= uint16_t(fresnel);
        }
        hit.pos += w * rq_get_t(ray_query) * abs(dot(-w, hit.normal)/dot(hit.wi, hit.normal)) * 1.0/WATER_IOR;
        was_water = true;
        continue;
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
    }
}

#endif // _MERIAN_QUAKE_RT_
