
struct Hit {
    vec3 pos;
    vec3 wi;
    vec3 normal;
    uint enc_geonormal;

    // Material
    f16vec3 albedo;
};

#define _get_t(ray_query, commited) rayQueryGetIntersectionTEXT(ray_query, commited)
#define _instance_id(ray_query, commited) rayQueryGetIntersectionInstanceIdEXT(ray_query, commited)
#define _primitive_index(ray_query, commited) rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, commited)
#define _barycentrics(ray_query, commited) vec3(1.0 - dot(rayQueryGetIntersectionBarycentricsEXT(ray_query, commited), vec2(1)), rayQueryGetIntersectionBarycentricsEXT(ray_query, commited))

#define instance_id(ray_query) _instance_id(ray_query, true)
#define primitive_index(ray_query) _primitive_index(ray_query, true)  
#define barycentrics(ray_query) _barycentrics(ray_query, true)

#define instance_id_uc(ray_query) _instance_id(ray_query, false)
#define primitive_index_uc(ray_query) _primitive_index(ray_query, false)  
#define barycentrics_uc(ray_query) _barycentrics(ray_query, false)

f16vec3 ldr_to_hdr(f16vec3 color) {
    const float16_t sum = color.x + color.y + color.z;
    return (sum > 0.hf) ? color.rgb / sum * 10.0hf * (exp2(3.5hf * sum) - 1.0hf) : f16vec3(0);
}

// Get warped texture coordinate
vec2 warp(const vec2 st) {
    return st + vec2(0.1 * sin(st.y * 1.5 + params.cl_time * 1.0),
              0.1 * sin(st.x * 1.4 + params.cl_time * 1.0))
    - vec2(.1, 0) * cos(st.x * 5 + params.cl_time * 2) * pow(max(sin(st.x * 5 + params.cl_time * 2), 0), 5)
    - vec2(.07, 0) * cos(-st.x * 5 + -st.y * 3 + params.cl_time * 3) * pow(max(sin(-st.x * 5 + -st.y * 3 + params.cl_time * 3), 0), 5);
}

// Initialize pos and wi with ray origin and ray direction and
// throughput, contribution as needed
void trace_ray(inout f16vec3 throughput, inout f16vec3 contribution, inout Hit hit) {
    uint16_t intersection = 0s;

    // TODO: Max bounces (prevent infinite bounces)
    while (intersection++ < MAX_INTERSECTIONS) {
        rayQueryEXT ray_query;

        // FIND NEXT HIT
        rayQueryInitializeEXT(ray_query,
                              tlas,
                              gl_RayFlagsCullBackFacingTrianglesEXT, 
                                                                        // Ray flags, None means: If not set in the instance the triangle
                                                                        // may not be opaque. This is a major performance hit
                                                                        // since we need to load the texture to determine if we want to trace
                                                                        // further.
                                                                        // Use gl_RayFlagsOpaqueEXT to treat all geometry as opaque
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
                const VertexExtraData extra_data =  buf_ext[nonuniformEXT(instance_id_uc(ray_query))].v[primitive_index_uc(ray_query)];
                const uint16_t alpha = extra_data.texnum_alpha >> 12;
                const uint16_t flags = extra_data.texnum_fb_flags >> 12;
                if (flags != MAT_FLAGS_NONE && flags != MAT_FLAGS_SPRITE) {
                    // treat sky, lava, slime,... not transparent for now
                    confirm = true;
                } else if (alpha != 0) { // 0 means use texture
                    confirm = decode_alpha(alpha) >= ALPHA_THRESHOLD;
                } else {
                    const vec2 st = extra_data.st * barycentrics_uc(ray_query);

                    // We covered the flags above, this surface cannot warp
                    // if (flags > 0 && flags < 6) {
                    //     warp(st);
                    // }
                    confirm = textureGather(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, 3).r >= ALPHA_THRESHOLD;
                }

                if (confirm)
                    rayQueryConfirmIntersectionEXT(ray_query);
            }
        }

        // NO HIT this should not happen in Quake, but it does -> treat that as sky.
        if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
            const f16vec3 sky = get_sky(hit.wi);
            contribution = throughput * sky;
            hit.albedo = sky;

            hit.pos = hit.pos + hit.wi * T_MAX;
            hit.enc_geonormal = geo_encode_normal(-hit.wi);
            hit.normal = -hit.wi;
            break;
        }

        // HIT
        const VertexExtraData extra_data =  buf_ext[nonuniformEXT(instance_id(ray_query))].v[primitive_index(ray_query)];
        const uint16_t flags = extra_data.texnum_fb_flags >> 12;

        if (flags == MAT_FLAGS_SKY) {
            const f16vec3 sky = get_sky(hit.wi);
            contribution += throughput * sky;
            hit.albedo = sky;

            hit.pos = hit.pos + hit.wi * T_MAX;
            hit.enc_geonormal = geo_encode_normal(-hit.wi);
            hit.normal = -hit.wi;
            break;
        }

        const vec2 st = (flags > 0 && flags < 5) ? warp(extra_data.st * barycentrics(ray_query)) : extra_data.st * barycentrics(ray_query);
        const f16vec4 albedo_texture = f16vec4(texture(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st));
        // NORMALS AND GLOSS
        vec3 du, dv;
        {
            const uvec3 prim_indexes = buf_idx[nonuniformEXT(instance_id(ray_query))].i[primitive_index(ray_query)];
            const mat3 verts = mat3(buf_vtx[nonuniformEXT(instance_id(ray_query))].v[prim_indexes.x],
                                    buf_vtx[nonuniformEXT(instance_id(ray_query))].v[prim_indexes.y],
                                    buf_vtx[nonuniformEXT(instance_id(ray_query))].v[prim_indexes.z]);
            hit.pos = verts * barycentrics(ray_query);
            hit.normal = normalize(cross(verts[2] - verts[0], verts[1] - verts[0]));
            hit.enc_geonormal = geo_encode_normal(hit.normal);

            du = verts[2] - verts[0];
            dv = verts[1] - verts[0];
        }

        if (extra_data.n1_brush == 0xffffffff) {
            const uint16_t texnum_normal = uint16_t(extra_data.n0_gloss_norm >> 16);
            // const uint16_t texnum_gloss = uint16_t(extra_data.n0_gloss_norm & 0xffff);

            if (texnum_normal > 0 && texnum_normal < MAX_GLTEXTURES) {
                const vec3 tangent_normal = texture(img_tex[nonuniformEXT(texnum_normal)], st).rgb;
                const f16vec2 duv1 = extra_data.st[2] - extra_data.st[0], duv2 = extra_data.st[1] - extra_data.st[0];
                const float16_t det = duv1.x * duv2.y - duv2.x * duv1.y;

                if (abs(det) > 1e-8) {
                    const vec3 du2 =  normalize(( duv2.y * du - duv1.y * dv) / det);
                    dv = -normalize((-duv2.x * du + duv1.x * dv) / det);
                    du = du2;
                }

                hit.normal = normalize(mat3(du, dv, hit.normal) * ((tangent_normal - vec3(0.5)) * vec3(2)));
            }


            // if (texnum_gloss > 0 && texnum_gloss < MAX_GLTEXTURES) {
            //     mat.gloss = float16_t(texture(img_tex[nonuniformEXT(texnum_gloss)], st).r);
            // }
        } else {
            hit.normal = normalize(mat3(geo_decode_normal(extra_data.n0_gloss_norm),
                                        geo_decode_normal(extra_data.n1_brush),
                                        geo_decode_normal(extra_data.n2)) * barycentrics(ray_query));
        }

        // MATERIAL
        if (flags == MAT_FLAGS_WATER) {
            hit.albedo = albedo_texture.rgb;
            //throughput *= albedo_texture.rgb;
            //hit.pos += 1e-3 * hit.wi;
            //continue;
        } else if (flags == MAT_FLAGS_WATERFALL) {
            contribution += throughput * albedo_texture.rgb;
            hit.albedo = albedo_texture.rgb;
            //hit.pos += 1e-3 * hit.wi;
            //continue;
        } else if (flags == MAT_FLAGS_SPRITE) {
            hit.albedo = ldr_to_hdr(albedo_texture.rgb);
            contribution += throughput * hit.albedo;
        } else if (flags == MAT_FLAGS_LAVA) {
            hit.albedo = 20.0hf * albedo_texture.rgb;
            contribution += throughput * hit.albedo;
        } else if (flags == MAT_FLAGS_SLIME) {
            hit.albedo = 0.5hf * albedo_texture.rgb;
            contribution += throughput * hit.albedo;
        } else if (flags == MAT_FLAGS_TELE) {
            hit.albedo = 5.0hf * albedo_texture.rgb;
            contribution += throughput * hit.albedo;
        } else {
            const uint16_t texnum_fb = uint16_t(extra_data.texnum_fb_flags & 0xfff);
            hit.albedo = albedo_texture.rgb;
            if (texnum_fb > 0 && texnum_fb < MAX_GLTEXTURES) {
                const f16vec3 emission = ldr_to_hdr(f16vec3(texture(img_tex[nonuniformEXT(texnum_fb)], st).rgb));
                if (any(greaterThan(emission, f16vec3(0)))) {
                    contribution += throughput * emission;
                    hit.albedo = emission;
                }
            }
        }

        break;
    }
}
