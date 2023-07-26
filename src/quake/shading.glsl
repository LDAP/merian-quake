struct ShadingMaterial {
    vec4 albedo;
    vec3 emission;
    vec3 normal; // normalized
    vec3 geo_normal;
    vec3 pos;
    float gloss;
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

vec3 get_emission(const uint texnum_fb, const vec2 st, const vec3 albedo, const uint flags) {
    if (flags == MAT_FLAGS_LAVA)
        return 20.0 * albedo;
    if (flags == MAT_FLAGS_SLIME)
        return 0.5 * albedo;
    if (flags == MAT_FLAGS_TELE)
        return 5.0 * albedo;
    if (flags == MAT_FLAGS_WATERFALL)
        return albedo;

    if (texnum_fb > 0) {
        vec3 emission = texture(img_tex[nonuniformEXT(texnum_fb)], st).rgb;
        const float sum = emission.x + emission.y + emission.z;
        if (sum > 0) {
            emission /= sum;
            emission *= 10.0 * (exp2(3.5 * sum) - 1.0);
        }
        return emission;
    }

    return vec3(0.);
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
                     const vec2 st0,
                     const vec2 st1,
                     const vec2 st2,
                     const vec3 geo_normal,
                     const vec3 tangent_normal) {
    vec3 du = v2 - v0, dv = v1 - v0;
    const vec2 duv1 = st2 - st0, duv2 = st1 - st0;
    const float det = duv1.x * duv2.y - duv2.x * duv1.y;
    if(abs(det) > 1e-8) {
      const vec3 du2 =  normalize(( duv2.y * du - duv1.y * dv) / det);
      dv = -normalize((-duv2.x * du + duv1.x * dv) / det);
      du = du2;
    }
    vec3 n = normalize(mat3(du, dv, geo_normal) * ((tangent_normal - vec3(0.5)) * vec3(2)));
    // if (dot(geo_normal, n) <= 0) n -= geo_normal * dot(geo_normal, n);
    return n;
}

void get_shading_material(const IntersectionInfo info,
                          const vec3 ray_origin,
                          const vec3 ray_dir,
                          out ShadingMaterial mat) {
    const VertexExtraData extra_data =  buf_ext[nonuniformEXT(info.instance_id)].v[info.primitive_index];
    const uint flags = extra_data.texnum_fb_flags >> 12;
    if (flags == MAT_FLAGS_SKY) {
        mat.albedo = vec4(0, 0, 0, 1);
        mat.emission = envmap(ray_dir);
        mat.normal = -ray_dir;
        mat.geo_normal = -ray_dir;
        mat.pos = ray_origin + T_MAX * ray_dir;
        mat.gloss = 0;
        return;
    }
    const vec2 st_0 = unpackHalf2x16(extra_data.st_0);
    const vec2 st_1 = unpackHalf2x16(extra_data.st_1);
    const vec2 st_2 = unpackHalf2x16(extra_data.st_2);

    vec2 st = st_0 * info.barycentrics.x
            + st_1 * info.barycentrics.y
            + st_2 * info.barycentrics.z;

    if (flags > 0 && flags < 6) {
        warp(st);
    }

    {
        // Load albedo
        // const uint texnum = extra_data.texnum_alpha & 0xfff;
        mat.albedo = texture(img_tex[nonuniformEXT(extra_data.texnum_alpha & 0xfff)], st);
        const uint alpha = extra_data.texnum_alpha >> 12;
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
        const uint texnum_gloss = extra_data.n0_gloss_norm & 0xffff;
        const uint texnum_normal = extra_data.n0_gloss_norm >> 16;
        
        if (texnum_normal > 0) {
            // overwrite geo normal with normal map normal
            const vec3 tangent_normal = texture(img_tex[nonuniformEXT(texnum_normal)], st).rgb;
            mat.normal = apply_normalmap(verts[0], verts[1], verts[2], st_0, st_1, st_2, mat.normal, tangent_normal);
        }
        if (texnum_gloss > 0) {
            mat.gloss = texture(img_tex[nonuniformEXT(texnum_gloss)], st).r;
        }
    } else {
        const vec3 n0 = geo_decode_normal(extra_data.n0_gloss_norm);
        const vec3 n1 = geo_decode_normal(extra_data.n1_brush);
        const vec3 n2 = geo_decode_normal(extra_data.n2);
        mat.normal = normalize(mat3(n0, n1, n2) * info.barycentrics);
        mat.gloss = 0;
    }
}
