// Checks flags and alpha to determine if the intersection should be confirmed.
bool confirm_intersection(const IntersectionInfo info) {
    const VertexExtraData extra_data =  buf_ext[nonuniformEXT(info.instance_id)].v[info.primitive_index];
    const uint16_t alpha = extra_data.texnum_alpha >> 12;
    const uint16_t flags = extra_data.texnum_fb_flags >> 12;
    if (flags != MAT_FLAGS_NONE && flags != MAT_FLAGS_SPRITE) {
        // treat sky, lava, slime,... not transparent for now
        return true;
    } else if (alpha != 0) { // 0 means use texture
        return decode_alpha(alpha) >= ALPHA_THRESHOLD;
    } else {
        const vec2 st = extra_data.st_0 * info.barycentrics.x
                      + extra_data.st_1 * info.barycentrics.y
                      + extra_data.st_2 * info.barycentrics.z;

        // We covered the flags above, this surface cannot warp
        // if (flags > 0 && flags < 6) {
        //     warp(st);
        // }

        // const uint texnum = extra_data.texnum_alpha & 0xfff;
        return textureGather(img_tex[nonuniformEXT(min(extra_data.texnum_alpha & 0xfff, MAX_GLTEXTURES - 1))], st, 3).r >= ALPHA_THRESHOLD;
    }
}

// Sets t < 0 if nothing was hit
void trace_ray(const vec3 x0, const vec3 dir, out IntersectionInfo info) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query,
                          tlas,
                          gl_RayFlagsCullBackFacingTrianglesEXT,    // Ray flags, None means: If not set in the instance the triangle
                                                                    // may not be opaque. This is a major performance hit
                                                                    // since we need to load the texture to determine if we want to trace
                                                                    // further.
                                                                    // Use gl_RayFlagsOpaqueEXT to treat all geometry as opaque
                          0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                          x0,
                          0,                     // Minimum t-value (we set it here to 0 and pull back the ray)
                                                 // such that the ray cannot escape in corners.
                          dir,
                          T_MAX);                // Maximum t-value

    while(rayQueryProceedEXT(ray_query)) {
        // We are only interessted in triangles. "false" means candidates too, not only commited
        if (rayQueryGetIntersectionTypeEXT(ray_query, false) != gl_RayQueryCandidateIntersectionTriangleEXT)
            continue;

        get_intersection_info_uncommited(ray_query, info);
        if (confirm_intersection(info))
            rayQueryConfirmIntersectionEXT(ray_query);
    }

    if(rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        get_intersection_info_commited(ray_query, info);
    else
        info.t = -1;
}
