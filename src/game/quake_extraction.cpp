#include "quake_extraction.hpp"

#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/utils/xorshift.hpp"

#include <cmath>

extern "C" {
#include "quakedef.h"

extern particle_t* active_particles;
extern cvar_t scr_fov, cl_gun_fovscale;
}

namespace merian_quake {

std::vector<merian::float3> bake_alias_pose_normals(aliashdr_t* hdr) {
    const auto* indexes = (const int16_t*)((uint8_t*)hdr + hdr->indexes);
    const auto* desc = (const aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
    const auto* trivertexes = (const trivertx_t*)((uint8_t*)hdr + hdr->vertexes);
    const uint32_t numverts = static_cast<uint32_t>(hdr->numverts);
    const uint32_t numposes = static_cast<uint32_t>(hdr->numposes);
    const uint32_t prim_count = static_cast<uint32_t>(hdr->numindexes / 3);

    // Quake winds CW so the outward face normal is cross(v2-v0, v1-v0). Positions stay
    // in byte-coord space — the per-axis scale cancels in the shader's inv-transpose.
    std::vector<merian::float3> normals(static_cast<size_t>(numposes) * numverts,
                                        merian::float3(0.f));
    for (uint32_t pose = 0; pose < numposes; pose++) {
        merian::float3* pose_normals = normals.data() + pose * numverts;
        const trivertx_t* pose_verts = trivertexes + pose * numverts;
        for (uint32_t t = 0; t < prim_count; t++) {
            const int vi0 = desc[indexes[t * 3 + 0]].vertindex;
            const int vi1 = desc[indexes[t * 3 + 1]].vertindex;
            const int vi2 = desc[indexes[t * 3 + 2]].vertindex;
            const merian::float3 p0(pose_verts[vi0].v[0], pose_verts[vi0].v[1],
                                    pose_verts[vi0].v[2]);
            const merian::float3 p1(pose_verts[vi1].v[0], pose_verts[vi1].v[1],
                                    pose_verts[vi1].v[2]);
            const merian::float3 p2(pose_verts[vi2].v[0], pose_verts[vi2].v[1],
                                    pose_verts[vi2].v[2]);
            const merian::float3 face_n = merian::cross(p2 - p0, p1 - p0);
            pose_normals[vi0] += face_n;
            pose_normals[vi1] += face_n;
            pose_normals[vi2] += face_n;
        }
        for (uint32_t v = 0; v < numverts; v++) {
            const float len2 = merian::dot(pose_normals[v], pose_normals[v]);
            pose_normals[v] =
                (len2 > 0.f) ? pose_normals[v] / std::sqrt(len2) : merian::float3(0.f, 0.f, 1.f);
        }
    }
    return normals;
}

bool sprite_world_basis(entity_t* ent,
                        msprite_t* psprite,
                        merian::float3& s_up,
                        merian::float3& s_right) {
    if (psprite == nullptr)
        return false;

    merian::float3 vpn, vright, vup, r_origin;
    VectorCopy(r_refdef.vieworg, r_origin);
    AngleVectors(r_refdef.viewangles, &vpn.x, &vright.x, &vup.x);

    merian::float3 v_forward;
    merian::float3 v_right;
    merian::float3 v_up;

    switch (psprite->type) {
    case SPR_VP_PARALLEL_UPRIGHT:
        v_up = merian::float3(0, 0, 1);
        v_right = merian::normalize(merian::cross(vpn, v_up));
        s_up = v_up;
        s_right = v_right;
        break;
    case SPR_FACING_UPRIGHT:
        VectorSubtract(ent->origin, &r_origin.x, &v_forward.x);
        v_forward.z = 0;
        VectorNormalizeFast(&v_forward.x);
        v_right = merian::float3(v_forward.y, -v_forward.x, 0);
        v_up = merian::float3(0, 0, 1);
        s_up = v_up;
        s_right = v_right;
        break;
    case SPR_VP_PARALLEL:
        s_up = vup;
        s_right = vright;
        break;
    case SPR_ORIENTED:
        AngleVectors(ent->angles, &v_forward.x, &v_right.x, &v_up.x);
        s_up = v_up;
        s_right = v_right;
        break;
    case SPR_VP_PARALLEL_ORIENTED: {
        const float angle = ent->angles[ROLL] * M_PI_DIV_180;
        const float sr = std::sin(angle);
        const float cr = std::cos(angle);
        v_right = (vright * cr) + (vup * sr);
        v_up = (vright * -sr) + (vup * cr);
        s_up = v_up;
        s_right = v_right;
        break;
    }
    default:
        return false;
    }
    s_up = merian::normalize(s_up);
    s_right = merian::normalize(s_right);
    return true;
}

void extract_particle_geo(std::vector<merian::PackedVertexData>& vertices,
                          std::vector<merian::float3>& prev_positions,
                          std::vector<merian::uint3>& indices,
                          const bool no_random,
                          const double prev_cl_time) {
    static const merian::float3 voff[4] = {
        {0.0f, 1.0f, 0.0f},
        {-0.5f, -0.5f, 0.87f},
        {-0.5f, -0.5f, -0.87f},
        {1.0f, -0.5f, 0.0f},
    };

    vec3_t vpn, vright, vup, r_origin;
    VectorCopy(r_refdef.vieworg, r_origin);
    AngleVectors(r_refdef.viewangles, vpn, vright, vup);

    for (particle_t* p = active_particles; p != nullptr; p = p->next) {
        float scale = (p->org[0] - r_origin[0]) * vpn[0] + (p->org[1] - r_origin[1]) * vpn[1] +
                      (p->org[2] - r_origin[2]) * vpn[2];
        if (scale < 20.f)
            scale = 1.08f;
        else
            scale = 1.f + scale * 0.004f;
        scale *= 0.5f;

        const uint32_t seed = no_random ? static_cast<uint32_t>(p->die)
                                        : static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
        merian::XORShift32 xrand{seed};

        const float velocity = merian::length(merian::as_float3(p->vel));
        const merian::float3 origin = merian::as_float3(p->org);
        const merian::float3 prev_origin = merian::as_float3(p->mv_prev_origin);

        const float particle_offset = static_cast<float>(2.0 * (xrand.next_double() - 0.5) +
                                                         2.0 * (xrand.next_double() - 0.5));
        const float rand_angle = static_cast<float>(xrand.next_double());
        const merian::float3 rand_v = merian::normalize(merian::float3(
            static_cast<float>(xrand.next_double()), static_cast<float>(xrand.next_double()),
            static_cast<float>(xrand.next_double())));

        const merian::float4x4 rot = merian::rotation(
            rand_v, (rand_angle + cl.time * 0.001f * velocity) * 2.f * static_cast<float>(M_PI));
        const merian::float4x4 prev_rot = merian::rotation(
            rand_v, (rand_angle + static_cast<float>(prev_cl_time) * 0.001f * velocity) * 2.f *
                        static_cast<float>(M_PI));

        merian::float3 vert[4];
        merian::float3 prev_vert[4];
        for (int k = 0; k < 4; k++) {
            const float vert_off = static_cast<float>(
                0.5 * ((xrand.next_double() - 0.5) + (xrand.next_double() - 0.5)));
            const float rand_scale = static_cast<float>(xrand.next_double());
            const merian::float4 corner(scale * voff[k] * (1.f + rand_scale) + vert_off, 1.f);
            vert[k] = origin + particle_offset + merian::mul(rot, corner).xyz();
            prev_vert[k] = prev_origin + particle_offset + merian::mul(prev_rot, corner).xyz();
        }
        VectorCopy(p->org, p->mv_prev_origin);

        // One tetrahedron per particle. uv.x is the palette index in [0,1] so the
        // particle material samples the diffuse / emission palette by uv.
        const uint32_t base = static_cast<uint32_t>(vertices.size());
        const float palette_uv =
            (static_cast<float>(static_cast<int>(p->color) & 0xff) + 0.5f) / 256.f;
        for (int k = 0; k < 4; k++) {
            merian::PackedVertexData pv{};
            pv.position = vert[k];
            pv.uv = merian::half2(palette_uv, 0.f);
            pv.encoded_tangent = 0;
            // Radial outward from the centroid — close enough to the averaged
            // face normal for billboard-style shading.
            const merian::float3 centroid = 0.25f * (vert[0] + vert[1] + vert[2] + vert[3]);
            pv.encoded_normal = merian::encode_normal(merian::normalize(vert[k] - centroid));
            vertices.push_back(pv);
            prev_positions.push_back(prev_vert[k]);
        }

        indices.push_back(merian::uint3(base + 0, base + 1, base + 2));
        indices.push_back(merian::uint3(base + 0, base + 2, base + 3));
        indices.push_back(merian::uint3(base + 0, base + 3, base + 1));
        indices.push_back(merian::uint3(base + 1, base + 3, base + 2));
    }
}

void lerp_alias_vertices(aliashdr_t* hdr,
                         const merian::float3* baked_normals,
                         const int pose1,
                         const int pose2,
                         const float blend,
                         const int prev_pose1,
                         const int prev_pose2,
                         const float prev_blend,
                         merian::PackedVertexData* vertices_dst,
                         merian::PackedPrevVertexData* prev_dst) {
    const auto* desc = (aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
    const auto* trivertexes = (trivertx_t*)((uint8_t*)hdr + hdr->vertexes);

    const float skin_w = static_cast<float>(hdr->skinwidth);
    const float skin_h = static_cast<float>(hdr->skinheight);
    const merian::float3* normals_pose1 = baked_normals + hdr->numverts * pose1;
    const merian::float3* normals_pose2 = baked_normals + hdr->numverts * pose2;

    for (int v = 0; v < hdr->numverts_vbo; v++) {
        const int vi = desc[v].vertindex;

        const auto& tv1 = trivertexes[hdr->numverts * pose1 + vi];
        const auto& tv2 = trivertexes[hdr->numverts * pose2 + vi];
        const merian::float3 p1{float(tv1.v[0]), float(tv1.v[1]), float(tv1.v[2])};
        const merian::float3 p2{float(tv2.v[0]), float(tv2.v[1]), float(tv2.v[2])};

        vertices_dst[v].position = merian::lerp(p1, p2, blend);
        vertices_dst[v].encoded_normal = merian::encode_normal(
            merian::normalize(merian::lerp(normals_pose1[vi], normals_pose2[vi], blend)));
        vertices_dst[v].uv =
            merian::half2((desc[v].st[0] + 0.5f) / skin_w, (desc[v].st[1] + 0.5f) / skin_h);
        vertices_dst[v].encoded_tangent = 0;

        const auto& ptv1 = trivertexes[hdr->numverts * prev_pose1 + vi];
        const auto& ptv2 = trivertexes[hdr->numverts * prev_pose2 + vi];
        const merian::float3 pp1{float(ptv1.v[0]), float(ptv1.v[1]), float(ptv1.v[2])};
        const merian::float3 pp2{float(ptv2.v[0]), float(ptv2.v[1]), float(ptv2.v[2])};
        prev_dst[v].position = merian::lerp(pp1, pp2, prev_blend);
    }
}

} // namespace merian_quake
