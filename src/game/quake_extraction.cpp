#include "quake_extraction.hpp"

#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/utils/xorshift.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <mutex>

extern "C" {
#include "quakedef.h"

extern particle_t* active_particles;
extern cvar_t scr_fov, cl_gun_fovscale;
}

namespace merian_quake {

namespace {

// Triangulate a Quake glpoly_t (n>=3 vertices) as a fan into `indices`,
// emitting positions/normals/uvs into `vertices`. `enc_n` is the (already
// encoded) plane normal in world space. `mat_prev_model` transforms the
// same model-space verts into their previous-frame world position.
void emit_brush_poly(const glpoly_t* p,
                     const merian::float4x4& mat_model,
                     const merian::float4x4& mat_prev_model,
                     const uint32_t enc_n,
                     std::vector<merian::PackedVertexData>& vertices,
                     std::vector<merian::float3>& prev_positions,
                     std::vector<merian::uint3>& indices) {
    if (p->numverts < 3)
        return;
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    for (int v = 0; v < p->numverts; v++) {
        const merian::float4 mv(merian::as_float3(p->verts[v]), 1.f);
        merian::PackedVertexData pv{};
        pv.position = merian::mul(mv, mat_model).xyz();
        pv.encoded_normal = enc_n;
        pv.uv = merian::half2(p->verts[v][3], p->verts[v][4]);
        pv.encoded_tangent = 0;
        vertices.push_back(pv);
        prev_positions.push_back(merian::mul(mv, mat_prev_model).xyz());
    }
    for (int v = 2; v < p->numverts; v++) {
        indices.push_back(merian::uint3(base, base + uint32_t(v) - 1u, base + uint32_t(v)));
    }
}

merian::float4x4 entity_world_transform(const float* origin, const float* angles) {
    std::array<float, 3> a = {-angles[0], angles[1], angles[2]};
    merian::float4x4 m = merian::identity();
    AngleVectors(a.data(), &m[0].x, &m[1].x, &m[2].x);
    m[1] *= -1;
    m[3] = merian::float4(origin[0], origin[1], origin[2], 1.f);
    return m;
}

} // namespace

void extract_alias_geo(entity_t* ent,
                       std::vector<merian::PackedVertexData>& vertices,
                       std::vector<merian::float3>& prev_positions,
                       std::vector<merian::uint3>& indices) {
    qmodel_t* m = ent->model;
    if (m == nullptr || m->type != mod_alias)
        return;

    // Mod_Extradata may rebind the cached header internally; serialize.
    static std::mutex quake_mutex;
    std::lock_guard<std::mutex> lock(quake_mutex);

    aliashdr_t* hdr = (aliashdr_t*)Mod_Extradata(ent->model);
    aliasmesh_t* desc = (aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
    int16_t* indexes = (int16_t*)((uint8_t*)hdr + hdr->indexes);
    trivertx_t* trivertexes = (trivertx_t*)((uint8_t*)hdr + hdr->vertexes);

    int f = ent->frame;
    if (f < 0 || f >= hdr->numposes)
        return;

    // Make the player gun FOV-independent (mirrors r_alias.c gun handling).
    merian::float3 fovscale(1.f);
    if (ent == &cl.viewent && scr_fov.value > 90.f && cl_gun_fovscale.value != 0.f) {
        const float t = std::tan(scr_fov.value * static_cast<float>(0.5 * M_PI / 180.0));
        fovscale.y = t;
        fovscale.z = t;
    }

    const merian::float4x4 scale = merian::transpose(
        merian::mul(merian::translation(merian::as_float3(hdr->scale_origin) * fovscale),
                    merian::scale(merian::as_float3(hdr->scale) * fovscale)));

    // Previous-frame model matrix from cached mv_prev_origin/angles. The
    // axis flip mirrors the current-frame logic below.
    std::array<float, 3> prev_angles = {-ent->mv_prev_angles[0], ent->mv_prev_angles[1],
                                        ent->mv_prev_angles[2]};
    merian::float4x4 mat_prev_model = merian::identity();
    AngleVectors(prev_angles.data(), &mat_prev_model[0].x, &mat_prev_model[1].x,
                 &mat_prev_model[2].x);
    mat_prev_model[3] = merian::float4(merian::as_float3(ent->mv_prev_origin), 1.f);
    mat_prev_model[1] *= -1;
    mat_prev_model = merian::mul(scale, mat_prev_model);
    const float prev_blend = ent->mv_prev_blend;

    lerpdata_t lerpdata;
    R_SetupAliasFrame(ent, hdr, ent->frame, &lerpdata);
    R_SetupEntityTransform(ent, &lerpdata);

    // Match the legacy axis flip convention.
    lerpdata.angles[0] *= -1;

    merian::float4x4 mat_model = merian::identity();
    AngleVectors(lerpdata.angles, &mat_model[0].x, &mat_model[1].x, &mat_model[2].x);
    mat_model[3] = merian::float4(merian::as_float3(lerpdata.origin), 1.f);
    mat_model[1] *= -1;
    mat_model = merian::mul(scale, mat_model);

    const merian::float3x3 mat_model_inv_t =
        merian::float3x3(merian::transpose(merian::inverse(mat_model)));

    const float skin_w = static_cast<float>(hdr->skinwidth);
    const float skin_h = static_cast<float>(hdr->skinheight);
    const uint32_t base = static_cast<uint32_t>(vertices.size());

    for (int v = 0; v < hdr->numverts_vbo; v++) {
        const int i_pose1 = hdr->numverts * lerpdata.pose1 + desc[v].vertindex;
        const int i_pose2 = hdr->numverts * lerpdata.pose2 + desc[v].vertindex;

        merian::float3 p1{static_cast<float>(trivertexes[i_pose1].v[0]),
                          static_cast<float>(trivertexes[i_pose1].v[1]),
                          static_cast<float>(trivertexes[i_pose1].v[2])};
        merian::float3 p2{static_cast<float>(trivertexes[i_pose2].v[0]),
                          static_cast<float>(trivertexes[i_pose2].v[1]),
                          static_cast<float>(trivertexes[i_pose2].v[2])};

        const merian::float3 world_pos =
            merian::mul(merian::float4(merian::lerp(p1, p2, lerpdata.blend), 1.f), mat_model).xyz();
        // Previous-frame world position: same model-space pose1/pose2, but
        // blended with last frame's blend factor and transformed by the
        // previous model matrix. This matches the legacy motion-vector
        // computation in quake_helpers.cpp.
        const merian::float3 prev_world_pos =
            merian::mul(merian::float4(merian::lerp(p1, p2, prev_blend), 1.f), mat_prev_model)
                .xyz();

        const merian::float3 n1 =
            merian::as_float3(r_avertexnormals[trivertexes[i_pose1].lightnormalindex]);
        const merian::float3 n2 =
            merian::as_float3(r_avertexnormals[trivertexes[i_pose2].lightnormalindex]);
        const merian::float3 world_n =
            merian::normalize(merian::mul(merian::lerp(n1, n2, lerpdata.blend), mat_model_inv_t));

        merian::PackedVertexData pv{};
        pv.position = world_pos;
        pv.encoded_normal = merian::encode_normal(world_n);
        pv.uv = merian::half2((desc[v].st[0] + 0.5f) / skin_w, (desc[v].st[1] + 0.5f) / skin_h);
        pv.encoded_tangent = 0;
        vertices.push_back(pv);
        prev_positions.push_back(prev_world_pos);
    }

    // Bookkeeping: keep prev-frame state up to date for future motion vectors.
    ent->mv_prev_blend = lerpdata.blend;
    VectorCopy(lerpdata.angles, ent->mv_prev_angles);
    VectorCopy(lerpdata.origin, ent->mv_prev_origin);

    for (int i = 0; i + 2 < hdr->numindexes; i += 3) {
        indices.push_back(merian::uint3(base + static_cast<uint32_t>(indexes[i + 0]),
                                        base + static_cast<uint32_t>(indexes[i + 1]),
                                        base + static_cast<uint32_t>(indexes[i + 2])));
    }
}

void extract_brush_entity_geo(entity_t* ent,
                              std::vector<merian::PackedVertexData>& vertices,
                              std::vector<merian::float3>& prev_positions,
                              std::vector<merian::uint3>& indices) {
    qmodel_t* m = ent->model;
    if (m == nullptr || m->type != mod_brush)
        return;

    const merian::float4x4 mat_model = entity_world_transform(ent->origin, ent->angles);
    const merian::float4x4 mat_prev_model =
        entity_world_transform(ent->mv_prev_origin, ent->mv_prev_angles);

    for (int i = 0; i < m->nummodelsurfaces; i++) {
        msurface_t* surf = &m->surfaces[m->firstmodelsurface + i];
        if (surf->texinfo == nullptr || surf->texinfo->texture == nullptr)
            continue;
        if (strcmp(surf->texinfo->texture->name, "skip") == 0)
            continue;

        merian::float3 plane_n = merian::as_float3(surf->plane->normal);
        if ((surf->flags & SURF_PLANEBACK) != 0)
            plane_n = -plane_n;
        // Transform plane normal to world space (no scale on entities, so the
        // upper 3x3 of mat_model is orthonormal — direct multiply is fine).
        plane_n = merian::mul(merian::float3x3(mat_model), plane_n);
        const uint32_t enc_n = merian::encode_normal(merian::normalize(plane_n));

        for (glpoly_t* p = surf->polys; p != nullptr; p = nullptr) {
            emit_brush_poly(p, mat_model, mat_prev_model, enc_n, vertices, prev_positions,
                            indices);
        }
    }

    VectorCopy(ent->origin, ent->mv_prev_origin);
    VectorCopy(ent->angles, ent->mv_prev_angles);
}

void extract_sprite_geo(entity_t* ent,
                        std::vector<merian::PackedVertexData>& vertices,
                        std::vector<merian::float3>& prev_positions,
                        std::vector<merian::uint3>& indices) {
    qmodel_t* m = ent->model;
    if (m == nullptr || m->type != mod_sprite)
        return;

    mspriteframe_t* frame = R_GetSpriteFrame(ent);
    if (frame == nullptr || frame->gltexture == nullptr)
        return;
    msprite_t* psprite = (msprite_t*)ent->model->cache.data;

    const float scale = ENTSCALE_DECODE(ent->scale);

    merian::float3 vpn, vright, vup, r_origin;
    VectorCopy(r_refdef.vieworg, r_origin);
    AngleVectors(r_refdef.viewangles, &vpn.x, &vright.x, &vup.x);

    merian::float3 v_forward;
    merian::float3 v_right;
    merian::float3 v_up;
    merian::float3 s_up;
    merian::float3 s_right;

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
        return;
    }
    s_up = merian::normalize(s_up);
    s_right = merian::normalize(s_right);

    // Two cross-axis quads (k=0 and k=1).
    for (int k = 0; k < 2; k++) {
        merian::float3 v0, v1, v2, v3;
        if (k == 0) {
            v0 = scale * (frame->down * s_up + frame->left * s_right);
            v1 = scale * (frame->up * s_up + frame->left * s_right);
            v2 = scale * (frame->up * s_up + frame->right * s_right);
            v3 = scale * (frame->down * s_up + frame->right * s_right);
        } else {
            v0 = scale * (frame->down * s_up - frame->left * s_right);
            v1 = scale * (frame->up * s_up - frame->left * s_right);
            v2 = scale * (frame->up * s_up - frame->right * s_right);
            v3 = scale * (frame->down * s_up - frame->right * s_right);
        }

        const merian::float3 origin = merian::as_float3(ent->origin);
        const merian::float3 prev_origin = merian::as_float3(ent->mv_prev_origin);
        const merian::float3 e0 = v2 - v0;
        const merian::float3 e1 = v1 - v0;
        const uint32_t enc_n = merian::encode_normal(merian::normalize(merian::cross(e0, e1)));

        const float smax = frame->smax;
        const float tmax = frame->tmax;

        const uint32_t base = static_cast<uint32_t>(vertices.size());
        auto push = [&](const merian::float3& p, float s, float t) {
            merian::PackedVertexData pv{};
            pv.position = p + origin;
            pv.encoded_normal = enc_n;
            pv.uv = merian::half2(s, t);
            pv.encoded_tangent = 0;
            vertices.push_back(pv);
            // Sprites only translate; reuse the local-space corner at the
            // previous origin so motion vectors track entity movement.
            prev_positions.push_back(p + prev_origin);
        };
        push(v0, 0.f, tmax);
        push(v1, 0.f, 0.f);
        push(v2, smax, 0.f);
        push(v3, smax, tmax);

        indices.push_back(merian::uint3(base + 0, base + 1, base + 2));
        indices.push_back(merian::uint3(base + 0, base + 2, base + 3));
    }

    VectorCopy(ent->origin, ent->mv_prev_origin);
}

void extract_entity_geo(entity_t* ent,
                        std::vector<merian::PackedVertexData>& vertices,
                        std::vector<merian::float3>& prev_positions,
                        std::vector<merian::uint3>& indices) {
    if (ent == nullptr || ent->model == nullptr)
        return;
    switch (ent->model->type) {
    case mod_alias:
        extract_alias_geo(ent, vertices, prev_positions, indices);
        break;
    case mod_brush:
        extract_brush_entity_geo(ent, vertices, prev_positions, indices);
        break;
    case mod_sprite:
        extract_sprite_geo(ent, vertices, prev_positions, indices);
        break;
    default:
        break;
    }
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

        merian::float3 vert[4];
        merian::float3 prev_vert[4];
        for (int l = 0; l < 3; l++) {
            const float particle_offset = static_cast<float>(2.0 * (xrand.next_double() - 0.5) +
                                                             2.0 * (xrand.next_double() - 0.5));
            const float rand_angle = static_cast<float>(xrand.next_double());
            const merian::float3 rand_v = merian::normalize(merian::float3(
                static_cast<float>(xrand.next_double()), static_cast<float>(xrand.next_double()),
                static_cast<float>(xrand.next_double())));

            const merian::float4x4 rot =
                merian::rotation(rand_v, (rand_angle + cl.time * 0.001f * velocity) * 2.f *
                                             static_cast<float>(M_PI));
            const merian::float4x4 prev_rot = merian::rotation(
                rand_v, (rand_angle + static_cast<float>(prev_cl_time) * 0.001f * velocity) * 2.f *
                            static_cast<float>(M_PI));
            for (int k = 0; k < 4; k++) {
                const float vert_off = static_cast<float>(
                    0.5 * ((xrand.next_double() - 0.5) + (xrand.next_double() - 0.5)));
                const float rand_scale = static_cast<float>(xrand.next_double());
                const merian::float4 corner(scale * voff[k] * (1.f + rand_scale) + vert_off, 1.f);
                vert[k] = origin + particle_offset + merian::mul(rot, corner).xyz();
                prev_vert[k] =
                    prev_origin + particle_offset + merian::mul(prev_rot, corner).xyz();
            }
        }
        VectorCopy(p->org, p->mv_prev_origin);

        // Build a tetrahedron (4 triangles) per particle and emit one
        // shared face-normal per vertex (encoded from the tet centroid).
        const uint32_t base = static_cast<uint32_t>(vertices.size());
        for (int k = 0; k < 4; k++) {
            merian::PackedVertexData pv{};
            pv.position = vert[k];
            pv.uv = merian::half2(0.f, 0.f);
            pv.encoded_tangent = 0;
            // Per-vertex normal: take the average of the three faces meeting
            // here — for a regular-ish tet that's roughly the radial outward
            // direction from the centroid, which is good enough for the
            // billboard-style shading the renderer does.
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

} // namespace merian_quake
