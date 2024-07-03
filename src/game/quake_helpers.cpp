#include "quake_helpers.hpp"

#include "glm/ext/matrix_transform.hpp"
#include "merian/utils/bitpacking.hpp"
#include "merian/utils/glm.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/xorshift.hpp"
#include "config.h"

#include <algorithm>
#include <array>
#include <glm/glm.hpp>
#include <mutex>
#include <vector>

extern "C" {
#include "quakedef.h"
#include "screen.h"

// from r_part.c
extern particle_t* active_particles;

extern cvar_t scr_fov, cl_gun_fovscale;
}

uint16_t make_texnum_alpha(gltexture_s* tex, entity_t* entity, msurface_t* surface) {
    uint16_t result = std::min(tex->texnum, (uint32_t)MAX_GLTEXTURES - 1);
    uint32_t alpha;
    if (entity) {
        // uint32_t alpha = CLAMP(0, (ent->alpha - 1.0) / 254.0 * 15, 15); // alpha in 4 bits
        // if (!TEXPREF_ALPHA)
        //     alpha = 15;
        // TODO: 0 means default, 1 means invisible, 255 is opaque, 2--254 is
        // really applicable
        // TODO: default means  map_lavaalpha > 0 ? map_lavaalpha :
        // map_wateralpha
        // TODO: or "slime" or "tele" instead of "lava"
    }
    if (tex->flags & TEXPREF_ALPHA) {
        // use texture alpha
        alpha = 0;
    } else {
        // fully opaque
        alpha = 15;
    }
    result |= alpha << 12;
    return result;
}

void add_particles(std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<VertexExtraData>& ext,
                   const uint32_t texnum_blood,
                   const uint32_t texnum_explosion,
                   const bool no_random,
                   const double prev_cl_time) {

    static const glm::vec3 voff[4] = {
        {0.0, 1.0, 0.0},
        {-0.5, -0.5, 0.87},
        {-0.5, -0.5, -0.87},
        {1.0, -0.5, 0.0},
    };

    vec3_t vpn, vright, vup, r_origin;
    VectorCopy(r_refdef.vieworg, r_origin);
    AngleVectors(r_refdef.viewangles, vpn, vright, vup);

    for (particle_t* p = active_particles; p; p = p->next) {
        // from r_part.c
        float scale = (p->org[0] - r_origin[0]) * vpn[0] + (p->org[1] - r_origin[1]) * vpn[1] +
                      (p->org[2] - r_origin[2]) * vpn[2];
        if (scale < 20)
            scale = 1 + 0.08;
        else
            scale = 1 + scale * 0.004;

        scale *= 0.5;

        uint32_t c = d_8to24table[(int)p->color];
        uint32_t seed = no_random ? static_cast<uint32_t>(p->die)
                                  : static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
        merian::XORShift32 xrand{seed};

        // Some heuristics to improve blood, fire, explosions
        uint32_t texnum = 0;
        uint32_t texnum_fb = 0;
        uint8_t* color_bytes = (uint8_t*)&c;

        if (color_bytes[1] == 0 && color_bytes[2] == 0 && color_bytes[0] > 10) {
            texnum = texnum_blood;
        } else if (p->type == pt_explode2) {
            texnum = texnum_explosion;
            texnum_fb = texnum_explosion;
            scale *= 2.0;
        } else if (p->type == pt_fire &&
                   (color_bytes[0] != color_bytes[1] || color_bytes[1] != color_bytes[2] ||
                    color_bytes[0] != color_bytes[2])) {
            texnum = texnum_explosion;
            texnum_fb = texnum_explosion;
            scale *= 2.0;
        } else if (0.299 * color_bytes[0] + 0.587 * color_bytes[1] + 0.114 * color_bytes[2] > 200) {
            // very bright colors are probably fire
            texnum = texnum_explosion;
            texnum_fb = texnum_explosion;
            scale *= 2.0;
        }

        glm::vec3 vert[4];
        glm::vec3 prev_vert[4];
        for (int l = 0; l < 3; l++) {
            const float particle_offset = 2 * (xrand.get() - 0.5) + 2 * (xrand.get() - 0.5);
            const double rand_angle = xrand.get();
            const glm::vec3 rand_v =
                glm::normalize(glm::vec3(xrand.get(), xrand.get(), xrand.get()));

            const glm::mat4 rotation = glm::rotate<float>(
                glm::identity<glm::mat4>(),
                (rand_angle + cl.time * 0.001 * glm::length(*merian::as_vec3(p->vel))) * 2 * M_PI,
                rand_v);
            const glm::mat4 prev_rotation = glm::rotate<float>(
                glm::identity<glm::mat4>(),
                (rand_angle + prev_cl_time * 0.001 * glm::length(*merian::as_vec3(p->vel))) * 2 *
                    M_PI,
                rand_v);
            for (int k = 0; k < 4; k++) {
                const float vertex_offset = 0.5 * ((xrand.get() - 0.5) + (xrand.get() - 0.5));
                const float rand_offset_scale = (float)xrand.get();

                vert[k] = *merian::as_vec3(p->org) + particle_offset +
                          glm::vec3(rotation * glm::vec4(scale * voff[k] * (1 + rand_offset_scale) +
                                                             vertex_offset,
                                                         1));
                prev_vert[k] =
                    *merian::as_vec3(p->mv_prev_origin) + particle_offset +
                    glm::vec3(
                        prev_rotation *
                        glm::vec4(scale * voff[k] * (1 + rand_offset_scale) + vertex_offset, 1));
            }
        }

        VectorCopy(p->org, p->mv_prev_origin);

        const uint32_t vtx_cnt = vtx.size() / 3;
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(vert[0][l]);
            prev_vtx.emplace_back(prev_vert[0][l]);
        }
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(vert[1][l]);
            prev_vtx.emplace_back(prev_vert[1][l]);
        }
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(vert[2][l]);
            prev_vtx.emplace_back(prev_vert[2][l]);
        }
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(vert[3][l]);
            prev_vtx.emplace_back(prev_vert[3][l]);
        }

        const uint32_t idx_size = idx.size();

        idx.emplace_back(vtx_cnt);
        idx.emplace_back(vtx_cnt + 1);
        idx.emplace_back(vtx_cnt + 2);

        idx.emplace_back(vtx_cnt);
        idx.emplace_back(vtx_cnt + 2);
        idx.emplace_back(vtx_cnt + 3);

        idx.emplace_back(vtx_cnt);
        idx.emplace_back(vtx_cnt + 3);
        idx.emplace_back(vtx_cnt + 1);

        idx.emplace_back(vtx_cnt + 1);
        idx.emplace_back(vtx_cnt + 3);
        idx.emplace_back(vtx_cnt + 2);

        // regular particle
        for (int k = 0; k < 4; k++) {
            if (texnum) {
                // texture patch
                const glm::vec3 n = glm::normalize(
                    glm::cross(*merian::as_vec3(&vtx[3 * idx[idx_size + 3 * k + 2]]) -
                                   *merian::as_vec3(&vtx[3 * idx[idx_size + 3 * k]]),
                               *merian::as_vec3(&vtx[3 * idx[idx_size + 3 * k + 1]]) -
                                   *merian::as_vec3(&vtx[3 * idx[idx_size + 3 * k]])));
                const uint32_t enc_n = merian::encode_normal(n);

                ext.emplace_back(texnum, texnum_fb, enc_n, enc_n, enc_n,
                                 merian::float_to_half(0), merian::float_to_half(1),
                                 merian::float_to_half(0), merian::float_to_half(0),
                                 merian::float_to_half(1), merian::float_to_half(0));
            } else {
                for (int i = 0; i < 3; i++)
                    color_bytes[0] =
                        std::clamp(color_bytes[0] * (1 + xrand.get() * 0.1 - 0.05), 0., 255.);

                uint32_t c_fb = 0;
                if (0.299 * color_bytes[0] + 0.587 * color_bytes[1] + 0.114 * color_bytes[2] >
                    150) {
                    c_fb = c; // bright colors are probably emitting
                }

                ext.emplace_back(0, 0 | (MAT_FLAGS_SOLID << 12), c, c_fb, 0,
                                 merian::float_to_half(0), merian::float_to_half(1),
                                 merian::float_to_half(0), merian::float_to_half(0),
                                 merian::float_to_half(1), merian::float_to_half(0));
            }
        }
    }
}

void add_geo_alias(entity_t* ent,
                   [[maybe_unused]] qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<VertexExtraData>& ext) {
    assert(m->type == mod_alias);

    static std::mutex quake_mutex;
    // An internal cache shifts the hdr pointer... :/
    std::lock_guard<std::mutex> lock(quake_mutex);

    // TODO: e->model->flags & MF_HOLEY <= enable alpha test
    // fprintf(stderr, "alias origin and angles %g %g %g -- %g %g %g\n",
    //     ent->origin[0], ent->origin[1], ent->origin[2],
    //     ent->angles[0], ent->angles[1], ent->angles[2]);

    aliashdr_t* hdr = (aliashdr_t*)Mod_Extradata(ent->model);
    aliasmesh_t* desc = (aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
    // the plural here really hurts but it's from quakespasm code:
    int16_t* indexes = (int16_t*)((uint8_t*)hdr + hdr->indexes);
    trivertx_t* trivertexes = (trivertx_t*)((uint8_t*)hdr + hdr->vertexes);

    // for(int f = 0; f < hdr->numposes; f++)
    // TODO: upload all vertices so we can just alter the indices on gpu
    int f = ent->frame;
    if (f < 0 || f >= hdr->numposes)
        return;

    // makes gun fov independent
    glm::vec3 fovscale(1.);
    if (ent == &cl.viewent && scr_fov.value > 90.f && cl_gun_fovscale.value)
        fovscale.y = fovscale.z = tan(scr_fov.value * (0.5f * M_PI / 180.f));

    glm::mat4 mat_prev_model = glm::identity<glm::mat4>();
    AngleVectors(ent->mv_prev_angles, &mat_prev_model[0].x, &mat_prev_model[1].x,
                 &mat_prev_model[2].x);
    mat_prev_model[3] = glm::vec4(*merian::as_vec3(ent->mv_prev_origin), 1);
    mat_prev_model[1] *= -1;

    // * ENTSCALE_DECODE(ent->scale)?
    mat_prev_model =
        mat_prev_model *
        glm::translate(glm::identity<glm::mat4>(), *merian::as_vec3(hdr->scale_origin) * fovscale);
    mat_prev_model = mat_prev_model * glm::scale(glm::identity<glm::mat4>(),
                                                 *merian::as_vec3(hdr->scale) * fovscale);

    lerpdata_t lerpdata;
    R_SetupAliasFrame(ent, hdr, ent->frame, &lerpdata);
    R_SetupEntityTransform(ent, &lerpdata);

    // angles: pitch yaw roll. axes: right fwd up
    lerpdata.angles[0] *= -1;
    glm::mat4 mat_model = glm::identity<glm::mat4>();
    glm::vec3 pos_pose1, pos_pose2;

    AngleVectors(lerpdata.angles, &mat_model[0].x, &mat_model[1].x, &mat_model[2].x);
    mat_model[3] = glm::vec4(*merian::as_vec3(lerpdata.origin), 1);
    mat_model[1] *= -1;

    // * ENTSCALE_DECODE(ent->scale)?
    mat_model = mat_model * glm::translate(glm::identity<glm::mat4>(),
                                           *merian::as_vec3(hdr->scale_origin) * fovscale);
    mat_model =
        mat_model * glm::scale(glm::identity<glm::mat4>(), *merian::as_vec3(hdr->scale) * fovscale);

    const glm::mat3 mat_model_inv_t = glm::transpose(glm::inverse(mat_model));

    uint32_t vtx_cnt = vtx.size() / 3;
    for (int v = 0; v < hdr->numverts_vbo; v++) {
        int i_pose1 = hdr->numverts * lerpdata.pose1 + desc[v].vertindex;
        int i_pose2 = hdr->numverts * lerpdata.pose2 + desc[v].vertindex;
        // get model pos
        for (int k = 0; k < 3; k++) {
            pos_pose1[k] = trivertexes[i_pose1].v[k];
            pos_pose2[k] = trivertexes[i_pose2].v[k];
        }
        // convert to world space
        const glm::vec3 world_pos =
            mat_model * glm::vec4(glm::mix(pos_pose1, pos_pose2, lerpdata.blend), 1.0);
        for (int k = 0; k < 3; k++)
            vtx.emplace_back(world_pos[k]);

        const glm::vec3 old_world_pos =
            mat_prev_model * glm::vec4(glm::mix(pos_pose1, pos_pose2, ent->mv_prev_blend), 1.0);
        for (int k = 0; k < 3; k++)
            prev_vtx.emplace_back(old_world_pos[k]);
    }

    ent->mv_prev_blend = lerpdata.blend;
    VectorCopy(lerpdata.angles, ent->mv_prev_angles);
    VectorCopy(lerpdata.origin, ent->mv_prev_origin);

    for (int i = 0; i < hdr->numindexes; i++)
        idx.emplace_back(vtx_cnt + indexes[i]);

    // normals for each vertex from above
    std::vector<uint32_t> tmpn(hdr->numverts_vbo);
    for (int v = 0; v < hdr->numverts_vbo; v++) {
        int i_pose1 = hdr->numverts * lerpdata.pose1 + desc[v].vertindex;
        int i_pose2 = hdr->numverts * lerpdata.pose2 + desc[v].vertindex;
        const glm::vec3* n_pose1 =
            merian::as_vec3(r_avertexnormals[trivertexes[i_pose1].lightnormalindex]);
        const glm::vec3* n_pose2 =
            merian::as_vec3(r_avertexnormals[trivertexes[i_pose2].lightnormalindex]);
        // convert to worldspace
        const glm::vec3 world_n =
            glm::normalize(mat_model_inv_t * glm::mix(*n_pose1, *n_pose2, lerpdata.blend));
        tmpn[v] = merian::encode_normal(world_n);
    }

    // add extra data for each primitive
    for (int i = 0; i < hdr->numindexes / 3; i++) {
        const int sk = glm::clamp(ent->skinnum, 0, hdr->numskins - 1),
                  fm = ((int)(cl.time * 10)) & 3;
        const uint16_t texnum_alpha = make_texnum_alpha(hdr->gltextures[sk][fm]);
        const uint16_t fb_texnum = hdr->fbtextures[sk][fm] ? hdr->fbtextures[sk][fm]->texnum : 0;

        uint32_t n0, n1, n2;
        if (hdr->nmtextures[sk][fm]) {
            // this discards the vertex normals
            n0 = merian::pack_uint32(hdr->gstextures[sk][fm] ? hdr->gstextures[sk][fm]->texnum : 0,
                                     hdr->nmtextures[sk][fm]->texnum);
            n1 = 0xffffffff; // mark as brush model -> to use normal map
            n2 = 0;
        } else {
            // the vertex normals
            n0 = tmpn[indexes[3 * i + 0]];
            n1 = tmpn[indexes[3 * i + 1]];
            n2 = tmpn[indexes[3 * i + 2]];
        }

        ext.emplace_back(texnum_alpha, fb_texnum, n0, n1, n2,
                         merian::float_to_half((desc[indexes[3 * i + 0]].st[0] + 0.5) /
                                                     (float)hdr->skinwidth),
                         merian::float_to_half((desc[indexes[3 * i + 0]].st[1] + 0.5) /
                                                     (float)hdr->skinheight),
                         merian::float_to_half((desc[indexes[3 * i + 1]].st[0] + 0.5) /
                                                     (float)hdr->skinwidth),
                         merian::float_to_half((desc[indexes[3 * i + 1]].st[1] + 0.5) /
                                                     (float)hdr->skinheight),
                         merian::float_to_half((desc[indexes[3 * i + 2]].st[0] + 0.5) /
                                                     (float)hdr->skinwidth),
                         merian::float_to_half((desc[indexes[3 * i + 2]].st[1] + 0.5) /
                                                     (float)hdr->skinheight));
    }
}

// geo_selector: 0 -> all, 1 -> opaque, 2 -> transparent
void add_geo_brush(entity_t* ent,
                   qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<VertexExtraData>& ext,
                   int geo_selector) {
    assert(m->type == mod_brush);

    std::array<float, 3> angles = {-ent->angles[0], ent->angles[1], ent->angles[2]};
    glm::mat4 mat_model = glm::identity<glm::mat4>();
    AngleVectors(angles.data(), &mat_model[0].x, &mat_model[1].x, &mat_model[2].x);
    mat_model[1] *= -1;
    VectorCopy(ent->origin, &mat_model[3].x);

    glm::mat4 mat_prev_model = glm::identity<glm::mat4>();
    std::array<float, 3> prev_angles = {-ent->mv_prev_angles[0], ent->mv_prev_angles[1],
                                        ent->mv_prev_angles[2]};
    AngleVectors(prev_angles.data(), &mat_prev_model[0].x, &mat_prev_model[1].x,
                 &mat_prev_model[2].x);
    mat_prev_model[1] *= -1;
    VectorCopy(ent->mv_prev_origin, &mat_prev_model[3].x);

    VectorCopy(ent->origin, ent->mv_prev_origin);
    VectorCopy(ent->angles, ent->mv_prev_angles);

    for (int i = 0; i < m->nummodelsurfaces; i++) {
        msurface_t* surf = &m->surfaces[m->firstmodelsurface + i];

        if (!strcmp(surf->texinfo->texture->name, "skip"))
            continue;

        // TODO: make somehow dynamic. don't want to re-upload the whole model just because
        // the texture animates. for now that means static brush models will not actually
        // animate their textures
        texture_t* t = R_TextureAnimation(surf->texinfo->texture, ent->frame);

        if (geo_selector == 1 && t->gltexture && (t->gltexture->flags & TEXPREF_ALPHA)) {
            continue;
        }
        if (geo_selector == 2 && (!t->gltexture || (t->gltexture->flags & TEXPREF_ALPHA) == 0)) {
            continue;
        }

        glpoly_t* p = surf->polys;
        while (p) {
            uint32_t vtx_cnt = vtx.size() / 3;
            for (int k = 0; k < p->numverts; k++) {
                const glm::vec3 coord = mat_model * glm::vec4(*merian::as_vec3(p->verts[k]), 1.0);
                const glm::vec3 prev_coord =
                    mat_prev_model * glm::vec4(*merian::as_vec3(p->verts[k]), 1.0);

                for (int l = 0; l < 3; l++) {
                    vtx.emplace_back(coord[l]);
                    prev_vtx.emplace_back(prev_coord[l]);
                }
            }

            for (int k = 2; k < p->numverts; k++) {
                idx.emplace_back(vtx_cnt);
                idx.emplace_back(vtx_cnt + k - 1);
                idx.emplace_back(vtx_cnt + k);
            }

            for (int k = 2; k < p->numverts; k++) {
                VertexExtraData extra{
                    .texnum_alpha = 0,
                    .texnum_fb_flags = 0,
                    .n0_gloss_norm = merian::pack_uint32(t->gloss ? t->gloss->texnum : 0,
                                                         t->norm ? t->norm->texnum : 0),
                    .n1_brush = 0xffffffff,
                    .n2 = 0,
                };
                uint32_t flags = MAT_FLAGS_NONE;
                if (surf->texinfo->texture->gltexture) {
                    extra.s_0 = merian::float_to_half(p->verts[0][3]);
                    extra.t_0 = merian::float_to_half(p->verts[0][4]);
                    extra.s_1 = merian::float_to_half(p->verts[k - 1][3]);
                    extra.t_1 = merian::float_to_half(p->verts[k - 1][4]);
                    extra.s_2 = merian::float_to_half(p->verts[k - 0][3]);
                    extra.t_2 = merian::float_to_half(p->verts[k - 0][4]);
                    extra.texnum_alpha = make_texnum_alpha(t->gltexture, ent, surf);
                    extra.texnum_fb_flags = t->fullbright ? t->fullbright->texnum : 0;

                    if (surf->flags & SURF_DRAWLAVA)
                        flags = MAT_FLAGS_LAVA;
                    if (surf->flags & SURF_DRAWSLIME)
                        flags = MAT_FLAGS_SLIME;
                    if (surf->flags & SURF_DRAWTELE)
                        flags = MAT_FLAGS_TELE;
                    if (surf->flags & SURF_DRAWWATER)
                        flags = MAT_FLAGS_WATER;
                    if (strstr(t->gltexture->name, "wfall"))
                        flags = MAT_FLAGS_WATERFALL; // hack for ad_tears and emissive waterfalls
                }
                if (surf->flags & SURF_DRAWSKY)
                    flags = MAT_FLAGS_SKY;
                // max textures is 4096 (12 bit) and we have 16. so we can put 4 bits
                // worth of flags here:
                extra.texnum_fb_flags |= flags << 12;

                ext.push_back(extra);
            }
            // p = p->next;
            p = 0; // XXX
        }
    }
}

void add_geo_sprite(entity_t* ent,
                    [[maybe_unused]] qmodel_t* m,
                    std::vector<float>& vtx,
                    std::vector<float>& prev_vtx,
                    std::vector<uint32_t>& idx,
                    std::vector<VertexExtraData>& ext) {
    assert(m->type == mod_sprite);

    glm::vec3 v_forward, v_right, v_up;
    msprite_t* psprite;
    mspriteframe_t* frame;
    glm::vec3 s_up, s_right;
    float angle, sr, cr;
    float scale = ENTSCALE_DECODE(ent->scale);

    // pretty much from r_sprite:
    glm::vec3 vpn, vright, vup, r_origin;
    VectorCopy(r_refdef.vieworg, r_origin);
    AngleVectors(r_refdef.viewangles, &vpn.x, &vright.x, &vup.x);

    frame = R_GetSpriteFrame(ent);
    psprite = (msprite_t*)ent->model->cache.data;

    if (!frame->gltexture)
        return;

    switch (psprite->type) {
    case SPR_VP_PARALLEL_UPRIGHT: // faces view plane, up is towards the heavens
        v_up[0] = 0;
        v_up[1] = 0;
        v_up[2] = 1;
        v_right = glm::normalize(glm::cross(vpn, v_up));
        s_up = v_up;
        s_right = v_right;
        break;
    case SPR_FACING_UPRIGHT: // faces camera origin, up is towards the heavens
        VectorSubtract(ent->origin, r_origin, v_forward);
        v_forward[2] = 0;
        VectorNormalizeFast(&v_forward.x);
        v_right[0] = v_forward[1];
        v_right[1] = -v_forward[0];
        v_right[2] = 0;
        v_up[0] = 0;
        v_up[1] = 0;
        v_up[2] = 1;
        s_up = v_up;
        s_right = v_right;
        break;
    case SPR_VP_PARALLEL: // faces view plane, up is towards the top of the screen
        s_up = vup;
        s_right = vright;
        break;
    case SPR_ORIENTED: // pitch yaw roll are independent of camera
        AngleVectors(ent->angles, &v_forward.x, &v_right.x, &v_up.x);
        s_up = v_up;
        s_right = v_right;
        break;
    case SPR_VP_PARALLEL_ORIENTED: // faces view plane, but obeys roll value
        angle = ent->angles[ROLL] * M_PI_DIV_180;
        sr = sin(angle);
        cr = cos(angle);
        v_right[0] = vright[0] * cr + vup[0] * sr;
        v_right[1] = vright[1] * cr + vup[1] * sr;
        v_right[2] = vright[2] * cr + vup[2] * sr;
        v_up[0] = vright[0] * -sr + vup[0] * cr;
        v_up[1] = vright[1] * -sr + vup[1] * cr;
        v_up[2] = vright[2] * -sr + vup[2] * cr;
        s_up = v_up;
        s_right = v_right;
        break;
    default:
        return;
    }

    s_up = glm::normalize(s_up);
    s_right = glm::normalize(s_right);

    // add two quads
    for (int k = 0; k < 2; k++) {
        glm::vec3 v0, v1, v2, v3;

        // clang-format off
        switch (k) {
        case 0: {
            v0 = scale * (frame->down * s_up + frame->left * s_right);
            v1 = scale * (frame->up * s_up + frame->left * s_right);
            v2 = scale * (frame->up * s_up + frame->right * s_right);
            v3 = scale * (frame->down * s_up + frame->right * s_right);
            break;
        }
        case 1: {
            v0 = scale * (frame->down * s_up - frame->left * s_right);
            v1 = scale * (frame->up * s_up - frame->left * s_right);
            v2 = scale * (frame->up * s_up - frame->right * s_right);
            v3 = scale * (frame->down * s_up - frame->right * s_right);
            break;
        }
        default:
            assert(0);
        }
        // clang-format on

        // add vertices - triangle fan
        uint32_t vtx_cnt = vtx.size() / 3;
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(v0[l] + ent->origin[l]);
            prev_vtx.emplace_back(v0[l] + ent->mv_prev_origin[l]);
        }
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(v1[l] + ent->origin[l]);
            prev_vtx.emplace_back(v1[l] + ent->mv_prev_origin[l]);
        }
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(v2[l] + ent->origin[l]);
            prev_vtx.emplace_back(v2[l] + ent->mv_prev_origin[l]);
        }
        for (int l = 0; l < 3; l++) {
            vtx.emplace_back(v3[l] + ent->origin[l]);
            prev_vtx.emplace_back(v3[l] + ent->mv_prev_origin[l]);
        }

        // add index - tiangle fan
        idx.emplace_back(vtx_cnt + 0);
        idx.emplace_back(vtx_cnt + 1);
        idx.emplace_back(vtx_cnt + 2);

        idx.emplace_back(vtx_cnt + 0);
        idx.emplace_back(vtx_cnt + 2);
        idx.emplace_back(vtx_cnt + 3);

        // add extra data
        const glm::vec3 e0(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]);
        const glm::vec3 e1(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]);
        const uint32_t n_enc = merian::encode_normal(glm::normalize(glm::cross(e0, e1)));

        const uint16_t texnum = make_texnum_alpha(frame->gltexture);

        // clang-format off
        ext.emplace_back(texnum,
                         MAT_FLAGS_SPRITE << 12, // sprite allways emits
                         n_enc, n_enc, n_enc,
                         merian::float_to_half(0),            merian::float_to_half(frame->tmax),
                         merian::float_to_half(0),            merian::float_to_half(0),
                         merian::float_to_half(frame->smax),  merian::float_to_half(0));
        ext.emplace_back(texnum,
                         MAT_FLAGS_SPRITE << 12, // sprite allways emits
                         n_enc, n_enc, n_enc,
                         merian::float_to_half(0),            merian::float_to_half(frame->tmax),
                         merian::float_to_half(frame->smax),  merian::float_to_half(0),
                         merian::float_to_half(frame->smax),  merian::float_to_half(frame->tmax));
        // clang-format on

    } // end three axes

    VectorCopy(ent->origin, ent->mv_prev_origin);
}

// Adds the geo from entity into the vectors.
void add_geo(entity_t* ent,
             std::vector<float>& vtx,
             std::vector<float>& prev_vtx,
             std::vector<uint32_t>& idx,
             std::vector<VertexExtraData>& ext) {
    if (!ent)
        return;
    qmodel_t* m = ent->model;
    if (!m)
        return;

    if (m->type == mod_alias) { // alias model:
        add_geo_alias(ent, m, vtx, prev_vtx, idx, ext);
    } else if (m->type == mod_brush) { // brush model:
        add_geo_brush(ent, m, vtx, prev_vtx, idx, ext);
    } else if (m->type == mod_sprite) {
        // explosions, decals, etc, this is R_DrawSpriteModel
        add_geo_sprite(ent, m, vtx, prev_vtx, idx, ext);
    }

    assert(ext.size() == idx.size() / 3);
    assert(vtx.size() % 3 == 0);
    assert(vtx.size() == prev_vtx.size());
}
