#include "renderer/render_markovchain.hpp"

#include "ext/json.hpp"
#include "game/quake_node.hpp"
#include "merian-nodes/common/gbuffer.glsl.h"
#include "merian/utils/bitpacking.hpp"
#include "merian/utils/concurrent/utils.hpp"
#include "merian/utils/glm.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/vector.hpp"
#include "merian/utils/xorshift.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "merian/vk/utils/math.hpp"

#include "grid.h"

#include "clear.comp.spv.h"
#include "quake.comp.spv.h"
#include "volume.comp.spv.h"
#include "volume_forward_project.comp.spv.h"

#include <fstream>
#include <random>

extern "C" {

#include "bgmusic.h"
#include "quakedef.h"
#include "screen.h"

// from r_alias.c
extern float r_avertexnormals[162][3];
// from r_part.c
extern particle_t* active_particles;

extern qboolean scr_drawloading;

extern cvar_t scr_fov, cl_gun_fovscale;
}

// Helper -----------------------------------------------------------------------------------------

uint16_t
make_texnum_alpha(gltexture_s* tex, entity_t* entity = nullptr, msurface_t* surface = nullptr) {
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
                   std::vector<RendererMarkovChain::VertexExtraData>& ext,
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
                                 merian::float_to_half_aprox(0), merian::float_to_half_aprox(1),
                                 merian::float_to_half_aprox(0), merian::float_to_half_aprox(0),
                                 merian::float_to_half_aprox(1), merian::float_to_half_aprox(0));
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
                                 merian::float_to_half_aprox(0), merian::float_to_half_aprox(1),
                                 merian::float_to_half_aprox(0), merian::float_to_half_aprox(0),
                                 merian::float_to_half_aprox(1), merian::float_to_half_aprox(0));
            }
        }
    }
}

void add_geo_alias(entity_t* ent,
                   [[maybe_unused]] qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<RendererMarkovChain::VertexExtraData>& ext) {
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
                         merian::float_to_half_aprox((desc[indexes[3 * i + 0]].st[0] + 0.5) /
                                                     (float)hdr->skinwidth),
                         merian::float_to_half_aprox((desc[indexes[3 * i + 0]].st[1] + 0.5) /
                                                     (float)hdr->skinheight),
                         merian::float_to_half_aprox((desc[indexes[3 * i + 1]].st[0] + 0.5) /
                                                     (float)hdr->skinwidth),
                         merian::float_to_half_aprox((desc[indexes[3 * i + 1]].st[1] + 0.5) /
                                                     (float)hdr->skinheight),
                         merian::float_to_half_aprox((desc[indexes[3 * i + 2]].st[0] + 0.5) /
                                                     (float)hdr->skinwidth),
                         merian::float_to_half_aprox((desc[indexes[3 * i + 2]].st[1] + 0.5) /
                                                     (float)hdr->skinheight));
    }
}

// geo_selector: 0 -> all, 1 -> opaque, 2 -> transparent
void add_geo_brush(entity_t* ent,
                   qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<RendererMarkovChain::VertexExtraData>& ext,
                   int geo_selector = 0) {
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
                RendererMarkovChain::VertexExtraData extra{
                    .texnum_alpha = 0,
                    .texnum_fb_flags = 0,
                    .n0_gloss_norm = merian::pack_uint32(t->gloss ? t->gloss->texnum : 0,
                                                         t->norm ? t->norm->texnum : 0),
                    .n1_brush = 0xffffffff,
                    .n2 = 0,
                };
                uint32_t flags = MAT_FLAGS_NONE;
                if (surf->texinfo->texture->gltexture) {
                    extra.s_0 = merian::float_to_half_aprox(p->verts[0][3]);
                    extra.t_0 = merian::float_to_half_aprox(p->verts[0][4]);
                    extra.s_1 = merian::float_to_half_aprox(p->verts[k - 1][3]);
                    extra.t_1 = merian::float_to_half_aprox(p->verts[k - 1][4]);
                    extra.s_2 = merian::float_to_half_aprox(p->verts[k - 0][3]);
                    extra.t_2 = merian::float_to_half_aprox(p->verts[k - 0][4]);
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
                    std::vector<RendererMarkovChain::VertexExtraData>& ext) {
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
                         merian::float_to_half_aprox(0),            merian::float_to_half_aprox(frame->tmax),
                         merian::float_to_half_aprox(0),            merian::float_to_half_aprox(0),
                         merian::float_to_half_aprox(frame->smax),  merian::float_to_half_aprox(0));
        ext.emplace_back(texnum,
                         MAT_FLAGS_SPRITE << 12, // sprite allways emits
                         n_enc, n_enc, n_enc,
                         merian::float_to_half_aprox(0),            merian::float_to_half_aprox(frame->tmax),
                         merian::float_to_half_aprox(frame->smax),  merian::float_to_half_aprox(0),
                         merian::float_to_half_aprox(frame->smax),  merian::float_to_half_aprox(frame->tmax));
        // clang-format on

    } // end three axes

    VectorCopy(ent->origin, ent->mv_prev_origin);
}

// Adds the geo from entity into the vectors.
void add_geo(entity_t* ent,
             std::vector<float>& vtx,
             std::vector<float>& prev_vtx,
             std::vector<uint32_t>& idx,
             std::vector<RendererMarkovChain::VertexExtraData>& ext) {
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

// If the supplied buffer is not nullptr and is large enough, it is returned and an upload is
// recorded. A new larger buffer is used otherwise.
template <typename T>
merian::BufferHandle ensure_buffer(const merian::ResourceAllocatorHandle& allocator,
                                   const vk::BufferUsageFlags usage,
                                   const vk::CommandBuffer& cmd,
                                   const std::vector<T>& data,
                                   const merian::BufferHandle optional_buffer,
                                   const std::optional<vk::DeviceSize> min_alignment = std::nullopt,
                                   const std::string& debug_name = {}) {
    merian::BufferHandle buffer;
    if (optional_buffer && optional_buffer->get_size() >= sizeof(T) * data.size()) {
        buffer = optional_buffer;

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer, {}, {},
            buffer->buffer_barrier(vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                       vk::AccessFlagBits::eShaderRead |
                                       vk::AccessFlagBits::eTransferRead,
                                   vk::AccessFlagBits::eTransferWrite),
            {});
    } else {
        // Make it a bit larger for logarithmic growth
        buffer =
            allocator->createBuffer(sizeof(T) * data.size() * 1.25, usage,
                                    merian::MemoryMappingType::NONE, debug_name, min_alignment);
    }
    allocator->getStaging()->cmdToBuffer(cmd, *buffer, 0, sizeof(T) * data.size(), data.data());
    return buffer;
}

// Creates a vertex and index buffer for rt on the device and records the upload.
// If the supplied buffers are not nullptr and are large enough, they are returned and an upload is
// recorded. Returns (vertex_buffer, index_buffer).
// Appropriate barriers are inserted.
std::tuple<merian::BufferHandle, merian::BufferHandle, merian::BufferHandle, merian::BufferHandle>
ensure_vertex_index_ext_buffer(const merian::ResourceAllocatorHandle& allocator,
                               const vk::CommandBuffer& cmd,
                               const std::vector<float>& vtx,
                               const std::vector<float>& prev_vtx,
                               const std::vector<uint32_t>& idx,
                               const std::vector<RendererMarkovChain::VertexExtraData>& ext,
                               const merian::BufferHandle optional_vtx_buffer,
                               const merian::BufferHandle optional_prev_vtx_buffer,
                               const merian::BufferHandle optional_idx_buffer,
                               const merian::BufferHandle optional_ext_buffer) {
    auto usage_rt = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    auto usage_storage = vk::BufferUsageFlagBits::eStorageBuffer;

    merian::BufferHandle vertex_buffer = ensure_buffer(
        allocator, usage_rt, cmd, vtx, optional_vtx_buffer, {}, "Quake: vertex buffer");
    merian::BufferHandle prev_vertex_buffer =
        ensure_buffer(allocator, usage_rt, cmd, prev_vtx, optional_prev_vtx_buffer, {},
                      "Quake: previous vertex buffer");
    merian::BufferHandle index_buffer = ensure_buffer(
        allocator, usage_rt, cmd, idx, optional_idx_buffer, {}, "Quake: index buffer");
    merian::BufferHandle ext_buffer = ensure_buffer(allocator, usage_storage, cmd, ext,
                                                    optional_ext_buffer, {}, "Quake: ext buffer");

    const std::array<vk::BufferMemoryBarrier2, 4> barriers = {
        vertex_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
                vk::AccessFlagBits2::eTransferWrite),
        prev_vertex_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
                vk::AccessFlagBits2::eTransferWrite),
        index_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
                vk::AccessFlagBits2::eTransferWrite),
        ext_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferWrite),
    };

    vk::DependencyInfo dep_info{{}, {}, barriers, {}};
    cmd.pipelineBarrier2(dep_info);

    return std::make_tuple(vertex_buffer, prev_vertex_buffer, index_buffer, ext_buffer);
}

// QuakeNode
// --------------------------------------------------------------------------------------

RendererMarkovChain::RendererMarkovChain(const merian::SharedContext& context,
                                         const merian::ResourceAllocatorHandle& allocator)
    : Node("MarkovChain Renderer"), context(context), allocator(allocator) {

    // PIPELINE CREATION
    rt_shader = std::make_shared<merian::ShaderModule>(context, merian_quake_comp_spv_size(),
                                                       merian_quake_comp_spv());
    clear_shader = std::make_shared<merian::ShaderModule>(context, merian_clear_comp_spv_size(),
                                                          merian_clear_comp_spv());
    volume_shader = std::make_shared<merian::ShaderModule>(context, merian_volume_comp_spv_size(),
                                                           merian_volume_comp_spv());
    volume_forward_project_shader = std::make_shared<merian::ShaderModule>(
        context, merian_volume_forward_project_comp_spv_size(),
        merian_volume_forward_project_comp_spv());

    quake_desc_set_layout =
        merian::DescriptorSetLayoutBuilder()
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, MAX_GEOMETRIES)
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, MAX_GEOMETRIES)
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, MAX_GEOMETRIES)
            .add_binding_acceleration_structure()
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, MAX_GEOMETRIES)
            .build_layout(context);
    quake_pool = std::make_shared<merian::DescriptorPool>(
        quake_desc_set_layout, 3, vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    binding_dummy_buffer =
        allocator->createBuffer(8, vk::BufferUsageFlagBits::eStorageBuffer,
                                merian::MemoryMappingType::NONE, "Quake: Dummy buffer");
}

RendererMarkovChain::~RendererMarkovChain() {}

// -------------------------------------------------------------------------------------------

std::vector<merian_nodes::InputConnectorHandle> RendererMarkovChain::describe_inputs() {
    return {
        con_blue_noise, con_prev_filtered, con_prev_volume_depth,
        con_prev_gbuf,  con_render_info,   con_textures,
    };
}

std::vector<merian_nodes::OutputConnectorHandle> RendererMarkovChain::describe_outputs(
    [[maybe_unused]] const merian_nodes::ConnectorIOMap& output_for_input) {

    con_irradiance = merian_nodes::ManagedVkImageOut::compute_write(
        "irradiance", vk::Format::eR32G32B32A32Sfloat, render_width, render_height);
    con_albedo = merian_nodes::ManagedVkImageOut::compute_write(
        "albedo", vk::Format::eR16G16B16A16Sfloat, render_width, render_height);
    con_mv = merian_nodes::ManagedVkImageOut::compute_write("mv", vk::Format::eR16G16Sfloat,
                                                            render_width, render_height);
    con_debug = merian_nodes::ManagedVkImageOut::compute_write(
        "debug", vk::Format::eR16G16B16A16Sfloat, render_width, render_height);
    con_moments = merian_nodes::ManagedVkImageOut::compute_write(
        "moments", vk::Format::eR32G32Sfloat, render_width, render_height);
    con_volume = merian_nodes::ManagedVkImageOut::compute_write(
        "volume", vk::Format::eR16G16B16A16Sfloat, render_width, render_height);
    con_volume_moments = merian_nodes::ManagedVkImageOut::compute_write(
        "volume_moments", vk::Format::eR32G32Sfloat, render_width, render_height);
    con_volume_depth = merian_nodes::ManagedVkImageOut::compute_write(
        "volume_depth", vk::Format::eR32Sfloat, render_width, render_height);
    con_volume_mv = merian_nodes::ManagedVkImageOut::compute_write(
        "volume_mv", vk::Format::eR16G16Sfloat, render_width, render_height);

    con_markovchain = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "markovchain", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             (mc_adaptive_buffer_size + mc_static_buffer_size) * sizeof(MCState),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc},
        true);
    con_lightcache = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "lightcache", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             light_cache_buffer_size * sizeof(LightCacheVertex),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc},
        true);
    con_gbuffer = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "gbuffer", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             render_width * render_height * sizeof(merian_nodes::GBuffer),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc});
    con_volume_distancemc = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "volume_distancemc", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             (render_width / distance_mc_grid_width + 2) *
                                 (render_height / distance_mc_grid_width + 2) *
                                 MAX_DISTANCE_MC_VERTEX_STATE_COUNT * sizeof(DistanceMCState),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc},
        true);

    return {
        con_irradiance,
        con_albedo,
        con_mv,
        con_debug,
        con_moments,
        con_volume,
        con_volume_moments,
        con_volume_depth,
        con_volume_mv,

        con_markovchain,
        con_lightcache,
        con_gbuffer,
        con_volume_distancemc,
    };
}

RendererMarkovChain::NodeStatusFlags
RendererMarkovChain::on_connected(const merian::DescriptorSetLayoutHandle& graph_desc_set_layout) {
    this->graph_desc_set_layout = graph_desc_set_layout;
    pipe.reset();

    prev_cl_time = cl.time;
    return {};
}

void RendererMarkovChain::process(merian_nodes::GraphRun& run,
                                  const vk::CommandBuffer& cmd,
                                  const merian::DescriptorSetHandle& graph_descriptor_set,
                                  const merian_nodes::NodeIO& io) {
    const QuakeNode::QuakeRenderInfo& render_info =
        std::any_cast<const QuakeNode::QuakeRenderInfo&>(io[con_render_info]);

    // (RE-) CREATE PIPELINE
    if (!pipe || !clear_pipe || !volume_pipe || !volume_forward_project_pipe) {
        glm::vec3 sun_dir;
        glm::vec3 sun_col;
        if (overwrite_sun) {
            sun_dir = overwrite_sun_dir;
            sun_col = overwrite_sun_col;
        } else {
            sun_dir = render_info.sun_direction;
            sun_col = render_info.sun_color;
        }
        if (glm::length(sun_dir) > 0)
            sun_dir = glm::normalize(sun_dir);

        if (randomize_seed) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<uint32_t> dist;
            seed = dist(rng);
        }

        // Quake sets fov assuming a 4x3 screen :D
        vid.width = render_width;
        vid.height = render_height;
        fov_tan_alpha_half = glm::tan(glm::radians(r_refdef.fov_x) / 2);

        auto pipe_layout = merian::PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(graph_desc_set_layout)
                               .add_descriptor_set_layout(quake_desc_set_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = merian::SpecializationInfoBuilder();
        const float draine_g = std::exp(-2.20679 / (volume_particle_size_um + 3.91029) - 0.428934);
        const float draine_a = std::exp(3.62489 - 8.29288 / (volume_particle_size_um + 5.52825));
        spec_builder.add_entry(
            local_size_x, local_size_y, spp, max_path_length, use_light_cache_tail,
            fov_tan_alpha_half, sun_dir.x, sun_dir.y, sun_dir.z, sun_col.r, sun_col.g, sun_col.b,
            adaptive_sampling, volume_spp, volume_use_light_cache, draine_g, draine_a, mc_samples,
            mc_samples_adaptive_prob, distance_mc_samples, mc_fast_recovery, light_cache_levels,
            light_cache_tan_alpha_half, light_cache_buffer_size, mc_adaptive_buffer_size,
            mc_static_buffer_size, mc_adaptive_grid_tan_alpha_half, mc_static_grid_width,
            mc_adaptive_grid_levels, distance_mc_grid_width, volume_max_t, surf_bsdf_p,
            volume_phase_p, dir_guide_prior, dist_guide_p, distance_mc_vertex_state_count, seed);

        auto spec = spec_builder.build();

        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, rt_shader, spec);
        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, clear_shader, spec);
        volume_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, volume_shader, spec);
        volume_forward_project_pipe = std::make_shared<merian::ComputePipeline>(
            pipe_layout, volume_forward_project_shader, spec);
    }

    auto& cur_frame_ptr = io.frame_data<std::shared_ptr<FrameData>>();
    if (!cur_frame_ptr)
        cur_frame_ptr = std::make_shared<FrameData>();
    FrameData& cur_frame = *cur_frame_ptr;

    cur_frame.pipe = pipe;
    cur_frame.clear_pipe = clear_pipe;
    cur_frame.volume_pipe = volume_pipe;
    cur_frame.volume_forward_project_pipe = volume_forward_project_pipe;

    // RESET MARKOV CHAINS AT ITERATION 0
    if (run.get_iteration() == 0ul) {
        // ZERO markov chains and light cache
        io[con_markovchain]->fill(cmd);
        io[con_lightcache]->fill(cmd);
        io[con_gbuffer]->fill(cmd);
        io[con_volume_distancemc]->fill(cmd);

        std::vector<vk::BufferMemoryBarrier> barriers = {
            io[con_markovchain]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eShaderRead),
            io[con_lightcache]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                               vk::AccessFlagBits::eShaderRead),
            io[con_gbuffer]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                            vk::AccessFlagBits::eShaderRead),
            io[con_volume_distancemc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead),
        };

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, barriers, {});
    }

    // CREATE COPIES FOR THE CURRENT FRAME IN FLIGHT
    if (!cur_frame.quake_sets) {
        cur_frame.quake_sets = std::make_shared<merian::DescriptorSet>(quake_pool);
        cur_frame.blas_builder = std::make_unique<merian::BLASBuilder>(context, allocator);
        cur_frame.tlas_builder = std::make_unique<merian::TLASBuilder>(context, allocator);

        merian::DescriptorSetUpdate update(cur_frame.quake_sets);

        for (uint32_t geometry = 0; geometry < MAX_GEOMETRIES; geometry++) {
            update.write_descriptor_buffer(BINDING_VTX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                           geometry);
            update.write_descriptor_buffer(BINDING_PREV_VTX_BUF, binding_dummy_buffer, 0,
                                           VK_WHOLE_SIZE, geometry);
            update.write_descriptor_buffer(BINDING_IDX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                           geometry);
            update.write_descriptor_buffer(BINDING_EXT_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                           geometry);
        }

        update.update(context);
    }

    // UPDATE GEOMETRY
    if (cl.worldmodel) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update geo");
        if (render_info.worldspawn) {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "upload static");
            update_static_geo(cmd, true, cur_frame);
            pipe.reset();
        } else {
            update_static_geo(cmd, false, cur_frame);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "upload dynamic");
            update_dynamic_geo(cmd, cur_frame, render_info.texnum_blood,
                               render_info.texnum_explosion);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "build acceleration structures");
            update_as(cmd, run.get_profiler(), cur_frame);
        }
    }

    if (!render_info.render || !cur_frame.tlas || !cl.worldmodel || scr_drawloading || render_info.worldspawn) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "clear");
        clear_pipe->bind(cmd);
        clear_pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        clear_pipe->bind_descriptor_set(cmd, cur_frame.quake_sets, 1);
        clear_pipe->push_constant(cmd, pc);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
        frame++;
        return;
    }

    // UPDATE PUSH CONSTANT
    pc.frame = frame - render_info.last_worldspawn_frame;
    if (sv_player) {
        // Demos do not have a player set
        pc.player.flags = 0;
        pc.player.flags |= sv_player->v.weapon == 1 ? PLAYER_FLAGS_TORCH : 0; // shotgun has torch
        pc.player.flags |= sv_player->v.waterlevel >= 3 ? PLAYER_FLAGS_UNDERWATER : 0;
    } else {
        pc.player = {0, 0, 0, 0};
    }
    pc.cl_time = cl.time;
    pc.prev_cam_x_mu_sx = pc.cam_x_mu_t;
    pc.prev_cam_w_mu_sy = pc.cam_w;
    pc.prev_cam_u_mu_sz = pc.cam_u;
    float rgt[3];
    AngleVectors(r_refdef.viewangles, &pc.cam_w.x, rgt, &pc.cam_u.x);
    pc.cam_x_mu_t = glm::vec4(*merian::as_vec3(r_refdef.vieworg), 1);
    pc.sky.fill(0);
    if (skybox_name[0]) {
        for (int i = 0; i < 6; i++)
            pc.sky[i] = skybox_textures[i]->texnum;
    } else if (solidskytexture) {
        pc.sky[0] = solidskytexture->texnum;
        pc.sky[1] = alphaskytexture->texnum;
        pc.sky[2] = static_cast<uint16_t>(-1u);
    }

    if (mu_t_s_overwrite) {
        pc.cam_x_mu_t.a = mu_t;
        pc.prev_cam_x_mu_sx.a = mu_s_div_mu_t.r * mu_t;
        pc.prev_cam_w_mu_sy.a = mu_s_div_mu_t.g * mu_t;
        pc.prev_cam_u_mu_sz.a = mu_s_div_mu_t.b * mu_t;
    } else {
        pc.cam_x_mu_t.a = Fog_GetDensity();
        pc.cam_x_mu_t.a *= pc.cam_x_mu_t.a;
        pc.cam_x_mu_t.a *= 0.1;

        const float* fog_color = Fog_GetColor();
        pc.prev_cam_x_mu_sx.a = fog_color[0] * pc.cam_x_mu_t.a;
        pc.prev_cam_w_mu_sy.a = fog_color[1] * pc.cam_x_mu_t.a;
        pc.prev_cam_u_mu_sz.a = fog_color[2] * pc.cam_x_mu_t.a;
    }
    {
        // motion tracking time diff
        const float time_diff = pc.cl_time - prev_cl_time;
        if (time_diff > 0)
            pc.cam_w.a = time_diff;
        else
            pc.cam_w.a = 1.0;
    }

    // BIND PIPELINE
    {
        // Surfaces and GBuffer

        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "quake.comp");
        pipe->bind(cmd);
        pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        pipe->bind_descriptor_set(cmd, cur_frame.quake_sets, 1);
        pipe->push_constant(cmd, pc);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
    }

    if (volume_forward_project && volume_spp > 0) {
        // Forward project motion vectors for volumes
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "volume forward project");
        volume_forward_project_pipe->bind(cmd);
        volume_forward_project_pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        volume_forward_project_pipe->bind_descriptor_set(cmd, cur_frame.quake_sets, 1);
        volume_forward_project_pipe->push_constant(cmd, pc);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
    }

    auto volume_mv_bar =
        io[con_volume_mv]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                   vk::AccessFlagBits::eShaderRead);
    auto gbuf_bar = io[con_gbuffer]->buffer_barrier(vk::AccessFlagBits::eMemoryWrite,
                                                    vk::AccessFlagBits::eMemoryRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, gbuf_bar, volume_mv_bar);
    {
        // Volumes

        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "volume");
        volume_pipe->bind(cmd);
        volume_pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        volume_pipe->bind_descriptor_set(cmd, cur_frame.quake_sets, 1);
        volume_pipe->push_constant(cmd, pc);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
    }

    if (dump_mc) {
        const std::size_t count =
            std::min(128 * 1024 * 1024 / sizeof(MCState), (std::size_t)mc_adaptive_buffer_size);
        const MCState* buf = static_cast<const MCState*>(allocator->getStaging()->cmdFromBuffer(
            cmd, *io[con_markovchain], 0, sizeof(MCState) * count));
        run.add_submit_callback([count, buf](const merian::QueueHandle& queue) {
            queue->wait_idle();
            nlohmann::json j;

            for (const MCState* v = buf; v < buf + count; v++) {
                nlohmann::json o;
                o["N"] = v->N;
                o["hash"] = v->hash;
                o["w_cos"] = v->w_cos;
                o["sum_w"] = v->sum_w;
                o["w_tgt"] = fmt::format("{} {} {}", v->w_tgt.x, v->w_tgt.y, v->w_tgt.z);

                j.emplace_back(o);
            }

            std::ofstream file("mc_dump.json");
            file << std::setw(4) << j << std::endl;
        });

        dump_mc = false;
    }

    prev_cl_time = cl.time;
    frame++;
}

RendererMarkovChain::RTGeometry
RendererMarkovChain::get_rt_geometry(const vk::CommandBuffer& cmd,
                                     const std::vector<float>& vtx,
                                     const std::vector<float>& prev_vtx,
                                     const std::vector<uint32_t>& idx,
                                     const std::vector<RendererMarkovChain::VertexExtraData>& ext,
                                     const std::unique_ptr<merian::BLASBuilder>& blas_builder,
                                     const RTGeometry old_geo,
                                     const bool force_rebuild,
                                     const vk::BuildAccelerationStructureFlagsKHR flags) {
    assert(!vtx.empty());
    assert(!prev_vtx.empty());
    assert(!idx.empty());
    assert(!ext.empty());

    RTGeometry geo;

    geo.vtx_count = vtx.size() / 3;
    geo.primitive_count = idx.size() / 3;

    std::tie(geo.vtx_buffer, geo.prev_vtx_buffer, geo.idx_buffer, geo.ext_buffer) =
        ensure_vertex_index_ext_buffer(allocator, cmd, vtx, prev_vtx, idx, ext,
                                       old_geo.prev_vtx_buffer, old_geo.vtx_buffer,
                                       old_geo.idx_buffer, old_geo.ext_buffer);

    vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
        vk::Format::eR32G32B32Sfloat,
        geo.vtx_buffer->get_device_address(),
        3 * sizeof(float),
        geo.vtx_count - 1,
        vk::IndexType::eUint32,
        {geo.idx_buffer->get_device_address()},
        {}};

    vk::AccelerationStructureGeometryKHR geometry{vk::GeometryTypeKHR::eTriangles, {triangles}};
    vk::AccelerationStructureBuildRangeInfoKHR range_info{geo.primitive_count, 0, 0, 0};

    if (old_geo.vtx_count == geo.vtx_count && old_geo.primitive_count == geo.primitive_count) {
        // Rebuild and update are possible
        if (force_rebuild) {
            blas_builder->queue_rebuild({geometry}, {range_info}, old_geo.blas, old_geo.blas_flags);
            geo.last_rebuild = frame;
        } else {
            blas_builder->queue_update({geometry}, {range_info}, old_geo.blas, old_geo.blas_flags);
        }
        geo.blas = old_geo.blas;
        geo.blas_flags = old_geo.blas_flags;
    } else {
        geo.blas = blas_builder->queue_build({geometry}, {range_info}, flags);
        geo.blas_flags = flags;
        geo.last_rebuild = frame;
    }

    return geo;
}

void RendererMarkovChain::update_static_geo(const vk::CommandBuffer& cmd,
                                            const bool refresh_geo,
                                            FrameData& cur_frame) {
    if (refresh_geo) {
        std::vector<RTGeometry> old_static_geo = current_static_geo;
        current_static_geo.clear();

        // clang-format off
        static_vtx.clear();
        static_prev_vtx.clear();
        static_idx.clear();
        static_ext.clear();

        add_geo_brush(cl_entities, cl_entities->model, static_vtx, static_prev_vtx, static_idx, static_ext, 1);
        if (!static_idx.empty()) {
            RTGeometry old_geo = old_static_geo.size() > 0 ? old_static_geo[0] : RTGeometry();
            current_static_geo.emplace_back(get_rt_geometry(cmd, static_vtx, static_vtx, static_idx, static_ext, cur_frame.blas_builder, old_geo, true, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace));
            current_static_geo.back().instance_flags = vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise
            | vk::GeometryInstanceFlagBitsKHR::eForceOpaque;
        }
        SPDLOG_DEBUG("static opaque geo: vtx size: {} idx size: {} ext size: {}", static_vtx.size(), static_idx.size(), static_ext.size());

        static_vtx.clear();
        static_prev_vtx.clear();
        static_idx.clear();
        static_ext.clear();

        add_geo_brush(cl_entities, cl_entities->model, static_vtx, static_prev_vtx, static_idx, static_ext, 2);
        if (!static_idx.empty()) {
            RTGeometry old_geo = old_static_geo.size() > 1 ? old_static_geo[1] : RTGeometry();
            current_static_geo.emplace_back(get_rt_geometry(cmd, static_vtx, static_vtx, static_idx, static_ext, cur_frame.blas_builder, old_geo, true, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace));
            current_static_geo.back().instance_flags = vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
        }
        SPDLOG_DEBUG("static non-opaque geo: vtx size: {} idx size: {} ext size: {}", static_vtx.size(), static_idx.size(), static_ext.size());

        // clang-format on
    }

    cur_frame.static_geometries = current_static_geo;
}

void RendererMarkovChain::update_dynamic_geo(const vk::CommandBuffer& cmd,
                                             FrameData& cur_frame,
                                             const uint32_t texnum_blood,
                                             const uint32_t texnum_explosion) {
    dynamic_vtx.clear();
    dynamic_prev_vtx.clear();
    dynamic_idx.clear();
    dynamic_ext.clear();

    std::future<void> future = context->thread_pool.submit<void>([&]() {
        if (playermodel == 1) {
            add_geo(&cl.viewent, dynamic_vtx, dynamic_prev_vtx, dynamic_idx, dynamic_ext);
        } else if (playermodel == 2) {
            add_geo(&cl.viewent, dynamic_vtx, dynamic_prev_vtx, dynamic_idx, dynamic_ext);
            add_geo(&cl_entities[cl.viewentity], dynamic_vtx, dynamic_prev_vtx, dynamic_idx,
                    dynamic_ext);
        }
        add_particles(dynamic_vtx, dynamic_prev_vtx, dynamic_idx, dynamic_ext, texnum_blood,
                      texnum_explosion, reproducible_renders, prev_cl_time);
    });

    const uint32_t number_tasks = context->thread_pool.size();

    std::vector<std::vector<float>> thread_dynamic_vtx(number_tasks);
    std::vector<std::vector<float>> thread_dynamic_prev_vtx(number_tasks);
    std::vector<std::vector<uint32_t>> thread_dynamic_idx(number_tasks);
    std::vector<std::vector<VertexExtraData>> thread_dynamic_ext(number_tasks);

    merian::parallel_for(
        std::max(cl_numvisedicts, cl.num_statics),
        [&](uint32_t index, uint32_t thread_index) {
            if (index < (uint32_t)cl_numvisedicts)
                add_geo(cl_visedicts[index], thread_dynamic_vtx[thread_index],
                        thread_dynamic_prev_vtx[thread_index], thread_dynamic_idx[thread_index],
                        thread_dynamic_ext[thread_index]);
            if (index < (uint32_t)cl.num_statics)
                add_geo(cl_static_entities + index, thread_dynamic_vtx[thread_index],
                        thread_dynamic_prev_vtx[thread_index], thread_dynamic_idx[thread_index],
                        thread_dynamic_ext[thread_index]);
        },
        context->thread_pool, number_tasks);

    future.get();

    for (uint32_t i = 0; i < number_tasks; i++) {
        uint32_t old_vtx_count = dynamic_vtx.size() / 3;

        merian::raw_copy_back(dynamic_vtx, thread_dynamic_vtx[i]);
        merian::raw_copy_back(dynamic_prev_vtx, thread_dynamic_prev_vtx[i]);
        merian::raw_copy_back(dynamic_ext, thread_dynamic_ext[i]);

        uint32_t old_idx_size = dynamic_idx.size();
        dynamic_idx.resize(old_idx_size + thread_dynamic_idx[i].size());
        for (uint idx = 0; idx < thread_dynamic_idx[i].size(); idx++) {
            dynamic_idx[old_idx_size + idx] = old_vtx_count + thread_dynamic_idx[i][idx];
        }
    }

    // for (int i=1 ; i<MAX_MODELS ; i++)
    //     add_geo(cl.model_precache+i, p->vtx + 3*vtx_cnt, p->idx + idx_cnt, 0, &vtx_cnt,
    //     &idx_cnt);
    // for (int i=0; i<cl_max_edicts; i++)
    //     add_geo(cl_entities+i, p->vtx + 3*vtx_cnt, p->idx + idx_cnt, 0, &vtx_cnt, &idx_cnt);

    SPDLOG_TRACE("dynamic geo: vtx size: {} idx size: {} ext size: {}", dynamic_vtx.size(),
                 dynamic_idx.size(), dynamic_ext.size());

    if (!dynamic_idx.empty()) {

        // Allows to find a old geometry that is large enough to be reused.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<RTGeometry>> vtx_prim_cnt_to_geo;
        for (auto& geo : cur_frame.dynamic_geometries) {
            std::pair<uint32_t, uint32_t> vtx_prim_cnt =
                std::make_pair(geo.vtx_count, geo.primitive_count);
            if (vtx_prim_cnt_to_geo.contains(vtx_prim_cnt))
                vtx_prim_cnt_to_geo[vtx_prim_cnt].emplace_back(std::move(geo));
            else
                vtx_prim_cnt_to_geo[vtx_prim_cnt] = {geo};
        }

        cur_frame.dynamic_geometries.clear();

        uint32_t new_vtx_cnt = dynamic_vtx.size() / 3;
        uint32_t new_prim_cnt = dynamic_idx.size() / 3;

        RTGeometry old_geo;
        auto candidates =
            vtx_prim_cnt_to_geo.lower_bound(std::make_pair(new_vtx_cnt, new_prim_cnt));
        if (candidates != vtx_prim_cnt_to_geo.end() && !(*candidates).second.empty()) {
            old_geo = (*candidates).second.back();
            (*candidates).second.pop_back();
        }

        bool force_rebuild = true; // frame - old_geo.last_rebuild > 1000;
        cur_frame.dynamic_geometries.emplace_back(
            get_rt_geometry(cmd, dynamic_vtx, dynamic_prev_vtx, dynamic_idx, dynamic_ext,
                            cur_frame.blas_builder, old_geo, force_rebuild,
                            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate));
        cur_frame.dynamic_geometries.back().instance_flags =
            vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
    } else {
        cur_frame.dynamic_geometries.clear();
    }
}

void RendererMarkovChain::update_as(const vk::CommandBuffer& cmd,
                                    const merian::ProfilerHandle profiler,
                                    FrameData& cur_frame) {
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "blas");
        cur_frame.blas_builder->get_cmds(cmd);
    }

    std::vector<RTGeometry> all_geometries;
    all_geometries.reserve(cur_frame.static_geometries.size() +
                           cur_frame.dynamic_geometries.size());
    merian::insert_all(all_geometries, cur_frame.static_geometries);
    merian::insert_all(all_geometries, cur_frame.dynamic_geometries);

    if (all_geometries.empty()) {
        cur_frame.tlas = nullptr;
        return;
    }

    assert(all_geometries.size() < MAX_GEOMETRIES);

    merian::DescriptorSetUpdate update(cur_frame.quake_sets);
    std::vector<vk::AccelerationStructureInstanceKHR> inst;
    for (uint32_t i = 0; i < all_geometries.size(); i++) {
        RTGeometry& geo = all_geometries[i];
        inst.emplace_back(vk::AccelerationStructureInstanceKHR{
            merian::transform_identity(),
            i,
            0xFF,
            0,
            geo.instance_flags,
            geo.blas->get_acceleration_structure_device_address(),
        });

        update.write_descriptor_buffer(BINDING_VTX_BUF, geo.vtx_buffer, 0, VK_WHOLE_SIZE, i);
        update.write_descriptor_buffer(BINDING_PREV_VTX_BUF, geo.prev_vtx_buffer, 0, VK_WHOLE_SIZE,
                                       i);
        update.write_descriptor_buffer(BINDING_IDX_BUF, geo.idx_buffer, 0, VK_WHOLE_SIZE, i);
        update.write_descriptor_buffer(BINDING_EXT_BUF, geo.ext_buffer, 0, VK_WHOLE_SIZE, i);
    }

    for (uint32_t i = all_geometries.size(); i < MAX_GEOMETRIES; i++) {
        update.write_descriptor_buffer(BINDING_VTX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE, i);
        update.write_descriptor_buffer(BINDING_PREV_VTX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                       i);
        update.write_descriptor_buffer(BINDING_IDX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE, i);
        update.write_descriptor_buffer(BINDING_EXT_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE, i);
    }

    const vk::BufferUsageFlags instances_buffer_usage =
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    cur_frame.instances_buffer =
        ensure_buffer(allocator, instances_buffer_usage, cmd, inst, cur_frame.instances_buffer, 16);
    const vk::BufferMemoryBarrier barrier = cur_frame.instances_buffer->buffer_barrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eAccelerationStructureWriteKHR);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, {}, barrier,
                        {});

    if (cur_frame.last_instances_size == inst.size()) {
        cur_frame.tlas_builder->queue_rebuild(inst.size(), cur_frame.instances_buffer,
                                              cur_frame.tlas);
    } else {
        cur_frame.tlas =
            cur_frame.tlas_builder->queue_build(inst.size(), cur_frame.instances_buffer);
        update.write_descriptor_acceleration_structure(BINDING_TLAS, *cur_frame.tlas);
        cur_frame.last_instances_size = inst.size();
    }

    update.update(context);

    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "tlas");
        cur_frame.tlas_builder->get_cmds(cmd);
        cur_frame.tlas_builder->cmd_barrier(cmd, vk::PipelineStageFlagBits::eComputeShader);
    }
}

RendererMarkovChain::NodeStatusFlags
RendererMarkovChain::configuration(merian::Configuration& config) {
    const int32_t old_render_width = render_width;
    const int32_t old_render_height = render_height;
    const int32_t old_spp = spp;
    const int32_t old_max_path_lenght = max_path_length;
    const int32_t old_use_light_cache_tail = use_light_cache_tail;
    const int32_t old_adaptive_sampling = adaptive_sampling;
    const int32_t old_volume_spp = volume_spp;
    const int32_t old_volume_use_light_cache = volume_use_light_cache;
    const float old_volume_particle_size_um = volume_particle_size_um;
    const int32_t old_mc_samples = mc_samples;
    const int32_t old_distance_mc_samples = distance_mc_samples;
    const float old_mc_samples_adaptive_prob = mc_samples_adaptive_prob;
    const int32_t old_mc_fast_recovery = mc_fast_recovery;
    const bool old_overwrite_sun = overwrite_sun;
    const glm::vec3 old_overwrite_sun_dir = overwrite_sun_dir;
    const glm::vec3 old_overwrite_sun_col = overwrite_sun_col;
    const float old_light_cache_levels = light_cache_levels;
    const float old_light_cache_tan_alpha_half = light_cache_tan_alpha_half;
    const uint32_t old_mc_adaptive_buffer_size = mc_adaptive_buffer_size;
    const uint32_t old_mc_static_buffer_size = mc_static_buffer_size;
    const float old_mc_adaptive_grid_tan_alpha_half = mc_adaptive_grid_tan_alpha_half;
    const int32_t old_mc_adaptive_grid_levels = mc_adaptive_grid_levels;
    const float old_mc_static_grid_width = mc_static_grid_width;
    const float old_distance_mc_grid_width = distance_mc_grid_width;
    const uint32_t old_light_cache_buffer_size = light_cache_buffer_size;
    const float old_volume_max_t = volume_max_t;
    const float old_surf_bsdf_p = surf_bsdf_p;
    const float old_volume_phase_p = volume_phase_p;
    const float old_dir_guide_prior = dir_guide_prior;
    const float old_dist_guide_p = dist_guide_p;
    const uint32_t old_distance_mc_vertex_state_count = distance_mc_vertex_state_count;
    const uint32_t old_seed = seed;
    const bool old_randomize_seed = randomize_seed;

    config.st_separate("General");
    config.config_bool("randomize seed", randomize_seed, "randomize seed at every graph build");
    if (!randomize_seed) {
        config.config_uint("seed", seed, "");
    } else {
        config.output_text(fmt::format("seed: {}", seed));
    }

    config.config_options("player model", playermodel, {"none", "gun only", "full"});

    config.config_int("render width", render_width, "The resolution for the raytracer");
    config.config_int("render height", render_height, "The resolution for the raytracer");

    config.st_separate("Guiding Markov chain");
    config.config_percent("ML Prior", dir_guide_prior);
    config.config_int("mc samples", mc_samples, 0, 30);
    config.config_percent("adaptive grid prob", mc_samples_adaptive_prob);
    config.config_uint("adaptive grid buf size", mc_adaptive_buffer_size,
                       "buffer size backing the hash grid");
    config.config_uint("static grid buf size", mc_static_buffer_size,
                       "buffer size backing the hash grid");
    config.config_float("mc adaptive tan(alpha/2)", mc_adaptive_grid_tan_alpha_half,
                        "the adaptive grid resolution, lower means higher resolution.", 0.0001);
    config.config_int("mc adaptive levels", mc_adaptive_grid_levels,
                      "number of quantization steps of the hash grid resolution");
    config.config_float("mc static width", mc_static_grid_width,
                        "the static grid width in worldspace units, lower means higher resolution",
                        0.1);
    config.config_bool("mc fast recovery", mc_fast_recovery,
                       "When enabled, markov chains are flooded with invalidated states when no "
                       "light is detected.");

    config.st_separate("RT Surface");
    config.config_int("spp", spp, 0, 15, "samples per pixel");
    // config.config_bool("adaptive sampling", adaptive_sampling, "Lowers spp adaptively");
    config.config_int("max path length", max_path_length, 0, 15, "maximum path length");
    config.config_percent("BSDF Prob", surf_bsdf_p, "the probability to use BSDF sampling");

    config.st_separate("RT Volume");
    config.config_int("volume spp", volume_spp, 0, 15, "samples per pixel for volume events");
    config.config_int("dist mc samples", distance_mc_samples, 0, 30);
    config.config_int("dist mc grid width", distance_mc_grid_width,
                      "the markov chain hash grid width in pixels");
    config.config_uint("dist mc states per vertex", distance_mc_vertex_state_count, 1,
                       MAX_DISTANCE_MC_VERTEX_STATE_COUNT,
                       "number of markov chain states per vertex");
    config.config_float("particle size", volume_particle_size_um, "in mircometer (5-50)", 0.1);
    config.config_percent("dist guide p", dist_guide_p, "higher means more distance guiding");
    config.config_float("volume max t", volume_max_t);
    config.config_percent("Phase Prob", volume_phase_p,
                          "the probability to use phase function sampling");
    config.config_bool("volume forward project", volume_forward_project);

    pc.rt_config.flags = 0;

    config.st_separate("Reproducibility");
    config.config_bool("reproducible renders", reproducible_renders,
                       "e.g. disables random behavior");

    config.st_separate("Light cache");
    config.config_bool("surf: use LC", use_light_cache_tail,
                       "use the light cache for the path tail");
    config.config_bool("volume: use LC", volume_use_light_cache,
                       "query light cache for non-emitting surfaces");
    config.config_float("LC levels", light_cache_levels);
    config.config_float("LC tan(alpha/2)", light_cache_tan_alpha_half,
                        "the light cache resolution, lower means higher resolution.", 0.0001);
    config.config_uint("LC buf size", light_cache_buffer_size,
                       "Size of buffer backing the hash grid");

    config.st_separate("Debug");
    config.config_bool("overwrite sun", overwrite_sun);
    if (overwrite_sun) {
        config.config_float3("sun dir", &overwrite_sun_dir.x);
        config.config_float3("sun col", &overwrite_sun_col.x);
    }
    config.config_bool("overwrite mu_t/s", mu_t_s_overwrite);
    if (mu_t_s_overwrite) {
        config.config_float("mu_t", mu_t, "", 0.000001);
        config.config_float3("mu_s / mu_t", &mu_s_div_mu_t.x);
    } else {
        config.output_text(fmt::format("mu_t: {}\nmu_s: ({}, {}, {})", pc.cam_x_mu_t.a,
                                       pc.prev_cam_x_mu_sx.a, pc.prev_cam_w_mu_sy.a,
                                       pc.prev_cam_u_mu_sz.a));
    }

    dump_mc = config.config_bool("Download 128MB MC states",
                                 "Dumps the states as json into mc_dump.json");

    // Only require a pipeline recreation
    if (old_spp != spp || old_max_path_lenght != max_path_length ||
        old_use_light_cache_tail != use_light_cache_tail ||
        old_adaptive_sampling != adaptive_sampling || old_volume_spp != volume_spp ||
        old_volume_use_light_cache != volume_use_light_cache ||
        old_volume_particle_size_um != volume_particle_size_um || old_mc_samples != mc_samples ||
        old_mc_samples_adaptive_prob != mc_samples_adaptive_prob ||
        old_distance_mc_samples != distance_mc_samples || old_overwrite_sun != overwrite_sun ||
        old_overwrite_sun_dir != overwrite_sun_dir || old_overwrite_sun_col != overwrite_sun_col ||
        old_mc_fast_recovery != mc_fast_recovery || old_light_cache_levels != light_cache_levels ||
        old_light_cache_tan_alpha_half != light_cache_tan_alpha_half ||
        old_mc_adaptive_grid_tan_alpha_half != mc_adaptive_grid_tan_alpha_half ||
        old_mc_adaptive_grid_levels != mc_adaptive_grid_levels ||
        old_volume_max_t != volume_max_t || old_surf_bsdf_p != surf_bsdf_p ||
        old_volume_phase_p != volume_phase_p || old_dir_guide_prior != dir_guide_prior ||
        old_dist_guide_p != dist_guide_p || old_seed != seed ||
        old_randomize_seed != randomize_seed) {
        pipe.reset();
    }

    // Change outputs and require a graph rebuild
    if (old_render_width != render_width || old_render_height != render_height ||
        old_mc_adaptive_buffer_size != mc_adaptive_buffer_size ||
        old_mc_static_buffer_size != mc_static_buffer_size ||
        old_mc_static_grid_width != mc_static_grid_width ||
        old_distance_mc_grid_width != distance_mc_grid_width ||
        old_light_cache_buffer_size != light_cache_buffer_size ||
        old_distance_mc_vertex_state_count != distance_mc_vertex_state_count) {
        return NEEDS_RECONNECT;
    }

    return {};
}
