#include "quake/quake_node.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "merian/utils/bitpacking.hpp"
#include "merian/utils/colors.hpp"
#include "merian/utils/glm.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/string.hpp"
#include "merian/utils/threads.hpp"
#include "merian/utils/xorshift.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "merian/vk/utils/math.hpp"

static const uint32_t spv[] = {
#include "quake.comp.spv.h"
};

static const uint32_t clear_spv[] = {
#include "clear.comp.spv.h"
};

struct QuakeData {
    // The first quake node sets this
    // If this is not null we won't allow new
    // Quake nodes
    QuakeNode* node{nullptr};
    quakeparms_t params;
};

extern "C" {

#include "bgmusic.h"
#include "quakedef.h"

// Quake uses lots of static global variables,
// so we need to do that to
// e.g. to make sure there is only one QuakeNode.
static QuakeData quake_data;
// from r_alias.c
extern float r_avertexnormals[162][3];
extern particle_t* active_particles;
extern cvar_t cl_maxpitch; // johnfitz -- variable pitch clamping
extern cvar_t cl_minpitch; // johnfitz -- variable pitch clamping
}
// CALLBACKS from within Quake --------------------------------------------------------------------

// called each time a new map is (re)loaded
extern "C" void QS_worldspawn() {
    quake_data.node->QS_worldspawn();
}

// called when a texture should be loaded
extern "C" void QS_texture_load(gltexture_t* glt, uint32_t* data) {
    quake_data.node->QS_texture_load(glt, data);
}

// called from within qs, pretty much a copy from in_sdl.c:
extern "C" void IN_Move(usercmd_t* cmd) {
    quake_data.node->IN_Move(cmd);
}

// Helper -----------------------------------------------------------------------------------------

static merian::TextureHandle make_rgb8_texture(const vk::CommandBuffer cmd,
                                               const merian::ResourceAllocatorHandle& allocator,
                                               const std::vector<uint32_t>& data,
                                               uint32_t width,
                                               uint32_t height,
                                               bool force_linear_filter = false,
                                               bool srgb = true) {
    static vk::ImageCreateInfo tex_image_info{
        {},
        vk::ImageType::e2D,
        vk::Format::eR8G8B8A8Srgb,
        {0, 0, 1},
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    static vk::ImageViewCreateInfo tex_view_info{
        {},
        VK_NULL_HANDLE,
        vk::ImageViewType::e2D,
        vk::Format::eR8G8B8A8Srgb,
        {},
        merian::first_level_and_layer(),
    };

    tex_image_info.extent.width = width;
    tex_image_info.extent.height = height;

    if (srgb) {
        tex_image_info.format = vk::Format::eR8G8B8A8Srgb;
        tex_view_info.format = vk::Format::eR8G8B8A8Srgb;
    } else {
        tex_image_info.format = vk::Format::eR8G8B8A8Unorm;
        tex_view_info.format = vk::Format::eR8G8B8A8Unorm;
    }

    merian::ImageHandle image =
        allocator->createImage(cmd, data.size() * sizeof(uint32_t), data.data(), tex_image_info);
    tex_view_info.image = *image;
    merian::TextureHandle tex = allocator->createTexture(image, tex_view_info);
    if (force_linear_filter)
        tex->attach_sampler(allocator->get_sampler_pool()->linear_repeat());
    else
        tex->attach_sampler(allocator->get_sampler_pool()->nearest_repeat());
    return tex;
}

void init_quake(const char* base_dir) {
    const char* argv[] = {"quakespasm", "-basedir", base_dir, "+skill", "2"};
    const int argc = 5;

    quake_data.params.basedir = base_dir; // does not work
    quake_data.params.argc = argc;
    quake_data.params.argv = (char**)argv;
    quake_data.params.errstate = 0;
    quake_data.params.memsize = 256 * 1024 * 1024; // qs default in 0.94.3
    quake_data.params.membase = malloc(quake_data.params.memsize);

    srand(1337); // quake uses this
    COM_InitArgv(quake_data.params.argc, quake_data.params.argv);
    Sys_Init();

    Sys_Printf("Quake %1.2f (c) id Software\n", VERSION);
    Sys_Printf("GLQuake %1.2f (c) id Software\n", GLQUAKE_VERSION);
    Sys_Printf("FitzQuake %1.2f (c) John Fitzgibbons\n", FITZQUAKE_VERSION);
    Sys_Printf("FitzQuake SDL port (c) SleepwalkR, Baker\n");
    Sys_Printf("QuakeSpasm " QUAKESPASM_VER_STRING " (c) Ozkan Sezer, Eric Wasylishen & others\n");

    Sys_Printf("Host_Init\n");
    Host_Init(); // this one has a lot of meat! we'll need to cut it short i suppose

    // S_BlockSound(); // only start when grabbing
    // close menu because we don't have an esc key:
    key_dest = key_game;
    m_state = m_none;
    IN_Activate();
}

void deinit_quake() {
    free(quake_data.params.membase);
}

uint16_t
make_texnum_alpha(gltexture_s* tex, entity_t* entity = nullptr, msurface_t* surface = nullptr) {
    uint16_t result = tex->texnum;
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
                   std::vector<uint32_t>& idx,
                   std::vector<QuakeNode::VertexExtraData>& ext,
                   const uint32_t texnum_blood,
                   const uint32_t texnum_explosion) {
    merian::XORShift32 xrand{1337};

    static const float voff[4][3] = {
        {0.0, 1.0, 0.0},
        {-0.5, -0.5, -0.87},
        {-0.5, -0.5, 0.87},
        {1.0, -0.5, 0.0},
    };

    for (particle_t* p = active_particles; p; p = p->next) {
        uint8_t* c = (uint8_t*)&d_8to24table[(int)p->color]; // that would be the colour. pick
                                                             // texture based on this:
        // smoke is r=g=b
        // blood is g=b=0
        // rocket trails are r=2*g=2*b a bit randomised
        const uint16_t tex_col = c[1] == 0 && c[2] == 0 ? texnum_blood : texnum_explosion;
        const uint16_t tex_lum = c[1] == 0 && c[2] == 0 ? 0 : texnum_explosion;

        float vert[4][3];
        for (int l = 0; l < 3; l++) {
            float off = 2 * (xrand.get() - 0.5) + 2 * (xrand.get() - 0.5);
            for (int k = 0; k < 4; k++)
                vert[k][l] =
                    p->org[l] + off + 2 * voff[k][l] + (xrand.get() - 0.5) + (xrand.get() - 0.5);
        }

        const uint32_t vtx_cnt = vtx.size() / 3;
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[0][l]);
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[1][l]);
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[2][l]);
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[3][l]);

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
        idx.emplace_back(vtx_cnt + 2);
        idx.emplace_back(vtx_cnt + 3);

        for (int k = 0; k < 4; k++) {
            ext.emplace_back(tex_col, tex_lum, 0, 0, 0, merian::float_to_half_aprox(0),
                             merian::float_to_half_aprox(1), merian::float_to_half_aprox(0),
                             merian::float_to_half_aprox(0), merian::float_to_half_aprox(1),
                             merian::float_to_half_aprox(0));
        }
    }
}

void add_geo_alias(entity_t* ent,
                   qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<QuakeNode::VertexExtraData>& ext) {
    assert(m->type == mod_alias);

    // TODO: e->model->flags & MF_HOLEY <= enable alpha test
    // fprintf(stderr, "alias origin and angles %g %g %g -- %g %g %g\n",
    //     ent->origin[0], ent->origin[1], ent->origin[2],
    //     ent->angles[0], ent->angles[1], ent->angles[2]);
    static std::mutex quake_mutex;
    quake_mutex.lock();
    aliashdr_t* hdr = (aliashdr_t*)Mod_Extradata(ent->model);
    quake_mutex.unlock();
    aliasmesh_t* desc = (aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
    // the plural here really hurts but it's from quakespasm code:
    int16_t* indexes = (int16_t*)((uint8_t*)hdr + hdr->indexes);
    trivertx_t* trivertexes = (trivertx_t*)((uint8_t*)hdr + hdr->vertexes);

    // for(int f = 0; f < hdr->numposes; f++)
    // TODO: upload all vertices so we can just alter the indices on gpu
    int f = ent->frame;
    if (f < 0 || f >= hdr->numposes)
        return;

    lerpdata_t lerpdata;
    R_SetupAliasFrame(ent, hdr, ent->frame, &lerpdata);
    R_SetupEntityTransform(ent, &lerpdata);
    // angles: pitch yaw roll. axes: right fwd up
    lerpdata.angles[0] *= -1;
    glm::mat3 mat_model;
    glm::vec3 pos_pose1, pos_pose2;

    AngleVectors(lerpdata.angles, &mat_model[0].x, &mat_model[1].x, &mat_model[2].x);
    mat_model[1] *= -1;

    const glm::mat3 mat_model_inv_t = glm::transpose(glm::inverse(mat_model));

    uint32_t vtx_cnt = vtx.size() / 3;
    for (int v = 0; v < hdr->numverts_vbo; v++) {
        int i_pose1 = hdr->numverts * lerpdata.pose1 + desc[v].vertindex;
        int i_pose2 = hdr->numverts * lerpdata.pose2 + desc[v].vertindex;
        // get model pos
        for (int k = 0; k < 3; k++) {
            pos_pose1[k] = trivertexes[i_pose1].v[k] * hdr->scale[k] + hdr->scale_origin[k];
            pos_pose2[k] = trivertexes[i_pose2].v[k] * hdr->scale[k] + hdr->scale_origin[k];
        }
        // convert to world space

        const glm::vec3 world_pos = *merian::as_vec3(lerpdata.origin) +
                                    mat_model * glm::mix(pos_pose1, pos_pose2, lerpdata.blend);
        for (int k = 0; k < 3; k++)
            vtx.emplace_back(world_pos[k]);
    }

    for (int i = 0; i < hdr->numindexes; i++)
        idx.emplace_back(vtx_cnt + indexes[i]);

    // normals for each vertex from above
    uint32_t* tmpn = (uint32_t*)alloca(sizeof(uint32_t) * hdr->numverts_vbo);
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
        *(tmpn + v) = merian::encode_normal(world_n);
    }

    // add extra data for each primitive
    for (int i = 0; i < hdr->numindexes / 3; i++) {
        const int sk = CLAMP(0, ent->skinnum, hdr->numskins - 1), fm = ((int)(cl.time * 10)) & 3;
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
            n0 = *(tmpn + indexes[3 * i + 0]);
            n1 = *(tmpn + indexes[3 * i + 1]);
            n2 = *(tmpn + indexes[3 * i + 2]);
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

void add_geo_brush(entity_t* ent,
                   qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<QuakeNode::VertexExtraData>& ext) {
    assert(m->type == mod_brush);

    // fprintf(stderr, "brush origin and angles %g %g %g -- %g %g %g with %d surfs\n",
    //     ent->origin[0], ent->origin[1], ent->origin[2],
    //     ent->angles[0], ent->angles[1], ent->angles[2],
    //     m->nummodelsurfaces);
    float angles[3] = {-ent->angles[0], ent->angles[1], ent->angles[2]};
    glm::mat3 mat_model;
    AngleVectors(angles, &mat_model[0].x, &mat_model[1].x, &mat_model[2].x);
    mat_model[1] *= -1;

    for (int i = 0; i < m->nummodelsurfaces; i++) {
        msurface_t* surf = &m->surfaces[m->firstmodelsurface + i];
        if (!strcmp(surf->texinfo->texture->name, "skip"))
            continue;

        glpoly_t* p = surf->polys;
        while (p) {
            uint32_t vtx_cnt = vtx.size() / 3;
            for (int k = 0; k < p->numverts; k++) {
                glm::vec3 coord =
                    mat_model * (*merian::as_vec3(p->verts[k])) + (*merian::as_vec3(ent->origin));

                for (int l = 0; l < 3; l++) {
                    vtx.emplace_back(coord[l]);
                }
            }

            for (int k = 2; k < p->numverts; k++) {
                idx.emplace_back(vtx_cnt);
                idx.emplace_back(vtx_cnt + k - 1);
                idx.emplace_back(vtx_cnt + k);
            }

            // TODO: make somehow dynamic. don't want to re-upload the whole model just because
            // the texture animates. for now that means static brush models will not actually
            // animate their textures
            texture_t* t = R_TextureAnimation(surf->texinfo->texture, ent->frame);

            for (int k = 2; k < p->numverts; k++) {
                QuakeNode::VertexExtraData extra{
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
                    qmodel_t* m,
                    std::vector<float>& vtx,
                    std::vector<uint32_t>& idx,
                    std::vector<QuakeNode::VertexExtraData>& ext) {
    assert(m->type == mod_sprite);

    vec3_t point, v_forward, v_right, v_up;
    msprite_t* psprite;
    mspriteframe_t* frame;
    float *s_up, *s_right;
    float angle, sr, cr;
    // XXX newer quakespasm has this: ENTSCALE_DECODE(ent->scale);
    float scale = 1.0f;

    vec3_t vpn, vright, vup, r_origin;
    VectorCopy(r_refdef.vieworg, r_origin);
    AngleVectors(r_refdef.viewangles, vpn, vright, vup);

    frame = R_GetSpriteFrame(ent);
    psprite = (msprite_t*)ent->model->cache.data;

    switch (psprite->type) {
    case SPR_VP_PARALLEL_UPRIGHT: // faces view plane, up is towards the heavens
        v_up[0] = 0;
        v_up[1] = 0;
        v_up[2] = 1;
        s_up = v_up;
        s_right = vright;
        break;
    case SPR_FACING_UPRIGHT: // faces camera origin, up is towards the heavens
        VectorSubtract(ent->origin, r_origin, v_forward);
        v_forward[2] = 0;
        VectorNormalizeFast(v_forward);
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
        AngleVectors(ent->angles, v_forward, v_right, v_up);
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

    // add three quads
    for (int k = 0; k < 3; k++) {
        float vert[4][3];

        vec3_t front;
        CrossProduct(s_up, s_right, front);
        VectorMA(ent->origin, frame->down * scale, k == 1 ? front : s_up, point);
        VectorMA(point, frame->left * scale, k == 2 ? front : s_right, point);
        for (int l = 0; l < 3; l++)
            vert[0][l] = point[l];

        VectorMA(ent->origin, frame->up * scale, k == 1 ? front : s_up, point);
        VectorMA(point, frame->left * scale, k == 2 ? front : s_right, point);
        for (int l = 0; l < 3; l++)
            vert[1][l] = point[l];

        VectorMA(ent->origin, frame->up * scale, k == 1 ? front : s_up, point);
        VectorMA(point, frame->right * scale, k == 2 ? front : s_right, point);
        for (int l = 0; l < 3; l++)
            vert[2][l] = point[l];

        VectorMA(ent->origin, frame->down * scale, k == 1 ? front : s_up, point);
        VectorMA(point, frame->right * scale, k == 2 ? front : s_right, point);
        for (int l = 0; l < 3; l++)
            vert[3][l] = point[l];

        // add vertices
        uint32_t vtx_cnt = vtx.size() / 3;
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[0][l]);
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[1][l]);
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[2][l]);
        for (int l = 0; l < 3; l++)
            vtx.emplace_back(vert[3][l]);

        // add index
        idx.emplace_back(vtx_cnt);
        idx.emplace_back(vtx_cnt + 2 - 1);
        idx.emplace_back(vtx_cnt + 2);

        idx.emplace_back(vtx_cnt);
        idx.emplace_back(vtx_cnt + 3 - 1);
        idx.emplace_back(vtx_cnt + 3);

        // add extra data
        float n[3],
            e0[] = {vert[2][0] - vert[0][0], vert[2][1] - vert[0][1], vert[2][2] - vert[0][2]},
            e1[] = {vert[1][0] - vert[0][0], vert[1][1] - vert[0][1], vert[1][2] - vert[0][2]};
        CrossProduct(e0, e1, n);
        uint32_t n_enc = merian::encode_normal(n);

        uint16_t texnum = 0;
        if (frame->gltexture) {
            texnum = make_texnum_alpha(frame->gltexture);
        }

        ext.emplace_back(texnum,
                         texnum, // sprite allways emits
                         n_enc, n_enc, n_enc, merian::float_to_half_aprox(0),
                         merian::float_to_half_aprox(1), merian::float_to_half_aprox(0),
                         merian::float_to_half_aprox(0), merian::float_to_half_aprox(1),
                         merian::float_to_half_aprox(0));
        ext.emplace_back(texnum,
                         texnum, // sprite allways emits
                         n_enc, n_enc, n_enc, merian::float_to_half_aprox(0),
                         merian::float_to_half_aprox(1), merian::float_to_half_aprox(1),
                         merian::float_to_half_aprox(0), merian::float_to_half_aprox(1),
                         merian::float_to_half_aprox(1));

    } // end three axes
}

// Adds the geo from entity into the vectors.
void add_geo(entity_t* ent,
             std::vector<float>& vtx,
             std::vector<uint32_t>& idx,
             std::vector<QuakeNode::VertexExtraData>& ext) {
    if (!ent)
        return;
    qmodel_t* m = ent->model;
    if (!m)
        return;

    if (m->type == mod_alias) { // alias model:
        add_geo_alias(ent, m, vtx, idx, ext);
        assert(ext.size() == idx.size() / 3);
        assert(vtx.size() % 3 == 0);
    } else if (m->type == mod_brush) { // brush model:
        add_geo_brush(ent, m, vtx, idx, ext);
        assert(ext.size() == idx.size() / 3);
        assert(vtx.size() % 3 == 0);
    } else if (m->type == mod_sprite) {
        // explosions, decals, etc, this is R_DrawSpriteModel
        add_geo_sprite(ent, m, vtx, idx, ext);
        assert(ext.size() == idx.size() / 3);
        assert(vtx.size() % 3 == 0);
    }
}

// If the supplied buffer is not nullptr and is large enough, it is returned and an upload is
// recorded. A new larger buffer is used otherwise.
template <typename T>
merian::BufferHandle
ensure_buffer(const merian::ResourceAllocatorHandle& allocator,
              const vk::BufferUsageFlags usage,
              const vk::CommandBuffer& cmd,
              const std::vector<T>& data,
              const merian::BufferHandle optional_buffer,
              const std::optional<vk::DeviceSize> min_alignment = std::nullopt) {
    merian::BufferHandle buffer;
    if (optional_buffer && optional_buffer->get_size() >= sizeof(T) * data.size()) {
        buffer = optional_buffer;
    } else {
        // Make it a bit larger for logarithmic growth
        buffer = allocator->createBuffer(sizeof(T) * data.size() * 1.25, usage, merian::NONE, {},
                                         min_alignment);
    }
    allocator->getStaging()->cmdToBuffer(cmd, *buffer, 0, sizeof(T) * data.size(), data.data());
    return buffer;
}

// Creates a vertex and index buffer for rt on the device and records the upload.
// If the supplied buffers are not nullptr and are large enough, they are returned and an upload is
// recorded. Returns (vertex_buffer, index_buffer).
// Appropriate barriers are inserted.
std::tuple<merian::BufferHandle, merian::BufferHandle, merian::BufferHandle>
ensure_vertex_index_ext_buffer(const merian::ResourceAllocatorHandle& allocator,
                               const vk::CommandBuffer& cmd,
                               const std::vector<float>& vtx,
                               const std::vector<uint32_t>& idx,
                               const std::vector<QuakeNode::VertexExtraData>& ext,
                               const merian::BufferHandle optional_vtx_buffer,
                               const merian::BufferHandle optional_idx_buffer,
                               const merian::BufferHandle optional_ext_buffer) {
    auto usage_rt = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    auto usage_storage = vk::BufferUsageFlagBits::eStorageBuffer;

    merian::BufferHandle vertex_buffer =
        ensure_buffer(allocator, usage_rt, cmd, vtx, optional_vtx_buffer);
    merian::BufferHandle index_buffer =
        ensure_buffer(allocator, usage_rt, cmd, idx, optional_idx_buffer);
    merian::BufferHandle ext_buffer =
        ensure_buffer(allocator, usage_storage, cmd, ext, optional_ext_buffer);

    const std::array<vk::BufferMemoryBarrier2, 3> barriers = {
        vertex_buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                       vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       vk::AccessFlagBits2::eTransferWrite,
                                       vk::AccessFlagBits2::eAccelerationStructureReadKHR |
                                           vk::AccessFlagBits2::eShaderRead),
        index_buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                      vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                      vk::AccessFlagBits2::eTransferWrite,
                                      vk::AccessFlagBits2::eAccelerationStructureReadKHR |
                                          vk::AccessFlagBits2::eShaderRead),
        ext_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eShaderRead),
    };

    vk::DependencyInfo dep_info{{}, {}, barriers, {}};
    cmd.pipelineBarrier2(dep_info);

    return std::make_tuple(vertex_buffer, index_buffer, ext_buffer);
}

// QuakeNode
// --------------------------------------------------------------------------------------

QuakeNode::QuakeNode(const merian::SharedContext& context,
                     const merian::ResourceAllocatorHandle& allocator,
                     const std::shared_ptr<merian::InputController> controller,
                     const uint32_t frames_in_flight,
                     const char* base_dir)
    : context(context), allocator(allocator), frames(frames_in_flight) {

    // QUAKE INIT
    if (quake_data.node) {
        throw std::runtime_error{"Only one quake node can be created."};
    }
    quake_data.node = this;
    host_parms = &quake_data.params;
    init_quake(base_dir);

    // PIPELINE CREATION
    shader = std::make_shared<merian::ShaderModule>(context, sizeof(spv), spv);

    quake_desc_set_layout =
        merian::DescriptorSetLayoutBuilder()
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, GEO_DESC_ARRAY_SIZE)
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, GEO_DESC_ARRAY_SIZE)
            .add_binding_storage_buffer(vk::ShaderStageFlagBits::eCompute, GEO_DESC_ARRAY_SIZE)
            .add_binding_combined_sampler(vk::ShaderStageFlagBits::eCompute, MAX_GLTEXTURES)
            .add_binding_acceleration_structure()
            .build_layout(context);
    quake_pool = std::make_shared<merian::DescriptorPool>(quake_desc_set_layout, frames_in_flight);
    for (auto& frame : frames) {
        frame.quake_sets = std::make_shared<merian::DescriptorSet>(quake_pool);
        frame.blas_builder = std::make_unique<merian::BLASBuilder>(context, allocator);
        frame.tlas_builder = std::make_unique<merian::TLASBuilder>(context, allocator);
    }

    binding_dummy_buffer = allocator->createBuffer(8, vk::BufferUsageFlagBits::eStorageBuffer);

    // clang-format off
    controller->set_key_event_callback([&](merian::InputController& controller, int key, int scancode, merian::InputController::KeyStatus action, int){
        if(key >= 65 && key <= 90) key |= 32;
        if (action == merian::InputController::PRESS) {
            Key_Event(key, true);
        } else if (action == merian::InputController::RELEASE) {
            Key_Event(key, false);
        }
        if (scancode == 1) { // ESCAPE
            controller.request_raw_mouse_input(false);
        }
    });
    controller->set_mouse_cursor_callback([&](merian::InputController& controller, double xpos, double ypos){
        if (controller.get_raw_mouse_input()) {
            this->mouse_x = xpos;
            this->mouse_y = ypos;
        }
    });
    controller->set_mouse_button_callback([&](merian::InputController& controller, merian::InputController::MouseButton button, merian::InputController::KeyStatus status, int){
        if (button == merian::InputController::MOUSE1) {
            if (!controller.get_raw_mouse_input()) {
                controller.request_raw_mouse_input(true);
                this->mouse_oldx = this->mouse_x;
                this->mouse_oldy = this->mouse_y;
                return;
            }
        }
        const int remap[] = {K_MOUSE1, K_MOUSE2, K_MOUSE3, K_MOUSE4, K_MOUSE5};
        Key_Event(remap[button], status == merian::InputController::PRESS);
    });
    controller->set_scroll_event_callback([&](merian::InputController&, double xoffset, double yoffset){
        if (yoffset > 0) {
            Key_Event(K_MWHEELUP, true);
            Key_Event(K_MWHEELUP, false);
        } else if (xoffset < 0) {
            Key_Event(K_MWHEELDOWN, true);
            Key_Event(K_MWHEELDOWN, false);
        }
    });
    // clang-format on

    audio_device = std::make_unique<merian::SDLAudioDevice>(
        merian::SDLAudioDevice::FORMAT_S16_LSB, [](uint8_t* stream, int len) {
            // from
            // https://github.com/sezero/quakespasm/blob/70df2b661e9c632d04825b259e63ad58c29c01ac/Quake/snd_sdl.c#L156
            int buffersize = shm->samples * (shm->samplebits / 8);
            int pos, tobufend;
            int len1, len2;

            if (!shm) { /* shouldn't happen, but just in case */
                memset(stream, 0, len);
                return;
            }

            pos = (shm->samplepos * (shm->samplebits / 8));
            if (pos >= buffersize)
                shm->samplepos = pos = 0;

            tobufend = buffersize - pos; /* bytes to buffer's end. */
            len1 = len;
            len2 = 0;

            if (len1 > tobufend) {
                len1 = tobufend;
                len2 = len - len1;
            }

            memcpy(stream, shm->buffer + pos, len1);

            if (len2 <= 0) {
                shm->samplepos += (len1 / (shm->samplebits / 8));
            } else { /* wraparound? */
                memcpy(stream + len1, shm->buffer, len2);
                shm->samplepos = (len2 / (shm->samplebits / 8));
            }

            if (shm->samplepos >= buffersize)
                shm->samplepos = 0;
        });
    if (sound)
        audio_device->unpause_audio();
}

QuakeNode::~QuakeNode() {
    deinit_quake();
}

// -------------------------------------------------------------------------------------------

void QuakeNode::QS_worldspawn() {
    SPDLOG_DEBUG("worldspawn");
    worldspawn = true;
}

void QuakeNode::QS_texture_load(gltexture_t* glt, uint32_t* data) {
    // LOG -----------------------------------------------

    std::string source = strcmp(glt->source_file, "") == 0 ? "memory" : glt->source_file;
    SPDLOG_DEBUG("texture_load {} {} {}x{} from {}, frame: {}", glt->texnum, glt->name, glt->width,
                 glt->height, source, glt->visframe);

    if (glt->width == 0 || glt->height == 0) {
        SPDLOG_WARN("image extent was 0. skipping");
        return;
    }

    // STORE SOME TEXTURE IDs ----------------------------

    // HACK: for blood patch
    if (!strcmp(glt->name, "progs/gib_1.mdl:frame0"))
        texnum_blood = glt->texnum;
    // HACK: for sparks and for emissive rocket particle trails
    if (!strcmp(glt->name, "progs/s_exp_big.spr:frame10"))
        texnum_explosion = glt->texnum;

    // classic quake sky
    if (merian::ends_with(glt->name, "_front"))
        texnum_skybox[1] = glt->texnum;
    if (merian::ends_with(glt->name, "_back"))
        texnum_skybox[0] = glt->texnum;

    // full featured cube map/arcane dimensions
    if (merian::starts_with(glt->name, "gfx/env/")) {
        if (merian::ends_with(glt->name, "_rt"))
            texnum_skybox[0] = glt->texnum;
        if (merian::ends_with(glt->name, "_bk"))
            texnum_skybox[1] = glt->texnum;
        if (merian::ends_with(glt->name, "_lf"))
            texnum_skybox[2] = glt->texnum;
        if (merian::ends_with(glt->name, "_ft"))
            texnum_skybox[3] = glt->texnum;
        if (merian::ends_with(glt->name, "_up"))
            texnum_skybox[4] = glt->texnum;
        if (merian::ends_with(glt->name, "_dn"))
            texnum_skybox[5] = glt->texnum;
    }

    // ALLOCATE ----------------------------

    // We store the texture on system memory for now
    // and upload in cmd_process later
    std::shared_ptr<QuakeTexture> texture = std::make_shared<QuakeTexture>(glt, data);
    // If we replace an existing texture the old texture is automatically freed.
    // TODO: Multiple frames in flight? (For each in flight an array, then copy from last, and
    // replace pending?)
    current_textures[glt->texnum] = texture;
    pending_uploads.insert(glt->texnum);
}

void QuakeNode::IN_Move(usercmd_t* cmd) {
    SPDLOG_TRACE("move");

    // pretty much a copy from in_sdl.c:

    int dmx = (this->mouse_x - this->mouse_oldx) * sensitivity.value;
    int dmy = (this->mouse_y - this->mouse_oldy) * sensitivity.value;
    this->mouse_oldx = this->mouse_x;
    this->mouse_oldy = this->mouse_y;

    if ((in_strafe.state & 1) || (lookstrafe.value && (in_mlook.state & 1)))
        cmd->sidemove += m_side.value * dmx;
    else
        cl.viewangles[YAW] -= m_yaw.value * dmx;

    if (in_mlook.state & 1) {
        if (dmx || dmy)
            V_StopPitchDrift();
    }

    if ((in_mlook.state & 1) && !(in_strafe.state & 1)) {
        cl.viewangles[PITCH] += m_pitch.value * dmy;
        /* johnfitz -- variable pitch clamping */
        if (cl.viewangles[PITCH] > cl_maxpitch.value)
            cl.viewangles[PITCH] = cl_maxpitch.value;
        if (cl.viewangles[PITCH] < cl_minpitch.value)
            cl.viewangles[PITCH] = cl_minpitch.value;
    } else {
        if ((in_strafe.state & 1) && noclip_anglehack)
            cmd->upmove -= m_forward.value * dmy;
        else
            cmd->forwardmove -= m_forward.value * dmy;
    }
}

// -------------------------------------------------------------------------------------------

std::tuple<std::vector<merian::NodeInputDescriptorImage>,
           std::vector<merian::NodeInputDescriptorBuffer>>
QuakeNode::describe_inputs() {
    return {
        {
            merian::NodeInputDescriptorImage::compute_read("gbuffer", 1),
            merian::NodeInputDescriptorImage::compute_read("mv", 0),
            merian::NodeInputDescriptorImage::compute_read("blue_noise", 0),
            merian::NodeInputDescriptorImage::compute_read("nee_in", 1),
        },
        {},
    };
}

std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
           std::vector<merian::NodeOutputDescriptorBuffer>>
QuakeNode::describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                            const std::vector<merian::NodeOutputDescriptorBuffer>&) {

    return {
        {
            merian::NodeOutputDescriptorImage::compute_write(
                "irradiance", vk::Format::eR16G16B16A16Sfloat, width, height),
            merian::NodeOutputDescriptorImage::compute_write(
                "albedo", vk::Format::eR16G16B16A16Sfloat, width, height),
            merian::NodeOutputDescriptorImage::compute_write(
                "gbuffer", vk::Format::eR32G32B32A32Sfloat, width, height),
            merian::NodeOutputDescriptorImage::compute_write(
                "nee_out", vk::Format::eR32G32B32A32Uint, width, height),
        },
        {},
    };
}

void QuakeNode::cmd_build(const vk::CommandBuffer& cmd,
                          const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                          const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                          const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                          const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs) {
    // GRAPH DESC SETS
    std::tie(graph_textures, graph_sets, graph_pool, graph_desc_set_layout) =
        merian::make_graph_descriptor_sets(context, allocator, image_inputs, buffer_inputs,
                                           image_outputs, buffer_outputs, graph_desc_set_layout);
    if (!pipe) {
        auto pipe_layout = merian::PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(graph_desc_set_layout)
                               .add_descriptor_set_layout(quake_desc_set_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = merian::SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, shader, spec_builder.build());
    }

    // DUMMY IMAGE as placeholder
    uint32_t missing_rgba = merian::uint32_from_rgba(1, 0, 1, 1);
    binding_dummy_image = make_rgb8_texture(
        cmd, allocator, {missing_rgba, missing_rgba, missing_rgba, missing_rgba}, 2, 2);

    // MAKE SURE ALL DESCRIPTORS ARE SET
    for (auto& frame : frames) {
        merian::DescriptorSetUpdate update(frame.quake_sets);
        for (uint32_t texnum = 0; texnum < MAX_GLTEXTURES; texnum++) {
            if (current_textures[texnum] && current_textures[texnum]->gpu_tex) {
                auto& tex = current_textures[texnum];
                update.write_descriptor_texture(BINDING_IMG_TEX, tex->gpu_tex, texnum);
            } else {
                update.write_descriptor_texture(BINDING_IMG_TEX, binding_dummy_image, texnum);
            }
        }
        update.update(context);
    }
}

void QuakeNode::cmd_process(const vk::CommandBuffer& cmd,
                            merian::GraphRun& run,
                            const uint32_t graph_set_index,
                            const std::vector<merian::ImageHandle>& image_inputs,
                            const std::vector<merian::BufferHandle>& buffer_inputs,
                            const std::vector<merian::ImageHandle>& image_outputs,
                            const std::vector<merian::BufferHandle>& buffer_outputs) {

    if (run.get_iteration() == 0) {
        // TODO: Clear output and feedback buffers
    }

    // UPDATE GAMESTATE (if not paused)
    if (!pause) {
        MERIAN_PROFILE_SCOPE(run.get_profiler(), "update gamestate");
        if (!pending_commands.empty()) {
            // only one command between each HostFrame
            Cmd_ExecuteString(pending_commands.front().c_str(), src_command);
            pending_commands.pop();
        }

        double newtime = Sys_DoubleTime();
        // Use (1. / 60.) else the Host_Frame() does nothing and sv_player is not valid below
        double time = old_time == 0 ? 1. / 60. : newtime - old_time;
        Host_Frame(time);
        // init some left/right vectors also used for sound
        R_SetupView();
        old_time = newtime;
    }

    if (!sv_player) {
        // TODO: Clear output and feedback buffers
        return;
    }

    // UPDATE PUSH CONSTANT (with player data)
    pc.frame = frame;
    pc.player.health = sv_player->v.health;
    pc.player.armor = sv_player->v.armorvalue;
    pc.player.flags = 0;
    pc.player.flags |= sv_player->v.weapon == 1 ? PLAYER_FLAGS_TORCH : 0; // shotgun has torch
    pc.player.flags |= sv_player->v.waterlevel >= 3 ? PLAYER_FLAGS_UNDERWATER : 0;
    pc.sky = texnum_skybox;
    pc.cl_time = cl.time;
    float rgt[3];
    AngleVectors(r_refdef.viewangles, &pc.cam_w.x, rgt, &pc.cam_u.x);
    pc.cam_x = glm::vec4(*merian::as_vec3(r_refdef.vieworg), 1);
    float fog_density = Fog_GetDensity();
    fog_density *= fog_density;
    pc.fog = glm::vec4(*merian::as_vec3(Fog_GetColor()), fog_density);

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update geo");
        if (worldspawn) {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "upload static");
            update_static_geo(cmd, true);
            worldspawn = false;
        } else {
            update_static_geo(cmd, false);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "upload dynamic");
            update_dynamic_geo(cmd);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "build acceleration structures");
            update_as(cmd, run.get_profiler());
        }
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update textures");
        update_textures(cmd);
    }

    FrameData& cur_frame = current_frame();
    if (!cur_frame.tlas) {
        // TODO: Clear output and feedback buffers
        return;
    }

    // BIND PIPELINE
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "quake.comp");
        pipe->bind(cmd);
        pipe->bind_descriptor_set(cmd, graph_sets[graph_set_index]);
        pipe->bind_descriptor_set(cmd, cur_frame.quake_sets, 1);
        pipe->push_constant(cmd, pc);
        cmd.dispatch((width + local_size_x - 1) / local_size_x,
                     (height + local_size_y - 1) / local_size_y, 1);
    }
    frame++;
}

void QuakeNode::update_textures(const vk::CommandBuffer& cmd) {
    // Make sure the current_textures is valid for the current frame
    for (auto texnum : pending_uploads) {
        std::shared_ptr<QuakeTexture>& tex = current_textures[texnum];
        assert(!tex->gpu_tex);
        const bool force_linear_filter = tex->flags & TEXPREF_LINEAR;
        tex->gpu_tex = make_rgb8_texture(cmd, allocator, tex->cpu_tex, tex->width, tex->height,
                                         force_linear_filter, !tex->linear);
    }
    // Update descriptors for this frame and copy texture ptr to keep them alive
    pending_uploads.clear();
    FrameData& cur_frame = current_frame();
    merian::DescriptorSetUpdate update(cur_frame.quake_sets);
    for (uint32_t texnum = 0; texnum < current_textures.size(); texnum++) {
        if (current_textures[texnum] != cur_frame.textures[texnum]) {
            cur_frame.textures[texnum] = current_textures[texnum];
            update.write_descriptor_texture(BINDING_IMG_TEX, current_textures[texnum]->gpu_tex,
                                            texnum);
        }
    }
    update.update(context);
}

void QuakeNode::update_static_geo(const vk::CommandBuffer& cmd, const bool refresh_geo) {
    FrameData& cur_frame = current_frame();
    if (refresh_geo) {
        static_vtx.clear();
        static_idx.clear();
        static_ext.clear();

        add_geo(cl_entities, static_vtx, static_idx, static_ext);

        SPDLOG_DEBUG("static geo: vtx size: {} idx size: {} ext size: {}", static_vtx.size(),
                     static_idx.size(), static_ext.size());

        if (!static_idx.empty()) {
            std::tie(current_static_vtx_buffer, current_static_idx_buffer,
                     current_static_ext_buffer) =
                ensure_vertex_index_ext_buffer(allocator, cmd, static_vtx, static_idx, static_ext,
                                               current_static_vtx_buffer, current_static_idx_buffer,
                                               current_static_ext_buffer);
            vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
                vk::Format::eR32G32B32Sfloat,
                current_static_vtx_buffer->get_device_address(),
                3 * sizeof(float),
                static_cast<uint32_t>(static_vtx.size() / 3 - 1),
                vk::IndexType::eUint32,
                {current_static_idx_buffer->get_device_address()},
                {}};

            vk::AccelerationStructureGeometryKHR geometry{vk::GeometryTypeKHR::eTriangles,
                                                          {triangles}};
            // Create offset info that allows us to say how many triangles and vertices to read
            vk::AccelerationStructureBuildRangeInfoKHR range_info{
                static_cast<uint32_t>(static_idx.size() / 3), 0, 0, 0};
            current_static_blas = cur_frame.blas_builder->queue_build(
                {geometry}, {range_info},
                vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                    vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction);
        }
    }
    if (current_static_blas) {
        cur_frame.static_vtx_buffer = current_static_vtx_buffer;
        cur_frame.static_idx_buffer = current_static_idx_buffer;
        cur_frame.static_ext_buffer = current_static_ext_buffer;
        cur_frame.static_blas = current_static_blas;
    } else {
        cur_frame.static_vtx_buffer = binding_dummy_buffer;
        cur_frame.static_idx_buffer = binding_dummy_buffer;
        cur_frame.static_ext_buffer = binding_dummy_buffer;
        cur_frame.static_blas = current_static_blas;
    }

    merian::DescriptorSetUpdate update(cur_frame.quake_sets);
    update.write_descriptor_buffer(BINDING_VTX_BUF, cur_frame.static_vtx_buffer, 0, VK_WHOLE_SIZE,
                                   ARRAY_IDX_STATIC);
    update.write_descriptor_buffer(BINDING_IDX_BUF, cur_frame.static_idx_buffer, 0, VK_WHOLE_SIZE,
                                   ARRAY_IDX_STATIC);
    update.write_descriptor_buffer(BINDING_EXT_BUF, cur_frame.static_ext_buffer, 0, VK_WHOLE_SIZE,
                                   ARRAY_IDX_STATIC);
    update.update(context);
}

void QuakeNode::update_dynamic_geo(const vk::CommandBuffer& cmd) {
    dynamic_vtx.clear();
    dynamic_idx.clear();
    dynamic_ext.clear();

    std::thread particle_viewent([&]() {
        add_geo(&cl.viewent, dynamic_vtx, dynamic_idx, dynamic_ext);
        add_particles(dynamic_vtx, dynamic_idx, dynamic_ext, texnum_blood, texnum_explosion);
    });

    const uint32_t concurrency = std::thread::hardware_concurrency();

    std::vector<std::vector<float>> thread_dynamic_vtx(concurrency);
    std::vector<std::vector<uint32_t>> thread_dynamic_idx(concurrency);
    std::vector<std::vector<VertexExtraData>> thread_dynamic_ext(concurrency);

    merian::parallel_for(
        std::max(cl_numvisedicts, cl.num_statics),
        [&](uint32_t index, uint32_t thread_index) {
            if (index < (uint32_t)cl_numvisedicts)
                add_geo(cl_visedicts[index], thread_dynamic_vtx[thread_index],
                        thread_dynamic_idx[thread_index], thread_dynamic_ext[thread_index]);
            if (index < (uint32_t)cl.num_statics)
                add_geo(cl_static_entities + index, thread_dynamic_vtx[thread_index],
                        thread_dynamic_idx[thread_index], thread_dynamic_ext[thread_index]);
        },
        concurrency);

    particle_viewent.join();

    for (uint32_t i = 0; i < concurrency; i++) {
        uint32_t old_vtx_count = dynamic_vtx.size() / 3;

        merian::raw_copy_back(dynamic_vtx, thread_dynamic_vtx[i]);
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

    FrameData& cur_frame = current_frame();
    if (!dynamic_idx.empty()) {
        std::tie(cur_frame.dynamic_vtx_buffer, cur_frame.dynamic_idx_buffer,
                 cur_frame.dynamic_ext_buffer) =
            ensure_vertex_index_ext_buffer(
                allocator, cmd, dynamic_vtx, dynamic_idx, dynamic_ext, cur_frame.dynamic_vtx_buffer,
                cur_frame.dynamic_idx_buffer, cur_frame.dynamic_ext_buffer);

        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            vk::Format::eR32G32B32Sfloat,
            cur_frame.dynamic_vtx_buffer->get_device_address(),
            3 * sizeof(float),
            static_cast<uint32_t>(dynamic_vtx.size() / 3 - 1),
            vk::IndexType::eUint32,
            {cur_frame.dynamic_idx_buffer->get_device_address()},
            {}};

        vk::AccelerationStructureGeometryKHR geometry{vk::GeometryTypeKHR::eTriangles, {triangles}};
        // Create offset info that allows us to say how many triangles and vertices to read
        vk::AccelerationStructureBuildRangeInfoKHR range_info{
            static_cast<uint32_t>(dynamic_idx.size() / 3), 0, 0, 0};

        vk::BuildAccelerationStructureFlagsKHR flags =
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
        if (cur_frame.last_dynamic_vtx_size == dynamic_vtx.size() &&
            cur_frame.last_dynamic_idx_size == dynamic_idx.size()) {
            if (frame % 100 == 0)
                cur_frame.blas_builder->queue_rebuild({geometry}, {range_info},
                                                      cur_frame.dynamic_blas, flags);
            else
                cur_frame.blas_builder->queue_update({geometry}, {range_info},
                                                     cur_frame.dynamic_blas, flags);
        } else {
            // Build new
            cur_frame.dynamic_blas =
                cur_frame.blas_builder->queue_build({geometry}, {range_info}, flags);
            merian::DescriptorSetUpdate update(cur_frame.quake_sets);
            update.write_descriptor_buffer(BINDING_VTX_BUF, cur_frame.dynamic_vtx_buffer, 0,
                                           VK_WHOLE_SIZE, ARRAY_IDX_DYNAMIC);
            update.write_descriptor_buffer(BINDING_IDX_BUF, cur_frame.dynamic_idx_buffer, 0,
                                           VK_WHOLE_SIZE, ARRAY_IDX_DYNAMIC);
            update.write_descriptor_buffer(BINDING_EXT_BUF, cur_frame.dynamic_ext_buffer, 0,
                                           VK_WHOLE_SIZE, ARRAY_IDX_DYNAMIC);
            update.update(context);
            cur_frame.last_dynamic_vtx_size = dynamic_vtx.size();
            cur_frame.last_dynamic_idx_size = dynamic_idx.size();
        }
    } else {
        merian::DescriptorSetUpdate update(cur_frame.quake_sets);
        update.write_descriptor_buffer(BINDING_VTX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                       ARRAY_IDX_DYNAMIC);
        update.write_descriptor_buffer(BINDING_IDX_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                       ARRAY_IDX_DYNAMIC);
        update.write_descriptor_buffer(BINDING_EXT_BUF, binding_dummy_buffer, 0, VK_WHOLE_SIZE,
                                       ARRAY_IDX_DYNAMIC);
        update.update(context);
    }
}

void QuakeNode::update_as(const vk::CommandBuffer& cmd, const merian::ProfilerHandle profiler) {
    FrameData& cur_frame = current_frame();
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "blas");
        cur_frame.blas_builder->get_cmds(cmd);
    }

    instances.clear();
    // We use the custom index in the shader to differentiate
    if (cur_frame.static_blas) {
        instances.push_back(vk::AccelerationStructureInstanceKHR{
            merian::transform_identity(),
            ARRAY_IDX_STATIC,
            0xFF,
            0,
            {}, // vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            cur_frame.static_blas->get_acceleration_structure_device_address(),
        });
    }
    if (cur_frame.dynamic_blas) {
        instances.push_back(vk::AccelerationStructureInstanceKHR{
            merian::transform_identity(),
            ARRAY_IDX_DYNAMIC,
            0xFF,
            0,
            {}, // vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            cur_frame.dynamic_blas->get_acceleration_structure_device_address(),
        });
    }

    if (!instances.empty()) {
        const vk::BufferUsageFlags instances_buffer_usage =
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
        cur_frame.instances_buffer = ensure_buffer(allocator, instances_buffer_usage, cmd,
                                                   instances, cur_frame.instances_buffer, 16);
        const vk::BufferMemoryBarrier barrier = cur_frame.instances_buffer->buffer_barrier(
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eAccelerationStructureWriteKHR);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, {},
                            barrier, {});

        if (cur_frame.last_instances_size == instances.size()) {
            // rebuild
            cur_frame.tlas_builder->queue_rebuild(instances.size(), cur_frame.instances_buffer,
                                                  cur_frame.tlas);
        } else {
            cur_frame.tlas =
                cur_frame.tlas_builder->queue_build(instances.size(), cur_frame.instances_buffer);

            merian::DescriptorSetUpdate update(cur_frame.quake_sets);
            update.write_descriptor_acceleration_structure(BINDING_TLAS, *cur_frame.tlas);
            update.update(context);
            cur_frame.last_instances_size = instances.size();
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "tlas");
            cur_frame.tlas_builder->get_cmds(cmd);
            cur_frame.tlas_builder->cmd_barrier(cmd, vk::PipelineStageFlagBits::eComputeShader);
        }
    } else {
        cur_frame.tlas = nullptr;
    }
}
