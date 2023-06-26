#include "quake/quake_node.hpp"
#include "config.h"
#include "merian/utils/bitpacking.hpp"
#include "merian/utils/glm.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/shader/shader_module.hpp"

static const uint32_t spv[] = {
#include "quake.comp.spv.h"
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

void init_quake(const char* base_dir) {
    const char* argv[] = {
        "quakespasm",
        "-basedir",
        base_dir,
        "+skill",
        "2",
        "-game",
        "ad",
        "+map",
        "e1m1",
        "-game",
        "SlayerTest", // needs different particle rules! also overbright
                      // and lack of alpha?
        "+map",
        "e1m2b",
        "+map",
        "ep1m1",
        "+map",
        "e1m1b",
        "+map",
        "st1m1",
    };
    const int argc = 9;

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
    aliashdr_t* hdr = (aliashdr_t*)Mod_Extradata(ent->model);
    aliasmesh_t* desc = (aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
    // the plural here really hurts but it's from quakespasm code:
    int16_t* indexes = (int16_t*)((uint8_t*)hdr + hdr->indexes);
    trivertx_t* trivertexes = (trivertx_t*)((uint8_t*)hdr + hdr->vertexes);

    // lerpdata_t  lerpdata;
    // R_SetupAliasFrame(hdr, ent->frame, &lerpdata);
    // R_SetupEntityTransform(ent, &lerpdata);
    // angles: pitch yaw roll. axes: right fwd up
    float angles[3] = {-ent->angles[0], ent->angles[1], ent->angles[2]};
    float fwd[3], rgt[3], top[3], pos[3];
    AngleVectors(angles, fwd, rgt, top);
    for (int k = 0; k < 3; k++)
        rgt[k] = -rgt[k]; // seems these come in flipped

    // for(int f = 0; f < hdr->numposes; f++)
    // TODO: upload all vertices so we can just alter the indices on gpu
    int f = ent->frame;
    if (f < 0 || f >= hdr->numposes)
        return;

    uint32_t vtx_cnt = vtx.size();
    for (int v = 0; v < hdr->numverts_vbo; v++) {
        int i = hdr->numverts * f + desc[v].vertindex;
        // get model pos
        for (int k = 0; k < 3; k++)
            pos[k] = trivertexes[i].v[k] * hdr->scale[k] + hdr->scale_origin[k];
        // convert to world space
        for (int k = 0; k < 3; k++)
            vtx.emplace_back(ent->origin[k] + rgt[k] * pos[1] + top[k] * pos[2] + fwd[k] * pos[0]);
    }

    for (int i = 0; i < hdr->numindexes; i++)
        idx[i] = vtx_cnt + indexes[i];

    // both options fail to extract correct creases/vertex normals for health/shells
    // in fact, the shambler has crazy artifacts all over. maybe this is all wrong and
    // just by chance happened to produce something similar enough sometimes?
    // TODO: fuck this vbo bs and get the mdl itself

    // normals for each vertex from above
    uint32_t* tmpn = (uint32_t*)alloca(sizeof(uint32_t) * hdr->numverts_vbo);
    for (int v = 0; v < hdr->numverts_vbo; v++) {
        int i = hdr->numverts * f + desc[v].vertindex;
        float nm[3], nw[3];
        memcpy(nm, r_avertexnormals[trivertexes[i].lightnormalindex], sizeof(float) * 3);
        for (int k = 0; k < 3; k++)
            nw[k] = nm[0] * fwd[k] + nm[1] * rgt[k] + nm[2] * top[k];
        *(tmpn + v) = merian::encode_normal(nw);
    }

    // add extra data for each primitive
    for (int i = 0; i < hdr->numindexes / 3; i++) {
        const int sk = CLAMP(0, ent->skinnum, hdr->numskins - 1), fm = ((int)(cl.time * 10)) & 3;
        const uint16_t texnum = hdr->gltextures[sk][fm]->texnum;
        const uint16_t fb_texnum = hdr->fbtextures[sk][fm] ? hdr->fbtextures[sk][fm]->texnum : 0;

        uint32_t n0, n1, n2;
        if (hdr->nmtextures[sk][fm]) {
            // this discards the vertex normals
            n0 = merian::pack_uint32(0, hdr->nmtextures[sk][fm]->texnum);
            n1 = 0xffffffff; // mark as brush model -> to use normal map
        } else {
            // the vertex normals -> currently not used in shader
            n0 = *(tmpn + indexes[3 * i + 0]);
            n1 = *(tmpn + indexes[3 * i + 1]);
            n2 = *(tmpn + indexes[3 * i + 2]);
        }

        ext.emplace_back(
            n0, n1, n2,
            merian::float_to_half((desc[indexes[3 * i + 0]].st[0] + 0.5) / (float)hdr->skinwidth),
            merian::float_to_half((desc[indexes[3 * i + 0]].st[1] + 0.5) / (float)hdr->skinheight),
            merian::float_to_half((desc[indexes[3 * i + 1]].st[0] + 0.5) / (float)hdr->skinwidth),
            merian::float_to_half((desc[indexes[3 * i + 1]].st[1] + 0.5) / (float)hdr->skinheight),
            merian::float_to_half((desc[indexes[3 * i + 2]].st[0] + 0.5) / (float)hdr->skinwidth),
            merian::float_to_half((desc[indexes[3 * i + 2]].st[1] + 0.5) / (float)hdr->skinheight),
            texnum, fb_texnum);
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
    float fwd[3], rgt[3], top[3];
    AngleVectors(angles, fwd, rgt, top);

    for (int i = 0; i < m->nummodelsurfaces; i++) {
#if WATER_MODE == WATER_MODE_FULL
        int wateroffset = 0;
    again:;
#endif

        msurface_t* surf = &m->surfaces[m->firstmodelsurface + i];
        if (!strcmp(surf->texinfo->texture->name, "skip"))
            continue;

        glpoly_t* p = surf->polys;
        while (p) {
            uint32_t vtx_cnt = vtx.size();
            for (int k = 0; k < p->numverts; k++) {
                for (int l = 0; l < 3; l++) {
                    float coord = p->verts[k][0] * fwd[l] - p->verts[k][1] * rgt[l] +
                                  p->verts[k][2] * top[l] + ent->origin[l];
#if WATER_MODE == WATER_MODE_FULL
                    if (wateroffset) {
                        // clang-format off
                        coord += fwd[l] * (p->verts[k][0] > (surf->mins[0] + surf->maxs[0]) / 2.0 ? WATER_DEPTH : -WATER_DEPTH);
                        coord -= rgt[l] * (p->verts[k][1] > (surf->mins[1] + surf->maxs[1]) / 2.0 ? WATER_DEPTH : -WATER_DEPTH);
                        if (l == 2)
                            coord -= WATER_DEPTH;
                        // clang-format on
                    }
#endif
                    vtx.emplace_back(coord);
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
                    merian::pack_uint32(t->gloss ? t->gloss->texnum : 0,
                                        t->norm ? t->norm->texnum : 0),
                    0xffffffff,
                    0,
                };
                if (surf->texinfo->texture->gltexture) {
                    extra.s_0 = merian::float_to_half(p->verts[0][3]);
                    extra.t_0 = merian::float_to_half(p->verts[0][4]);
                    extra.s_1 = merian::float_to_half(p->verts[k - 1][3]);
                    extra.t_1 = merian::float_to_half(p->verts[k - 1][4]);
                    extra.s_2 = merian::float_to_half(p->verts[k - 0][3]);
                    extra.t_2 = merian::float_to_half(p->verts[k - 0][4]);
                    extra.texnum_alpha = t->gltexture->texnum;
                    extra.texnum_fb_flags = t->fullbright ? t->fullbright->texnum : 0;

                    // alpha
                    uint32_t ai = CLAMP(0, (ent->alpha - 1.0) / 254.0 * 15, 15); // alpha in 4 bits
                    if (!ent->alpha)
                        ai = 15;
                    // TODO: 0 means default, 1 means invisible, 255 is opaque, 2--254 is
                    // really applicable
                    // TODO: default means  map_lavaalpha > 0 ? map_lavaalpha :
                    // map_wateralpha
                    // TODO: or "slime" or "tele" instead of "lava"
                    extra.texnum_alpha |= ai << 12;

                    // max textures is 4096 (12 bit) and we have 16. so we can put 4 bits
                    // worth of flags here:
                    uint32_t flags = 0;
                    if (surf->flags & SURF_DRAWLAVA)
                        flags = 1;
                    if (surf->flags & SURF_DRAWSLIME)
                        flags = 2;
                    if (surf->flags & SURF_DRAWTELE)
                        flags = 3;
                    if (surf->flags & SURF_DRAWWATER)
                        flags = 4;
                    if (surf->flags & SURF_DRAWSKY)
                        flags = 6;
#if WATER_MODE == WATER_MODE_FULL
                    if (wateroffset)
                        flags = 5; // this is our procedural water lower mark
#endif
                    if (strstr(t->gltexture->name, "wfall"))
                        flags = 7; // hack for ad_tears and emissive waterfalls

                    extra.texnum_fb_flags |= flags << 12;
                }
                // ?! What does that?
                if (surf->flags & SURF_DRAWSKY)
                    extra.texnum_alpha = 0xfff;

                ext.push_back(extra);
            }

#if WATER_MODE == WATER_MODE_FULL
            if (!wateroffset && (surf->flags & SURF_DRAWWATER)) {
                // TODO: and normal points the right way?
                wateroffset = 1;
                goto again;
            }
#endif
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
        uint32_t vtx_cnt = vtx.size();
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
            texnum = frame->gltexture->texnum;
        }

        ext.emplace_back(n_enc, n_enc, n_enc, merian::float_to_half(0), merian::float_to_half(1),
                         merian::float_to_half(0), merian::float_to_half(0),
                         merian::float_to_half(1), merian::float_to_half(0), texnum,
                         texnum // sprite allways emits
        );
        ext.emplace_back(n_enc, n_enc, n_enc, merian::float_to_half(0), merian::float_to_half(1),
                         merian::float_to_half(1), merian::float_to_half(0),
                         merian::float_to_half(1), merian::float_to_half(1), texnum,
                         texnum // sprite allways emits
        );

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
    // if (qs_data.worldspawn)
    //     return;

    // TODO: lerp between lerpdata.pose1 and pose2 using blend
    // TODO: apply transformation matrix cpu side, whatever.
    // TODO: later: put into rt animation kernel
    if (m->type == mod_alias) { // alias model:
        add_geo_alias(ent, m, vtx, idx, ext);
        assert(ext.size() == idx.size() / 3);
    } else if (m->type == mod_brush) { // brush model:
        add_geo_brush(ent, m, vtx, idx, ext);
        assert(ext.size() == idx.size() / 3);
    } else if (m->type == mod_sprite) {
        // explosions, decals, etc, this is R_DrawSpriteModel
        add_geo_sprite(ent, m, vtx, idx, ext);
        assert(ext.size() == idx.size() / 3);
    }
}

// QuakeNode
// --------------------------------------------------------------------------------------

QuakeNode::QuakeNode(const merian::SharedContext& context,
                     const merian::ResourceAllocatorHandle& allocator,
                     const char* base_dir)
    : context(context), allocator(allocator) {

    // QUAKE INIT
    if (quake_data.node) {
        throw std::runtime_error{"Only one quake node can be created."};
    }
    quake_data.node = this;
    host_parms = &quake_data.params;
    init_quake(base_dir);

    // PIPELINE CREATION
    shader = std::make_shared<merian::ShaderModule>(context, sizeof(spv), spv);
}

QuakeNode::~QuakeNode() {
    deinit_quake();
}

// -------------------------------------------------------------------------------------------

void QuakeNode::QS_worldspawn() {
    // Called after ~5 frames when textures are loaded.
    SPDLOG_DEBUG("worldspawn");
    worldspawn = true;
}

void QuakeNode::QS_texture_load(gltexture_t* glt, uint32_t* data) {
    // LOG -----------------------------------------------

    std::string source = strcmp(glt->source_file, "") == 0 ? "memory" : glt->source_file;
    SPDLOG_DEBUG("texture_load {} {} {}x{} from {}, frame: {}", glt->texnum, glt->name, glt->width,
                 glt->height, source, glt->visframe);

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
    textures.insert(std::make_pair(glt->texnum, texture));
    pending_uploads.push(glt->texnum);
}

void QuakeNode::IN_Move(usercmd_t* cmd) {
    SPDLOG_DEBUG("move");

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
    // SELECT MAP

    // if (frame == 0) {
    //     // careful to only do this at == 0 so sv_player (among others) will not crash
    //     const char* p_exec = nullptr; // map exec string from examples/ad.cfg
    //     if (p_exec) {
    //         Cmd_ExecuteString(p_exec, src_command);
    //         sv_player_set = 0; // just in case we loaded a map (demo, savegame)
    //     }
    // }

    // GRAPH DESC SETS
    std::tie(graph_textures, graph_sets, graph_pool, graph_desc_set_layout) =
        merian::make_graph_descriptor_sets(context, allocator, image_inputs, buffer_inputs,
                                           image_outputs, buffer_outputs, graph_desc_set_layout);
}

void QuakeNode::cmd_process(const vk::CommandBuffer& cmd,
                            merian::GraphRun& run,
                            const uint32_t graph_set_index,
                            const std::vector<merian::ImageHandle>& image_inputs,
                            const std::vector<merian::BufferHandle>& buffer_inputs,
                            const std::vector<merian::ImageHandle>& image_outputs,
                            const std::vector<merian::BufferHandle>& buffer_outputs) {

    // UPDATE GAMESTATE (if not paused)
    if (!pause) {
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

    // UPDATE PUSH CONSTANT (with player data)

    // TODO: maybe check if sv_player is valid
    // for (int i=0;i<svs.maxclients && !sv_player_set; i++, host_client++)
    // {
    //   if (!host_client->active) continue;
    //   sv_player = host_client->edict;
    //   sv_player_set = 1;
    // }

    pc.frame = frame;
    pc.torch = sv_player->v.weapon == 1; // shotgun has torch
    pc.water = sv_player->v.waterlevel >= 3;
    pc.health = sv_player->v.health;
    pc.armor = sv_player->v.armorvalue;
    pc.sky = texnum_skybox;
    pc.cl_time = cl.time;
    float rgt[3];
    AngleVectors(r_refdef.viewangles, &pc.cam_w.x, rgt, &pc.cam_u.x);
    copy_to_vec4(r_refdef.vieworg, pc.cam_x);
    glm::vec3 fog_color = vec3_from_float(Fog_GetColor());
    float fog_density = Fog_GetDensity();
    fog_density *= fog_density;
    pc.fog = glm::vec4(fog_color, fog_density);

    // Quake loads static geo only after a few frames...
    if (frame == 10)
        prepare_static_geo(cmd);
    prepare_dynamic_geo(cmd);
    prepare_tlas(cmd);

    frame++;
}

void QuakeNode::prepare_static_geo(const vk::CommandBuffer& cmd) {}
void QuakeNode::prepare_dynamic_geo(const vk::CommandBuffer& cmd) {}
void QuakeNode::prepare_tlas(const vk::CommandBuffer& cmd) {}
