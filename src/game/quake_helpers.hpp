#pragma once

#include <glm/glm.hpp>
#include <vector>

extern "C" {
#include "quakedef.h"
}

struct VertexExtraData {
    // texnum and alpha in upper 4 bits
    // Alpha meaning: 0: use texture, [1,15] map to [0,1] where 15 is fully opaque and 1
    // transparent
    uint16_t texnum_alpha{};
    // 12 bit fullbright_texnum or 0 if not bright, 4 bit flags (most significant)
    // for flags see MAT_FLAGS_* in config.h
    uint16_t texnum_fb_flags{};

    // Normals encoded using encode_normal
    // or glossmap texnum and normalmap texnum
    // if n1_brush = ~0.
    uint32_t n0_gloss_norm{};
    // Marks as brush model if ~0, else second normal
    uint32_t n1_brush{};
    uint32_t n2{};

    // Texture coords, encoded using float_to_half
    uint16_t s_0{};
    uint16_t t_0{};
    uint16_t s_1{};
    uint16_t t_1{};
    uint16_t s_2{};
    uint16_t t_2{};
};

uint16_t
make_texnum_alpha(gltexture_s* tex, entity_t* entity = nullptr, msurface_t* surface = nullptr);

void add_particles(std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<VertexExtraData>& ext,
                   const uint32_t texnum_blood,
                   const uint32_t texnum_explosion,
                   const bool no_random,
                   const double prev_cl_time);

void add_geo_alias(entity_t* ent,
                   [[maybe_unused]] qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<VertexExtraData>& ext);

// geo_selector: 0 -> all, 1 -> opaque, 2 -> transparent
void add_geo_brush(entity_t* ent,
                   qmodel_t* m,
                   std::vector<float>& vtx,
                   std::vector<float>& prev_vtx,
                   std::vector<uint32_t>& idx,
                   std::vector<VertexExtraData>& ext,
                   int geo_selector = 0);

void add_geo_sprite(entity_t* ent,
                    [[maybe_unused]] qmodel_t* m,
                    std::vector<float>& vtx,
                    std::vector<float>& prev_vtx,
                    std::vector<uint32_t>& idx,
                    std::vector<VertexExtraData>& ext);

// Adds the geo from entity into the vectors.
void add_geo(entity_t* ent,
             std::vector<float>& vtx,
             std::vector<float>& prev_vtx,
             std::vector<uint32_t>& idx,
             std::vector<VertexExtraData>& ext);
