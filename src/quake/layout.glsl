#include "config.h"

#extension GL_EXT_scalar_block_layout       : require
#extension GL_EXT_shader_16bit_storage      : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstant { 
    vec4 cam_x;
    vec4 cam_w;
    vec4 cam_u;
    vec4 fog;
    int torch;
    int water;
    uint sky_rt, sky_bk, sky_lf, sky_ft, sky_up, sky_dn;
    float cl_time; // quake time
    int   ref;     // use reference sampling
    int   health;
    int   armor;
    int frame;
} params;

// GRAPH IN/OUTs

layout(set = 0, binding = 0) uniform sampler2D img_gbuf_in;
layout(set = 0, binding = 1) uniform sampler2D img_mv;
layout(set = 0, binding = 2) uniform sampler2D img_blue;
layout(set = 0, binding = 3) uniform usampler2D img_nee_in; // mc states

layout(set = 0, binding = 4) uniform writeonly image2D img_irradiance;
layout(set = 0, binding = 5) uniform writeonly image2D img_albedo;
layout(set = 0, binding = 6) uniform writeonly image2D img_gbuf_out;
layout(set = 0, binding = 7) uniform writeonly uimage2D img_nee_out; // mc states

// QUAKE 

// See quake_node.hpp
struct VertexExtraData {

    uint n0_gloss_norm;

    uint n1_brush;
    uint n2;

    uint st_0;
    uint st_1;
    uint st_2;

    uint16_t texnum_alpha;
    uint16_t texnum_fb_flags;
};

layout(set = 1, binding = BINDING_VTX_BUF, scalar) buffer buf_vtx_t {
    // vertex positons
    vec3 v[];
} buf_vtx[GEO_DESC_ARRAY_SIZE];

layout(set = 1, binding = BINDING_IDX_BUF, scalar) buffer buf_idx_t {
    // index data for every instance
    uvec3 i[];
} buf_idx[GEO_DESC_ARRAY_SIZE];

layout(set = 1, binding = BINDING_EXT_BUF, scalar) buffer buf_ext_t {
    // extra geo info
    VertexExtraData v[];
} buf_ext[GEO_DESC_ARRAY_SIZE];

layout(set = 1, binding = BINDING_IMG_TEX) uniform sampler2D img_tex[MAX_GLTEXTURES];

layout(set = 1, binding = BINDING_TLAS) uniform accelerationStructureEXT tlas;
