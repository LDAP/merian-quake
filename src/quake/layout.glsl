#include "config.h"
#include "grid.h"

#extension GL_EXT_scalar_block_layout       : require
#extension GL_EXT_shader_16bit_storage      : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout (constant_id = 2) const int SPP = 1;
layout (constant_id = 3) const int MAX_PATH_LENGTH = 3;
layout (constant_id = 4) const int USE_LIGHT_CACHE_TAIL = 0;

layout(push_constant) uniform PushConstant { 
    vec4 cam_x;
    vec4 cam_w;
    vec4 cam_u;
    vec4 prev_cam_x;
    vec4 prev_cam_w;
    vec4 prev_cam_u;

    uint sky_rt_bk, sky_lf_ft, sky_up_dn;

    float cl_time; // quake time
    uint frame;
    uint player; // see `PlayerData` in quake_node.hpp
    uint rt_config; // see `RTConfig` in in quake_node.hpp
} params;

uint rt_flags()        { return params.rt_config & 0xff; }
float bsdp_p()         { return ((params.rt_config >> 16) & 0xff) / 255.; }
float ml_prior()       { return ((params.rt_config >> 24) & 0xff) / 255.; }


// GRAPH IN/OUTs

layout(set = 0, binding = 0) uniform sampler2D img_blue;

layout(set = 0, binding = 1) uniform writeonly image2D img_irradiance;
layout(set = 0, binding = 2) uniform writeonly image2D img_albedo;
layout(set = 0, binding = 3) uniform writeonly image2D img_gbuf;
layout(set = 0, binding = 4) uniform writeonly image2D img_mv;
layout(set = 0, binding = 5, scalar) buffer buf_cells {
    MCVertex mc_states[];
};
layout(set = 0, binding = 6, scalar) buffer buf_lightcache {
    LightCacheVertex light_cache[];
};

// QUAKE 

// See quake_node.hpp
struct VertexExtraData {
    uint16_t texnum_alpha;
    uint16_t texnum_fb_flags;
    
    uint n0_gloss_norm;
    uint n1_brush;
    uint n2;

    uint st_0;
    uint st_1;
    uint st_2;
};

layout(set = 1, binding = BINDING_VTX_BUF, scalar) buffer buf_vtx_t {
    // vertex positons
    vec3 v[];
} buf_vtx[MAX_GEOMETRIES];

layout(set = 1, binding = BINDING_IDX_BUF, scalar) buffer buf_idx_t {
    // index data for every instance
    uvec3 i[];
} buf_idx[MAX_GEOMETRIES];

layout(set = 1, binding = BINDING_EXT_BUF, scalar) buffer buf_ext_t {
    // extra geo info
    VertexExtraData v[];
} buf_ext[MAX_GEOMETRIES];

layout(set = 1, binding = BINDING_IMG_TEX) uniform sampler2D img_tex[MAX_GLTEXTURES];

layout(set = 1, binding = BINDING_TLAS) uniform accelerationStructureEXT tlas;
