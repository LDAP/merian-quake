#include "config.h"
#include "grid.h"
#include "merian-nodes/common/gbuffer.glsl.h"

#extension GL_EXT_scalar_block_layout       : require
#extension GL_EXT_shader_16bit_storage      : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout (constant_id = 2) const int MAX_SPP = 4;
layout (constant_id = 3) const int MAX_PATH_LENGTH = 3;
layout (constant_id = 4) const int USE_LIGHT_CACHE_TAIL = 0;
layout (constant_id = 5) const float FOV_TAN_ALPHA_HALF = 0;
layout (constant_id = 6) const float SUN_W_X = 0;
layout (constant_id = 7) const float SUN_W_Y = 0;
layout (constant_id = 8) const float SUN_W_Z = 0;
layout (constant_id = 9) const float SUN_COLOR_R = 0;
layout (constant_id = 10) const float SUN_COLOR_G = 0;
layout (constant_id = 11) const float SUN_COLOR_B = 0;
layout (constant_id = 12) const int ADAPTIVE_SAMPLING = 0;
layout (constant_id = 13) const int VOLUME_SPP = 0;
layout (constant_id = 14) const float MU_T = 0.;
layout (constant_id = 15) const float MU_S = 0.;
layout (constant_id = 16) const int VOLUME_USE_LIGHT_CACHE = 0;

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

#define rt_flags() (params.rt_config & 0xff)
#define bsdf_p() (((params.rt_config >> 16) & 0xff) / 255.)
#define ml_prior() (((params.rt_config >> 24) & 0xff) / 255.)

// GRAPH image in
layout(set = 0, binding = 0) uniform sampler2D img_blue;
layout(set = 0, binding = 1) uniform sampler2D img_prev_filtered;

// GRAPH buffer in
layout(set = 0, binding = 2, scalar) buffer buf_filtered_image {
    float mean_variance[];
};
layout(set = 0, binding = 3, scalar) buffer buf_prev_gbuf {
    GBuffer prev_gbuffer[];
};

// GRAPH image out
layout(set = 0, binding = 4) uniform writeonly image2D img_irradiance;
layout(set = 0, binding = 5) uniform writeonly image2D img_albedo;
layout(set = 0, binding = 6) uniform writeonly image2D img_mv;
layout(set = 0, binding = 7) uniform writeonly image2D img_debug;
layout(set = 0, binding = 8) uniform writeonly image2D img_moments;
layout(set = 0, binding = 9) uniform writeonly image2D img_volume;
layout(set = 0, binding = 10) uniform writeonly image2D img_volume_moments;

// GRAPH buffer out
layout(set = 0, binding = 11, scalar) buffer buf_mc_states {
    MCState mc_states[];
};
layout(set = 0, binding = 12, scalar) buffer buf_light_cache {
    LightCacheVertex light_cache[];
};
layout(set = 0, binding = 13, scalar) buffer buf_gbuf {
    GBuffer gbuffer[];
};

// QUAKE 

// See quake_node.hpp
struct VertexExtraData {
    uint16_t texnum_alpha;
    uint16_t texnum_fb_flags;
    
    uint n0_gloss_norm;
    uint n1_brush;
    uint n2;

    f16mat3x2 st;
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
