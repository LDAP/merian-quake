#include "config.h"
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
layout (constant_id = 14) const int VOLUME_USE_LIGHT_CACHE = 0;
layout (constant_id = 15) const float DRAINE_G = 0.65;
layout (constant_id = 16) const float DRAINE_A = 32.0;
layout (constant_id = 17) const int MC_SAMPLES = 5;
layout (constant_id = 18) const float MC_SAMPLES_ADAPTIVE_PROB = 0.7;
layout (constant_id = 19) const int DISTANCE_MC_SAMPLES = 3;
layout (constant_id = 20) const int MC_FAST_RECOVERY = 1;
layout (constant_id = 21) const float LIGHT_CACHE_LEVELS = 32.;
layout (constant_id = 22) const float LIGHT_CACHE_TAN_ALPHA_HALF = 0.002;
layout (constant_id = 23) const uint LIGHT_CACHE_BUFFER_SIZE = 4000000;
layout (constant_id = 24) const uint MC_ADAPTIVE_BUFFER_SIZE = 32777259;
layout (constant_id = 25) const uint MC_STATIC_BUFFER_SIZE = 800009;
layout (constant_id = 26) const float MC_ADAPTIVE_GRID_TAN_ALPHA_HALF = 0.003;
layout (constant_id = 27) const float MC_STATIC_GRID_WIDTH = 25.3;
layout (constant_id = 28) const int MC_ADAPTIVE_GRID_LEVELS = 10;
layout (constant_id = 29) const int DISTANCE_MC_GRID_WIDTH = 25;
layout (constant_id = 30) const float VOLUME_MAX_T = 1000.;
layout (constant_id = 31) const float SURF_BSDF_P = 0.15;
layout (constant_id = 32) const float VOLUME_PHASE_P = 0.3;
layout (constant_id = 33) const float DIR_GUIDE_PRIOR = 0.2;
layout (constant_id = 34) const float DIST_GUIDE_P = 0.0;
layout (constant_id = 35) const uint DISTANCE_MC_VERTEX_STATE_COUNT = MAX_DISTANCE_MC_VERTEX_STATE_COUNT;
layout (constant_id = 36) const uint SEED = 0;

#include "grid.h"

layout(push_constant) uniform PushConstant { 
    vec4 cam_x; // contains mu_t in alpha
    vec4 cam_w; // contains last cl_time - cl_time (or 1 if paused) in alpha
    vec4 cam_u;
    vec4 prev_cam_x; // contains mu_s.r in alpha
    vec4 prev_cam_w; // contains mu_s.g in alpha
    vec4 prev_cam_u; // contains mu_s.b in alpha

    uint sky_rt_bk, sky_lf_ft, sky_up_dn;

    float cl_time; // quake time
    uint frame;
    uint player; // see `PlayerData` in quake_node.hpp
    uint rt_config; // see `RTConfig` in in quake_node.hpp
} params;

#define rt_flags() (params.rt_config & 0xff)
#define MU_T params.cam_x.a
#define MU_S vec3(params.prev_cam_x.a, params.prev_cam_w.a, params.prev_cam_u.a)
#define TIME_DIFF params.cam_w.a

// GRAPH image in
layout(set = 0, binding = 0) uniform sampler2D img_blue;
layout(set = 0, binding = 1) uniform sampler2D img_prev_filtered;
layout(set = 0, binding = 2) uniform sampler2D img_prev_volume_depth;

// GRAPH buffer in
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
layout(set = 0, binding = 11) uniform writeonly image2D img_volume_depth;
layout(set = 0, binding = 12, rg16f) uniform image2D img_volume_mv;

// GRAPH buffer out
layout(set = 0, binding = 13, scalar) buffer buf_mc_states {
    MCState mc_states[];
};
layout(set = 0, binding = 14, scalar) buffer buf_light_cache {
    LightCacheVertex light_cache[];
};
layout(set = 0, binding = 15, scalar) buffer buf_gbuf {
    GBuffer gbuffer[];
};
layout(set = 0, binding = 16, scalar) buffer buf_dist_mc_states {
    DistanceMCVertex distance_mc_states[];
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

layout(set = 1, binding = BINDING_PREV_VTX_BUF, scalar) buffer buf_prev_vtx_t {
    // vertex positons
    vec3 v[];
} buf_prev_vtx[MAX_GEOMETRIES];

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
