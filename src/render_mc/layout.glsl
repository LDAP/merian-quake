#include "../config.h"
#include "merian-nodes/common/gbuffer.glsl.h"

#extension GL_EXT_scalar_block_layout       : require
#extension GL_EXT_shader_16bit_storage      : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout (constant_id = 2) const int MAX_SPP = 4;
layout (constant_id = 3) const int MAX_PATH_LENGTH = 3;
layout (constant_id = 4) const bool USE_LIGHT_CACHE_TAIL = false;
layout (constant_id = 5) const float FOV_TAN_ALPHA_HALF = 0;
layout (constant_id = 6) const float SUN_W_X = 0;
layout (constant_id = 7) const float SUN_W_Y = 0;
layout (constant_id = 8) const float SUN_W_Z = 0;
layout (constant_id = 9) const float SUN_COLOR_R = 0;
layout (constant_id = 10) const float SUN_COLOR_G = 0;
layout (constant_id = 11) const float SUN_COLOR_B = 0;
layout (constant_id = 12) const int ADAPTIVE_SAMPLING = 0;
layout (constant_id = 13) const int VOLUME_SPP = 0;
layout (constant_id = 14) const bool VOLUME_USE_LIGHT_CACHE = false;
layout (constant_id = 15) const float DRAINE_G = 0.65;
layout (constant_id = 16) const float DRAINE_A = 32.0;
layout (constant_id = 17) const int MC_SAMPLES = 5;
layout (constant_id = 18) const float MC_SAMPLES_ADAPTIVE_PROB = 0.7;
layout (constant_id = 19) const int DISTANCE_MC_SAMPLES = 3;
layout (constant_id = 20) const bool MC_FAST_RECOVERY = true;
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
#include "../scene_info.glsl.h"

layout(push_constant) uniform PushConstant { 
    UniformData params;
};

// GRAPH in
layout(set = 0, binding = 0) uniform sampler2D img_blue;
layout(set = 0, binding = 1) uniform sampler2D img_prev_filtered;
layout(set = 0, binding = 2) uniform sampler2D img_prev_volume_depth;

layout(set = 0, binding = 3, scalar) buffer buf_prev_gbuf {
    GBuffer prev_gbuffer[];
};

layout(set = 0, binding = 4) uniform sampler2D img_tex[MAX_GLTEXTURES];

layout(set = 0, binding = 5, scalar) buffer readonly restrict buf_vtx_t {
    // vertex positons
    vec3 v[];
} buf_vtx[MAX_GEOMETRIES];

layout(set = 0, binding = 6, scalar) buffer readonly restrict buf_prev_vtx_t {
    // vertex positons
    vec3 v[];
} buf_prev_vtx[MAX_GEOMETRIES];

layout(set = 0, binding = 7, scalar) buffer readonly restrict buf_idx_t {
    // index data for every instance
    uvec3 i[];
} buf_idx[MAX_GEOMETRIES];

layout(set = 0, binding = 8, scalar) buffer readonly restrict buf_ext_t {
    // extra geo info
    VertexExtraData v[];
} buf_ext[MAX_GEOMETRIES];

layout(set = 0, binding = 9) uniform accelerationStructureEXT tlas;


// GRAPH out
layout(set = 0, binding = 10) uniform writeonly restrict image2D img_irradiance;
layout(set = 0, binding = 11) uniform writeonly restrict image2D img_albedo;
layout(set = 0, binding = 12) uniform writeonly restrict image2D img_mv;
layout(set = 0, binding = 13) uniform writeonly restrict image2D img_debug;
layout(set = 0, binding = 14) uniform writeonly restrict image2D img_moments;
layout(set = 0, binding = 15) uniform writeonly restrict image2D img_volume;
layout(set = 0, binding = 16) uniform writeonly restrict image2D img_volume_moments;
layout(set = 0, binding = 17) uniform writeonly restrict image2D img_volume_depth;
layout(set = 0, binding = 18, rg16f) uniform restrict image2D img_volume_mv;

// GRAPH buffer out
layout(set = 0, binding = 19, scalar) buffer restrict buf_mc_states {
    MCState mc_states[];
};
layout(set = 0, binding = 20, scalar) buffer restrict buf_light_cache {
    LightCacheVertex light_cache[];
};
layout(set = 0, binding = 21, scalar) buffer restrict buf_gbuf {
    GBuffer gbuffer[];
};
layout(set = 0, binding = 22, scalar) buffer restrict buf_dist_mc_states {
    DistanceMCVertex distance_mc_states[];
};
