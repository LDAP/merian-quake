#include "../config.h"
#include "merian-shaders/gbuffer.glsl.h"

#extension GL_EXT_scalar_block_layout       : require
#extension GL_EXT_shader_16bit_storage      : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

// #ifndef SURFACE_SPP
// #define SURFACE_SPP 4
// #endif
// #ifndef MAX_PATH_LENGTH
// #define MAX_PATH_LENGTH 3
// #endif
// #ifndef USE_LIGHT_CACHE_TAIL
// #define USE_LIGHT_CACHE_TAIL false
// #endif
// #ifndef FOV_TAN_ALPHA_HALF
// #define FOV_TAN_ALPHA_HALF 0
// #endif
// #ifndef SUN_W_X
// #define SUN_W_X 0
// #endif
// #ifndef SUN_W_Y
// #define SUN_W_Y 0
// #endif
// #ifndef SUN_W_Z
// #define SUN_W_Z 0
// #endif
// #ifndef SUN_COLOR_R
// #define SUN_COLOR_R 0
// #endif
// #ifndef SUN_COLOR_G
// #define SUN_COLOR_G 0
// #endif
// #ifndef SUN_COLOR_B
// #define SUN_COLOR_B 0
// #endif
// #ifndef ADAPTIVE_SAMPLING
// #define ADAPTIVE_SAMPLING 0
// #endif
// #ifndef VOLUME_SPP
// #define VOLUME_SPP 0
// #endif
// #ifndef VOLUME_USE_LIGHT_CACHE
// #define VOLUME_USE_LIGHT_CACHE false
// #endif
// #ifndef DRAINE_G
// #define DRAINE_G 0.65
// #endif
// #ifndef DRAINE_A
// #define DRAINE_A 32.0
// #endif
// #ifndef MC_SAMPLES
// #define MC_SAMPLES 5
// #endif
// #ifndef MC_SAMPLES_ADAPTIVE_PROB
// #define MC_SAMPLES_ADAPTIVE_PROB 0.7
// #endif
// #ifndef DISTANCE_MC_SAMPLES
// #define DISTANCE_MC_SAMPLES 3
// #endif
// #ifndef MC_FAST_RECOVERY
// #define MC_FAST_RECOVERY true
// #endif
// #ifndef LIGHT_CACHE_BUFFER_SIZE
// #define LIGHT_CACHE_BUFFER_SIZE 4000000
// #endif
// #ifndef LC_GRID_STEPS_PER_UNIT_SIZE
// #define LC_GRID_STEPS_PER_UNIT_SIZE 6.0
// #endif
// #ifndef LC_GRID_TAN_ALPHA_HALF
// #define LC_GRID_TAN_ALPHA_HALF 0.002
// #endif
// #ifndef LC_GRID_MIN_WIDTH
// #define LC_GRID_MIN_WIDTH 0.01
// #endif
// #ifndef LC_GRID_POWER
// #define LC_GRID_POWER 2
// #endif
// #ifndef MC_ADAPTIVE_BUFFER_SIZE
// #define MC_ADAPTIVE_BUFFER_SIZE 32777259
// #endif
// #ifndef MC_ADAPTIVE_GRID_TAN_ALPHA_HALF
// #define MC_ADAPTIVE_GRID_TAN_ALPHA_HALF 0.002
// #endif
// #ifndef MC_ADAPTIVE_GRID_MIN_WIDTH
// #define MC_ADAPTIVE_GRID_MIN_WIDTH 0.01
// #endif
// #ifndef MC_ADAPTIVE_GRID_POWER
// #define MC_ADAPTIVE_GRID_POWER 4.0
// #endif
// #ifndef MC_ADAPTIVE_GRID_STEPS_PER_UNIT_SIZE
// #define MC_ADAPTIVE_GRID_STEPS_PER_UNIT_SIZE 4.743416490252569
// #endif
// #ifndef MC_STATIC_BUFFER_SIZE
// #define MC_STATIC_BUFFER_SIZE 800009
// #endif
// #ifndef MC_STATIC_GRID_WIDTH
// #define MC_STATIC_GRID_WIDTH 25.3
// #endif
// #ifndef DISTANCE_MC_GRID_WIDTH
// #define DISTANCE_MC_GRID_WIDTH 25
// #endif
// #ifndef VOLUME_MAX_T
// #define VOLUME_MAX_T 1000.
// #endif
// #ifndef SURF_BSDF_P
// #define SURF_BSDF_P 0.15
// #endif
// #ifndef VOLUME_PHASE_P
// #define VOLUME_PHASE_P 0.3
// #endif
// #ifndef DIR_GUIDE_PRIOR
// #define DIR_GUIDE_PRIOR 0.2
// #endif
// #ifndef DIST_GUIDE_P
// #define DIST_GUIDE_P 0.0
// #endif
// #ifndef DISTANCE_MC_VERTEX_STATE_COUNT
// #define DISTANCE_MC_VERTEX_STATE_COUNT MAX_DISTANCE_MC_VERTEX_STATE_COUNT
// #endif
// #ifndef SEED
// #define SEED 0
// #endif
// #ifndef DEBUG_OUTPUT_CONNECTED
// #define DEBUG_OUTPUT_CONNECTED false
// #endif
// #ifndef DEBUG_OUTPUT_SELECTOR
// #define DEBUG_OUTPUT_SELECTOR 0
// #endif

#include "grid.h"
#include "../scene_info.glsl.h"
#include "../hit.glsl.h"

layout(push_constant) uniform PushConstant { 
    UniformData params;
};

// --- GRAPH in ---
layout(set = 0, binding = 0, scalar) buffer readonly restrict buf_vtx_t {
    // vertex positons
    vec3 v[];
} buf_vtx[MAX_GEOMETRIES];

layout(set = 0, binding = 1, scalar) buffer readonly restrict buf_prev_vtx_t {
    // vertex positons
    vec3 v[];
} buf_prev_vtx[MAX_GEOMETRIES];

layout(set = 0, binding = 2, scalar) buffer readonly restrict buf_idx_t {
    // index data for every instance
    uvec3 i[];
} buf_idx[MAX_GEOMETRIES];

layout(set = 0, binding = 3, scalar) buffer readonly restrict buf_ext_t {
    // extra geo info
    VertexExtraData v[];
} buf_ext[MAX_GEOMETRIES];
layout(set = 0, binding = 4, scalar) buffer readonly restrict buf_gbuf {
    GBuffer gbuffer[];
};
layout(set = 0, binding = 5, scalar) buffer readonly restrict buf_hits {
    Hit hits[];
};
layout(set = 0, binding = 6) uniform sampler2D img_tex[MAX_GLTEXTURES];
layout(set = 0, binding = 7) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 8) uniform sampler2D img_prev_volume_depth;
layout(set = 0, binding = 9, scalar) buffer readonly restrict buf_prev_ssmc {
    SSMCState prev_ssmc[];
};

// --- GRAPH out ---
layout(set = 0, binding = 10) uniform writeonly restrict image2D img_irradiance;
layout(set = 0, binding = 11) uniform writeonly restrict image2D img_moments;

layout(set = 0, binding = 12) uniform writeonly restrict image2D img_volume;
layout(set = 0, binding = 13) uniform writeonly restrict image2D img_volume_moments;
layout(set = 0, binding = 14) uniform writeonly restrict image2D img_volume_depth;
layout(set = 0, binding = 15, rg16f) uniform restrict    image2D img_volume_mv;
layout(set = 0, binding = 16) uniform writeonly restrict image2D img_debug;

// GRAPH buffer out
layout(set = 0, binding = 17, scalar) buffer restrict buf_mc_states {
    MCState mc_states[];
};
layout(set = 0, binding = 18, scalar) buffer restrict buf_light_cache {
    LightCacheVertex light_cache[];
};
layout(set = 0, binding = 19, scalar) buffer restrict buf_dist_mc_states {
    DistanceMCVertex distance_mc_states[];
};
layout(set = 0, binding = 20, scalar) buffer restrict buf_ssmc {
    SSMCState ssmc[];
};

