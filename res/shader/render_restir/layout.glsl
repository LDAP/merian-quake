#extension GL_EXT_ray_tracing                       : enable
#extension GL_EXT_ray_query                         : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout (constant_id = 2) const int SURFACE_SPP = 4;
layout (constant_id = 3) const float FOV_TAN_ALPHA_HALF = 0;
layout (constant_id = 4) const float SUN_W_X = 0;
layout (constant_id = 5) const float SUN_W_Y = 0;
layout (constant_id = 6) const float SUN_W_Z = 0;
layout (constant_id = 7) const float SUN_COLOR_R = 0;
layout (constant_id = 8) const float SUN_COLOR_G = 0;
layout (constant_id = 9) const float SUN_COLOR_B = 0;
layout (constant_id = 10) const float VOLUME_MAX_T = 1000.;
layout (constant_id = 11) const uint SEED = 0;
layout (constant_id = 12) const bool DEBUG_OUTPUT_CONNECTED = false;
layout (constant_id = 13) const int DEBUG_OUTPUT_SELECTOR = 0;
layout (constant_id = 14) const bool VISIBILITY_SHADE = false;
layout (constant_id = 15) const float TEMPORAL_NORMAL_REJECT_COS = 0.8;
layout (constant_id = 16) const float TEMPORAL_DEPTH_REJECT = 0.1;
layout (constant_id = 17) const float SPATIAL_NORMAL_REJECT_COS = 0.8;
layout (constant_id = 18) const float SPATIAL_DEPTH_REJECT = 0.1;
layout (constant_id = 19) const int TEMPORAL_CLAMP_M = 32 * 20;
layout (constant_id = 20) const int SPATIAL_RADIUS = 30;
layout (constant_id = 21) const int TEMPORAL_BIAS_CORRECTION = 0;
layout (constant_id = 22) const int SPATIAL_BIAS_CORRECTION = 0;
layout (constant_id = 23) const int SUBGROUP_SIZE = 32;
layout (constant_id = 24) const float BOILING_FILTER_STRENGTH = 0.0;
layout (constant_id = 25) const int SPATIAL_REUSE_ITERATIONS = 1;
layout (constant_id = 26) const int APPLY_MV = 0;

#include "../config.h"
#include "../scene_info.glsl.h"
#include "../hit.glsl.h"
#include "restir_di_reservoir.glsl.h"

#include "merian-shaders/image_buffer.glsl.h"

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

MAKE_GBUFFER_READONLY_LAYOUT(set = 0, binding = 4, gbuffer);

MAKE_GBUFFER_READONLY_LAYOUT(set = 0, binding = 5, prev_gbuffer);

layout(set = 0, binding = 6, scalar) buffer readonly restrict buf_hits {
    Hit hits[];
};
layout(set = 0, binding = 7) uniform sampler2D img_tex[MAX_GLTEXTURES];
layout(set = 0, binding = 8) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 9, scalar) buffer readonly restrict buf_prev_reservoirs_in {
    ReSTIRDIReservoir reservoirs_prev_read[];
};
layout(set = 0, binding = 10) uniform sampler2D img_mv;


// --- GRAPH out ---
layout(set = 0, binding = 11) uniform writeonly restrict image2D img_irradiance;
layout(set = 0, binding = 12) uniform writeonly restrict image2D img_moments;
layout(set = 0, binding = 13) uniform writeonly restrict image2D img_debug;



layout(set = 1, binding = 0, scalar) buffer readonly buf_reservoirs_in {
    ReSTIRDIReservoir reservoirs_spatial_read[];
};

layout(set = 1, binding = 1, scalar) buffer buf_reservoirs_out {
    ReSTIRDIReservoir reservoirs[];
};
