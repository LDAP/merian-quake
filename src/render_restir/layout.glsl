#extension GL_EXT_ray_tracing                       : enable
#extension GL_EXT_ray_query                         : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout (constant_id = 2) const int MAX_SPP = 4;
layout (constant_id = 3) const int MAX_PATH_LENGTH = 3;
layout (constant_id = 4) const float FOV_TAN_ALPHA_HALF = 0;
layout (constant_id = 5) const float SUN_W_X = 0;
layout (constant_id = 6) const float SUN_W_Y = 0;
layout (constant_id = 7) const float SUN_W_Z = 0;
layout (constant_id = 8) const float SUN_COLOR_R = 0;
layout (constant_id = 9) const float SUN_COLOR_G = 0;
layout (constant_id = 10) const float SUN_COLOR_B = 0;
layout (constant_id = 11) const float VOLUME_MAX_T = 1000.;
layout (constant_id = 12) const uint SEED = 0;
layout (constant_id = 13) const bool DEBUG_OUTPUT_CONNECTED = false;
layout (constant_id = 14) const int DEBUG_OUTPUT_SELECTOR = 0;

#include "../config.h"
#include "../scene_info.glsl.h"
#include "../hit.glsl.h"
#include "reservoir.glsl.h"

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
layout(set = 0, binding = 8, scalar) buffer readonly restrict buf_reservoirs_in {
    Reservoir prev_reservoirs[];
};


// --- GRAPH out ---
layout(set = 0, binding = 9) uniform writeonly restrict image2D img_irradiance;
layout(set = 0, binding = 10) uniform writeonly restrict image2D img_moments;
layout(set = 0, binding = 11) uniform writeonly restrict image2D img_debug;
layout(set = 0, binding = 12, scalar) buffer writeonly restrict buf_reservoirs_out {
    Reservoir reservoirs[];
};
