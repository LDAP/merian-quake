#include "config.h"

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

layout(set = 1, binding = BINDING_VTX_BUF) buffer buf_vtx_t {
  // 3x float vertex data for every vertex
  float v[];
} buf_vtx[];

layout(set = 1, binding = BINDING_IDX_BUF) buffer buf_idx_t {
  // index data for every instance
  uint i[];
} buf_idx[];

layout(std430, set = 1, binding = BINDING_EXT_BUF) buffer buf_ext_t {
  // extra geo info
  uint v[];
} buf_ext[];

layout(set = 1, binding = BINDING_IMG_TEX) uniform sampler2D img_tex[];
layout(set = 1, binding = BINDING_TLAS) uniform accelerationStructureEXT rt_accel;
