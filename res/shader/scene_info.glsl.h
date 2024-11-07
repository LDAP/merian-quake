#ifndef _SCENE_INFO_H
#define _SCENE_INFO_H

#include "merian-shaders/gbuffer.glsl.h"

// See quake_node.hpp
struct VertexExtraData {
    uint16_t texnum_alpha;
    uint16_t texnum_fb_flags;

    uint n0_gloss_norm;
    uint n1_brush;
    uint n2;

    f16mat3x2 st;
};

struct UniformData {
    vec4 cam_x; // contains mu_t in alpha
    vec4 cam_w; // contains last cl_time - cl_time (or 1 if paused) in alpha
    vec4 cam_u;
    vec4 prev_cam_x; // contains mu_s.r in alpha
    vec4 prev_cam_w; // contains mu_s.g in alpha
    vec4 prev_cam_u; // contains mu_s.b in alpha

    uint sky_rt_bk, sky_lf_ft, sky_up_dn;

    float cl_time; // quake time
    uint frame;
    uint player;    // see `PlayerData` in quake_node.hpp
    uint rt_config; // see `RTConfig` in in quake_node.hpp
};

#define rt_flags() (params.rt_config & 0xff)
#define MU_T params.cam_x.a
#define MU_S vec3(params.prev_cam_x.a, params.prev_cam_w.a, params.prev_cam_u.a)
#define TIME_DIFF params.cam_w.a


#endif // _SCENE_INFO_H
