#pragma once

#include "merian/utils/vector_matrix.hpp"

#include <array>
#include <cstdint>

namespace merian_quake {

struct PlayerData {
    // see PLAYER_* in config.h
    unsigned char flags;
    unsigned char padding0;
    unsigned char padding1;
    unsigned char padding2;
};

struct RTConfig {
    unsigned char flags = 0;
    unsigned char padding0;
    unsigned char padding1;
    unsigned char padding2;
};

struct UniformData {
    merian::float4 cam_x_mu_t; // pos, and fog mu_t in alpha
    merian::float4 cam_w;      // forward, and time_diff in alpha (set to 1. if 0.)
    merian::float4 cam_u;      // up

    merian::float4 prev_cam_x_mu_sx;
    merian::float4 prev_cam_w_mu_sy;
    merian::float4 prev_cam_u_mu_sz;

    // The texnums for sky_rt, sky_bk, sky_lf, sky_ft, sky_up, sky_dn;
    std::array<uint16_t, 6> sky;

    // quake time
    float cl_time;
    uint32_t frame;

    PlayerData player;
    RTConfig rt_config;
};

struct ConstantData {
    merian::float3 sun_color;
    merian::float3 sun_direction;

    float fov;
    float fov_tan_alpha_half;
    float volume_max_t = 1000;
};

struct QuakeRenderInfo {
    // Can be used as push constant.
    // Updated every frame and only valid if render == true
    UniformData uniform;

    // Does only change if a new world is loaded or settings are changed
    ConstantData constant;

    // If this is false do not render, just clear your outputs.
    bool render;
    // Set if new constant data is available. For example, if a new map was loaded, maybe reset
    // stuff?
    bool constant_data_update = true;
};

} // namespace merian_quake
