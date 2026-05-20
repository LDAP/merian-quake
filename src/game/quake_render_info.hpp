#pragma once

#include "merian/utils/vector_matrix.hpp"

#include <array>
#include <cstdint>

namespace merian_quake {

struct PlayerData {
    unsigned char flags; // PLAYER_* in config.h
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

    // sky_rt, sky_bk, sky_lf, sky_ft, sky_up, sky_dn texnums
    std::array<uint16_t, 6> sky;

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
    // Push-constant friendly; only valid when render == true.
    UniformData uniform;

    // Changes only on world load / setting changes.
    ConstantData constant;

    // false: do not render, clear outputs.
    bool render;
    // true: new constant data available (e.g. a new map loaded).
    bool constant_data_update = true;
};

} // namespace merian_quake
