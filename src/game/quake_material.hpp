#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"

#include <cstdint>
#include <cstring>

namespace merian_quake {

// Surface flag values mirror MAT_FLAGS_* in res/shader/config.h. Kept inline
// here to avoid forcing every translation unit that touches QuakeMaterial to
// drag in the shader-side config header.
enum class QuakeSurfaceFlags : uint16_t {
    None = 0,
    Lava = 1,
    Slime = 2,
    Tele = 3,
    Water = 4,
    Sky = 5,
    Waterfall = 6,
    Sprite = 7,
    Solid = 8,
};

// 12-byte payload that mirrors the trailing fields of merian::QuakeMaterial in
// res/shader/quake-material.slang. The Slang side reads this via the
// MaterialPayload blob, so the order, sizes and packing must stay in sync.
struct QuakeMaterialPayload {
    merian::TextureID fullbright_tex{};
    merian::TextureID normal_tex{};
    merian::TextureID gloss_tex{};
    uint16_t surface_flags{};
    uint8_t alpha_mode{};
    uint8_t _pad{};
};
static_assert(sizeof(QuakeMaterialPayload) == 10,
              "QuakeMaterialPayload layout must match Slang QuakeMaterial");

// Sentinel value used by the Slang side (kQuakeNoTexture) for "no texture".
static constexpr merian::TextureID QUAKE_NO_TEXTURE = merian::TextureID(0xFFFF);

struct QuakeMaterial : merian::Material {
    QuakeMaterialPayload payload;

    QuakeMaterial() {
        // alpha_texture_id is consumed by MaterialSystem::alpha_test for
        // alpha-mask discard. Default to "no alpha mask".
        header.alpha_texture_id = merian::TextureID(-1);
        payload.fullbright_tex = QUAKE_NO_TEXTURE;
        payload.normal_tex = QUAKE_NO_TEXTURE;
        payload.gloss_tex = QUAKE_NO_TEXTURE;
    }

    uint32_t get_payload_size() const override {
        return static_cast<uint32_t>(sizeof(QuakeMaterialPayload));
    }

    void write_payload(void* dest) const override {
        std::memcpy(dest, &payload, sizeof(QuakeMaterialPayload));
    }
};

// Slang module path (relative to a shader search-path entry; the merian-quake
// app adds res/ as one of its search paths).
inline constexpr const char* QUAKE_MATERIAL_SLANG_MODULE_PATH = "shader/quake-material.slang";
inline constexpr const char* QUAKE_MATERIAL_SLANG_TYPE_NAME = "merian::QuakeMaterial";

} // namespace merian_quake
