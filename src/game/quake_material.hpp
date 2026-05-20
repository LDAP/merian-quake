#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"

#include <cstdint>
#include <cstring>

namespace merian_quake {

// Mirrors the trailing fields of merian::QuakeMaterial in
// res/shader/quake-material.slang — order, sizes and packing must stay in sync.
// surface_flags carries MAT_TYPE_* (res/shader/config.h); brush variants alias
// Quake's SURF_DRAW* bits so msurface_t::flags can be assigned directly.
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

// kQuakeNoTexture on the Slang side.
static constexpr merian::TextureID QUAKE_NO_TEXTURE = merian::TextureID(0xFFFF);

struct QuakeMaterial : merian::Material {
    QuakeMaterialPayload payload;

    QuakeMaterial() {
        // (-1) disables MaterialSystem::alpha_test's alpha-mask discard.
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

// Resolved against the shader search path (the app adds res/).
inline constexpr const char* QUAKE_MATERIAL_SLANG_MODULE_PATH = "shader/quake-material.slang";
inline constexpr const char* QUAKE_MATERIAL_SLANG_TYPE_NAME = "merian::QuakeMaterial";

} // namespace merian_quake
