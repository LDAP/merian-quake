// Quake material model: registered with merian::MaterialSystem from QuakeScene.
// Payload layout mirrors QuakeMaterialPayload in src/game/quake_material.hpp.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;

namespace merian {

// Sentinel matching TextureID(-1) used in QuakeMaterialPayload for "no texture".
static const TextureID kQuakeNoTexture = TextureID(0xFFFF);

// Surface flag values, must match MAT_FLAGS_* in res/shader/config.h.
static const uint kQuakeSurfaceNone = 0;
static const uint kQuakeSurfaceLava = 1;
static const uint kQuakeSurfaceSlime = 2;
static const uint kQuakeSurfaceTele = 3;
static const uint kQuakeSurfaceWater = 4;
static const uint kQuakeSurfaceSky = 5;
static const uint kQuakeSurfaceWaterfall = 6;
static const uint kQuakeSurfaceSprite = 7;
static const uint kQuakeSurfaceSolid = 8;

struct QuakeMaterial : MaterialModel {
    typealias MaterialSampleImpl = QuakeMaterialSample;

    MaterialHeader header;

    // Compact payload (must match QuakeMaterialPayload C++-side layout).
    TextureID base_tex;
    TextureID fullbright_tex;
    TextureID normal_tex;
    TextureID gloss_tex;
    uint16_t  surface_flags;  // MAT_FLAGS_*
    uint8_t   alpha_mode;     // 0: use texture alpha; [1,15] map to [0,1] opaque
    uint8_t   _pad;

    QuakeMaterialSample
    get_sample(const ShadingData sd, const TextureManager tm, const LODSampler lod) {
        float3 albedo = float3(1, 1, 1);
        if (base_tex != kQuakeNoTexture) {
            float4 tex_color = lod.sample(tm.get_sampler2D(base_tex), sd.uv);
            albedo = tex_color.rgb;
        }
        QuakeMaterialSample s;
        s.brdf = LambertDiffuseBRDF(albedo);
        return s;
    }
}

struct QuakeMaterialSample : MaterialSample {
    typealias BSDFImpl = LambertDiffuseBRDF;
    LambertDiffuseBRDF brdf;
    BSDFImpl get_bsdf() {
        return brdf;
    }
}

}
