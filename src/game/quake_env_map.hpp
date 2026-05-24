#pragma once

#include "merian-shaders/scene/env_map.hpp"
#include "merian-shaders/utils/texture-manager-data.slangh"
#include "merian/utils/vector_matrix.hpp"

#include <array>
#include <string>

namespace merian_quake {

class QuakeClassicSkyEnvMap : public merian::EnvMap {
  public:
    QuakeClassicSkyEnvMap(merian::TextureID solid, merian::TextureID alpha)
        : solid(solid), alpha(alpha) {}

    merian::SlangComposition::SlangModule get_slang_module() const override {
        return merian::SlangComposition::SlangModule::from_path("shader/quake_sky.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::QuakeClassicSkyEnvMap";
    }

    void write_to(merian::ShaderCursor cursor) const override {
        cursor["solid"] = solid;
        cursor["alpha"] = alpha;
        cursor["sun_dir"] = sun_dir;
        cursor["sun_color"] = sun_color;
    }

    void set_sun(const merian::float3& dir, const merian::float3& color) {
        sun_dir = dir;
        sun_color = color;
    }

  private:
    merian::TextureID solid;
    merian::TextureID alpha;
    merian::float3 sun_dir{0, 0, 1};
    merian::float3 sun_color{0};
};

class QuakeCubemapSkyEnvMap : public merian::EnvMap {
  public:
    QuakeCubemapSkyEnvMap(std::array<merian::TextureID, 6> faces) : faces(faces) {}

    merian::SlangComposition::SlangModule get_slang_module() const override {
        return merian::SlangComposition::SlangModule::from_path("shader/quake_sky.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::QuakeCubemapSkyEnvMap";
    }

    void write_to(merian::ShaderCursor cursor) const override {
        cursor["rt"] = faces[0];
        cursor["bk"] = faces[1];
        cursor["lf"] = faces[2];
        cursor["ft"] = faces[3];
        cursor["up"] = faces[4];
        cursor["dn"] = faces[5];
        cursor["sun_dir"] = sun_dir;
        cursor["sun_color"] = sun_color;
    }

    void set_sun(const merian::float3& dir, const merian::float3& color) {
        sun_dir = dir;
        sun_color = color;
    }

  private:
    std::array<merian::TextureID, 6> faces;
    merian::float3 sun_dir{0, 0, 1};
    merian::float3 sun_color{0};
};

} // namespace merian_quake
