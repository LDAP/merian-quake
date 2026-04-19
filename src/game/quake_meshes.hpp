#pragma once

#include "merian-shaders/scene/scene-data.slangh"
#include "merian-shaders/scene/scene.hpp"

#include <vector>

namespace merian_quake {

// Static brush geometry: pre-baked PackedVertexData + uint3 indices,
// world-space coords, one per material partition. Lifetime spans a map
// load; never marked dirty after creation.
class QuakeBrushMesh : public merian::Mesh {
  public:
    std::vector<merian::PackedVertexData> vertices;
    std::vector<merian::uint3> indices;

    uint32_t get_vertex_count() const override {
        return static_cast<uint32_t>(vertices.size());
    }
    uint32_t get_primitive_count() const override {
        return static_cast<uint32_t>(indices.size());
    }

    merian::float3 get_position(uint32_t v) const override {
        return vertices[v].position;
    }
    merian::float3 get_normal(uint32_t v) const override;
    merian::float2 get_uv(uint32_t v) const override {
        return merian::float2(vertices[v].uv);
    }
    merian::float4 get_tangent(uint32_t v) const override;

    merian::uint3 get_indices(uint32_t p) const override {
        return indices[p];
    }
};

} // namespace merian_quake
