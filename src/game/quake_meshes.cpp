#include "quake_meshes.hpp"

#include "merian/utils/normal_encoding.hpp"

namespace merian_quake {

merian::float3 QuakeBrushMesh::get_normal(const uint32_t v) const {
    return merian::decode_normal(vertices[v].encoded_normal);
}

merian::float4 QuakeBrushMesh::get_tangent(const uint32_t v) const {
    const uint32_t enc = vertices[v].encoded_tangent;
    const merian::float3 t = merian::decode_normal(enc & ~1u);
    const float sign = (enc & 1u) != 0u ? -1.f : 1.f;
    return merian::float4(t.x, t.y, t.z, sign);
}

} // namespace merian_quake
