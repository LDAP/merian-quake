#pragma once

#include "merian-shaders/scene/scene-data.slangh"
#include "merian/utils/vector_matrix.hpp"

#include <vector>

extern "C" {
#include "quakedef.h"
}

namespace merian_quake {

// Per-frame geometry extraction for dynamic Quake entities.
//
// All helpers append world-space PackedVertexData triangles to the supplied
// vectors. Tangents are written as a zero placeholder; downstream BSDFs are
// expected to reconstruct from triangle edges. Per-vertex material data
// (texnums, alpha mode, surface flags) is intentionally dropped — the new
// pipeline carries that on the QuakeMaterial payload bound per-mesh.
//
// The extractors update each entity's mv_prev_origin / mv_prev_angles /
// mv_prev_blend bookkeeping so future per-frame deltas (motion vectors)
// have the previous-frame state available; the prev-vertex stream itself
// is not yet uploaded to the Scene.

void extract_alias_geo(entity_t* ent,
                       std::vector<merian::PackedVertexData>& vertices,
                       std::vector<merian::uint3>& indices);

void extract_brush_entity_geo(entity_t* ent,
                              std::vector<merian::PackedVertexData>& vertices,
                              std::vector<merian::uint3>& indices);

void extract_sprite_geo(entity_t* ent,
                        std::vector<merian::PackedVertexData>& vertices,
                        std::vector<merian::uint3>& indices);

// Dispatches based on ent->model->type. Silently skips entities without a
// model or with an unsupported type.
void extract_entity_geo(entity_t* ent,
                        std::vector<merian::PackedVertexData>& vertices,
                        std::vector<merian::uint3>& indices);

// Particle billboards. `no_random` makes the per-particle jitter
// reproducible across frames (used by the reference render path).
void extract_particle_geo(std::vector<merian::PackedVertexData>& vertices,
                          std::vector<merian::uint3>& indices,
                          bool no_random,
                          double prev_cl_time);

} // namespace merian_quake
