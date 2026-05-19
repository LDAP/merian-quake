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
// `prev_positions` is appended in lock-step with `vertices` and contains the
// world-space position each vertex held in the previous frame (motion vectors
// for the renderer). The extractors also update each entity's
// mv_prev_origin / mv_prev_angles / mv_prev_blend bookkeeping so the next
// frame can do the same.

void extract_alias_geo(entity_t* ent,
                       std::vector<merian::PackedVertexData>& vertices,
                       std::vector<merian::float3>& prev_positions,
                       std::vector<merian::uint3>& indices);

void extract_brush_entity_geo(entity_t* ent,
                              std::vector<merian::PackedVertexData>& vertices,
                              std::vector<merian::float3>& prev_positions,
                              std::vector<merian::uint3>& indices);

// Compute the world-space (up, right) basis for a sprite entity given its
// model's sprite type. Returns false if the type is unsupported.
bool sprite_world_basis(entity_t* ent,
                        msprite_t* psprite,
                        merian::float3& s_up,
                        merian::float3& s_right);

// Particle billboards. `no_random` makes the per-particle jitter
// reproducible across frames (used by the reference render path).
void extract_particle_geo(std::vector<merian::PackedVertexData>& vertices,
                          std::vector<merian::float3>& prev_positions,
                          std::vector<merian::uint3>& indices,
                          bool no_random,
                          double prev_cl_time);

// Lerp alias vertices in raw (unscaled) object space — positions are direct
// byte-coord lerps, normals are normalized lerps of pre-baked smooth per-pose
// normals (layout: numposes * numverts float3 entries, indexed by original
// vertindex from hdr->meshdesc). The caller puts scale/scale_origin into the
// SceneNode transform. `vertices_dst` / `prev_dst` must hold at least
// hdr->numverts_vbo entries.
void lerp_alias_vertices(aliashdr_t* hdr,
                         const merian::float3* baked_normals,
                         int pose1,
                         int pose2,
                         float blend,
                         int prev_pose1,
                         int prev_pose2,
                         float prev_blend,
                         merian::PackedVertexData* vertices_dst,
                         merian::PackedPrevVertexData* prev_dst);

} // namespace merian_quake
