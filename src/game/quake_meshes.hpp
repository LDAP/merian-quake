#pragma once

#include "merian-shaders/scene/scene-data.slangh"
#include "merian-shaders/scene/scene.hpp"

#include <vector>

namespace merian_quake {

// Static brush geometry in world space. HostPacked source, never dirty after creation.
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

    Mesh::MeshData get_data() const override {
        return Mesh::HostPacked{vertices.data(), nullptr, indices.data()};
    }
};

// Per-frame CPU-rebuilt mesh (sprites, particles). Stores prev_vertices for motion vectors.
class QuakeHostDynamicMesh : public merian::Mesh {
  public:
    std::vector<merian::PackedVertexData> vertices;
    std::vector<merian::PackedPrevVertexData> prev_vertices;
    std::vector<merian::uint3> indices;

    uint32_t get_vertex_count() const override {
        return static_cast<uint32_t>(vertices.size());
    }
    uint32_t get_primitive_count() const override {
        return static_cast<uint32_t>(indices.size());
    }

    Mesh::MeshData get_data() const override {
        return Mesh::HostPacked{
            vertices.data(),
            prev_vertices.empty() ? nullptr : prev_vertices.data(),
            indices.data(),
        };
    }
};

// One per visible alias entity. Host-visible staging buffers filled with lerped
// model-space vertices each frame; Scene copies them to device-local.
class AliasInstanceMesh : public merian::Mesh {
  public:
    merian::BufferHandle vb_staging;
    merian::BufferHandle prev_vb_staging;
    merian::BufferHandle ib_shared; // borrowed from AliasModelInfo
    uint32_t vertex_count = 0;
    uint32_t primitive_count = 0;

    uint32_t get_vertex_count() const override {
        return vertex_count;
    }
    uint32_t get_primitive_count() const override {
        return primitive_count;
    }

    Mesh::MeshData get_data() const override {
        return Mesh::DeviceStaged{vb_staging, prev_vb_staging, ib_shared};
    }
};

// One per (visible brush entity, material partition). Static device-local geometry
// built once; only the parent SceneNode transform changes per frame.
class BrushEntityMesh : public merian::Mesh {
  public:
    merian::BufferHandle vb;
    merian::BufferHandle ib;
    uint32_t vertex_count = 0;
    uint32_t primitive_count = 0;

    uint32_t get_vertex_count() const override {
        return vertex_count;
    }
    uint32_t get_primitive_count() const override {
        return primitive_count;
    }

    Mesh::MeshData get_data() const override {
        return Mesh::DeviceLocal{vb, {}, ib};
    }
};

} // namespace merian_quake
