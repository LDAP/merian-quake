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

    MeshVertexData get_vertices() const override {
        return HostPacked<merian::PackedVertexData>{vertices.data()};
    }
    MeshPrevVertexData get_prev_vertices() const override {
        return std::monostate{};
    }
    MeshIndexData get_indices() const override {
        return HostPacked<void>{indices.data()};
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

    MeshVertexData get_vertices() const override {
        return HostPacked<merian::PackedVertexData>{vertices.data()};
    }
    MeshPrevVertexData get_prev_vertices() const override {
        if (prev_vertices.empty())
            return std::monostate{};
        return HostPacked<merian::PackedPrevVertexData>{prev_vertices.data()};
    }
    MeshIndexData get_indices() const override {
        return HostPacked<void>{indices.data()};
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

    MeshVertexData get_vertices() const override {
        return DeviceStaged{vb_staging};
    }
    MeshPrevVertexData get_prev_vertices() const override {
        if (!prev_vb_staging)
            return std::monostate{};
        return DeviceStaged{prev_vb_staging};
    }
    MeshIndexData get_indices() const override {
        return DeviceLocal{ib_shared};
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

    MeshVertexData get_vertices() const override {
        return DeviceLocal{vb};
    }
    MeshPrevVertexData get_prev_vertices() const override {
        return std::monostate{};
    }
    MeshIndexData get_indices() const override {
        return DeviceLocal{ib};
    }
};

} // namespace merian_quake
