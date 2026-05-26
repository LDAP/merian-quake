#pragma once

#include "game/quake_draw.hpp"

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <array>
#include <cstdint>

namespace merian_quake {

// Replays UIDrawCommands as premultiplied-alpha textured quads.
class QuakeUiNode : public merian::Node {
  public:
    struct Vertex {
        std::array<float, 2> pos;
        std::array<float, 2> uv;
        uint32_t rgba;
    };
    struct Batch {
        int32_t texnum;
        uint32_t first_index;
        uint32_t index_count;
        UIScissor scissor;
    };

    QuakeUiNode();
    ~QuakeUiNode() override;

    std::vector<merian::InputConnectorDescriptor> describe_inputs() override;

    std::vector<merian::OutputConnectorDescriptor>
    describe_outputs(const merian::NodeIOLayout& io_layout) override;

    void initialize(const merian::ContextHandle& context,
                    const merian::ResourceAllocatorHandle& allocator) override;

    void process(merian::GraphRun& run,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& props) override;

  private:
    void ensure_pipeline(vk::Format color_format);
    void ensure_white_texture(const merian::CommandBufferHandle& cmd);

    merian::ContextHandle context;
    merian::ResourceAllocatorHandle allocator;

    merian::PtrInHandle<UIDrawCommands> con_ui_draw_commands =
        merian::PtrIn<UIDrawCommands>::create();
    merian::PtrInHandle<merian::Scene> con_scene = merian::PtrIn<merian::Scene>::create();
    merian::VkSampledImageInHandle con_extent_ref = merian::VkSampledImageIn::fragment_read();
    merian::ManagedVkImageOutHandle con_out;

    vk::Extent3D extent{};

    merian::DescriptorSetLayoutHandle descriptor_layout;
    merian::PipelineLayoutHandle pipeline_layout;
    merian::PipelineHandle pipeline;
    vk::Format current_format = vk::Format::eUndefined;

    merian::BufferHandle vertex_buffer;
    merian::BufferHandle index_buffer;
    merian::TextureHandle white_texture;

    // Reused across frames to keep capacity.
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Batch> batches;

    struct Stats {
        uint32_t cmds = 0;
        uint32_t vertices = 0;
        uint32_t indices = 0;
        uint32_t draw_calls = 0;
        uint32_t canvas_events = 0;
        uint32_t scissor_events = 0;
        uint32_t color_events = 0;
        uint32_t unique_textures = 0;
    };
    Stats last_stats{};
};

using QuakeUiNodeHandle = std::shared_ptr<QuakeUiNode>;

} // namespace merian_quake
