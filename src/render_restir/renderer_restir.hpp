#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"
#include "merian-nodes/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-nodes/connectors/connector_utils.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "merian-nodes/graph/node.hpp"

#include "game/quake_node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

class RendererRESTIR : public merian_nodes::Node {
  public:
    // Per-frame data and updates
    struct Pipelines {
        merian::PipelineHandle clear;
        merian::PipelineHandle generate_samples;
        merian::PipelineHandle temporal_reuse;
        merian::PipelineHandle spatial_reuse;
        merian::PipelineHandle shade;

        bool recreate = true;
    };

  public:
    RendererRESTIR(const merian::ContextHandle& context,
                   const merian::ResourceAllocatorHandle& allocator);

    ~RendererRESTIR() override;

    // -----------------------------------------------------

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs(const merian_nodes::NodeIOLayout& io_layout) override;

    NodeStatusFlags
    on_connected(const merian_nodes::NodeIOLayout& io_layout,
                 const merian::DescriptorSetLayoutHandle& graph_desc_set_layout) override;

    void process(merian_nodes::GraphRun& run,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian_nodes::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
    const merian::ContextHandle context;
    const merian::ResourceAllocatorHandle allocator;

    merian::ShaderModuleHandle generate_samples_shader;
    merian::ShaderModuleHandle temporal_reuse_shader;
    merian::ShaderModuleHandle spatial_reuse_shader;
    merian::ShaderModuleHandle shade_shader;
    merian::ShaderModuleHandle clear_shader;

    merian_nodes::VkBufferInHandle con_vtx = merian_nodes::VkBufferIn::compute_read("vtx");
    merian_nodes::VkBufferInHandle con_prev_vtx =
        merian_nodes::VkBufferIn::compute_read("prev_vtx");
    merian_nodes::VkBufferInHandle con_idx = merian_nodes::VkBufferIn::compute_read("idx");
    merian_nodes::VkBufferInHandle con_ext = merian_nodes::VkBufferIn::compute_read("ext");
    merian_nodes::GBufferInHandle con_gbuffer = merian_nodes::GBufferIn::compute_read("gbuffer");
    merian_nodes::GBufferInHandle con_prev_gbuffer =
        merian_nodes::GBufferIn::compute_read("prev_gbuffer", 1);
    merian_nodes::VkBufferInHandle con_hits = merian_nodes::VkBufferIn::compute_read("hits");

    merian_nodes::VkSampledImageInHandle con_textures =
        merian_nodes::VkSampledImageIn::compute_read("textures");
    merian_nodes::VkTLASInHandle con_tlas = merian_nodes::VkTLASIn::compute_read("tlas");

    merian_nodes::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian_nodes::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian_nodes::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian_nodes::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");
    merian_nodes::VkBufferInHandle con_reservoirs_in =
        merian_nodes::VkBufferIn::compute_read("reservoirs", 1);
    merian_nodes::VkSampledImageInHandle con_mv =
        merian_nodes::VkSampledImageIn::compute_read("mv");

    merian_nodes::ManagedVkImageOutHandle con_irradiance;
    merian_nodes::ManagedVkImageOutHandle con_moments;
    merian_nodes::ManagedVkImageOutHandle con_debug;
    merian_nodes::ManagedVkBufferOutHandle con_reservoirs_out;

    //-----------------------------------------------------

    merian::DescriptorSetLayoutHandle graph_desc_set_layout;
    merian::PipelineLayoutHandle pipe_layout;

    Pipelines pipelines;

    // ----------------------------------------------------

    merian::DescriptorSetLayoutHandle reservoir_pingpong_layout;
    merian::BufferHandle pong_buffer;

    // Spec constants
    static constexpr uint32_t LOCAL_SIZE_X = 8;
    static constexpr uint32_t LOCAL_SIZE_Y = 8;

    int32_t spp = 1;

    uint32_t seed = 0;
    bool randomize_seed = true;
    int32_t debug_output_selector = 0;
    int32_t apply_mv = 0;

    bool temporal_reuse_enable = false;
    int32_t spatial_reuse_iterations = 0;
    bool visibility_shade = false;
    int32_t temporal_clamp_m = 32 * 20;
    float boiling_filter_strength = 0.0;

    float temporal_normal_reject_cos = 0.96;
    float temporal_depth_reject_percent = 0.1;
    float spatial_normal_reject_cos = 0.96;
    float spatial_depth_reject_percent = 0.1;
    int32_t spatial_radius = 30;
    int32_t temporal_bias_correction = 0;
    int32_t spatial_bias_correction = 0;
};
