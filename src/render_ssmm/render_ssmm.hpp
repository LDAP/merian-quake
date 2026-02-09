#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"
#include "merian-nodes/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-nodes/connectors/connector_utils.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "merian-nodes/graph/node.hpp"

#include "game/quake_node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

class RendererSSMM : public merian::Node {
  public:
    RendererSSMM();

    virtual void initialize(const merian::ContextHandle& context,
                            const merian::ResourceAllocatorHandle& allocator) override;

    ~RendererSSMM();

    // -----------------------------------------------------

    std::vector<merian::InputConnectorHandle> describe_inputs() override;

    std::vector<merian::OutputConnectorHandle>
    describe_outputs(const merian::NodeIOLayout& io_layout) override;

    NodeStatusFlags
    on_connected(const merian::NodeIOLayout& io_layout,
                 const merian::DescriptorSetLayoutHandle& graph_desc_set_layout) override;

    void process(merian::GraphRun& run,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
    merian::ContextHandle context;
    merian::ResourceAllocatorHandle allocator;

    merian::EntryPointHandle rt_shader;
    merian::EntryPointHandle clear_shader;

    merian::VkBufferInHandle con_vtx = merian::VkBufferIn::compute_read("vtx");
    merian::VkBufferInHandle con_prev_vtx = merian::VkBufferIn::compute_read("prev_vtx");
    merian::VkBufferInHandle con_idx = merian::VkBufferIn::compute_read("idx");
    merian::VkBufferInHandle con_ext = merian::VkBufferIn::compute_read("ext");
    merian::GBufferInHandle con_gbuffer = merian::GBufferIn::compute_read("gbuffer");
    merian::VkBufferInHandle con_hits = merian::VkBufferIn::compute_read("hits");
    merian::VkSampledImageInHandle con_textures =
        merian::VkSampledImageIn::compute_read("textures");
    merian::VkTLASInHandle con_tlas = merian::VkTLASIn::compute_read("tlas");
    merian::VkSampledImageInHandle con_mv = merian::VkSampledImageIn::compute_read("mv");
    merian::VkBufferInHandle con_prev_ssmc = merian::VkBufferIn::compute_read("prev_ssmc", 1);

    merian::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");

    merian::ManagedVkImageOutHandle con_irradiance;
    merian::ManagedVkImageOutHandle con_moments;

    merian::ManagedVkBufferOutHandle con_ssmc;

    //-----------------------------------------------------

    merian::DescriptorSetLayoutHandle graph_desc_set_layout;
    merian::PipelineLayoutHandle pipe_layout;

    merian::PipelineHandle pipe;
    merian::PipelineHandle clear_pipe;

    // ----------------------------------------------------

    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;

    int32_t spp = 1;
    float surf_bsdf_p = 0.15;

    float ml_prior_n = .20;
    uint32_t ml_max_n = 1024;
    float ml_min_alpha = 0.01;
    uint32_t smis_group_size = 5;

    uint32_t seed = 0;
    bool randomize_seed = true;
};
