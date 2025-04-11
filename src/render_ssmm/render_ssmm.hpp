#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"
#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_buffer_array_in.hpp"
#include "merian-nodes/connectors/vk_texture_array_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "merian-nodes/graph/node.hpp"

#include "game/quake_node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

class RendererSSMM : public merian_nodes::Node {
  public:
    RendererSSMM(const merian::ContextHandle& context,
                 const merian::ResourceAllocatorHandle& allocator);

    ~RendererSSMM();

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

    merian::ShaderModuleHandle rt_shader;
    merian::ShaderModuleHandle clear_shader;

    merian_nodes::VkBufferArrayInHandle con_vtx =
        merian_nodes::VkBufferArrayIn::compute_read("vtx");
    merian_nodes::VkBufferArrayInHandle con_prev_vtx =
        merian_nodes::VkBufferArrayIn::compute_read("prev_vtx");
    merian_nodes::VkBufferArrayInHandle con_idx =
        merian_nodes::VkBufferArrayIn::compute_read("idx");
    merian_nodes::VkBufferArrayInHandle con_ext =
        merian_nodes::VkBufferArrayIn::compute_read("ext");
    merian_nodes::ManagedVkBufferInHandle con_gbuffer =
        merian_nodes::ManagedVkBufferIn::compute_read("gbuffer");
    merian_nodes::ManagedVkBufferInHandle con_hits =
        merian_nodes::ManagedVkBufferIn::compute_read("hits");
    merian_nodes::VkTextureArrayInHandle con_textures =
        merian_nodes::VkTextureArrayIn::compute_read("textures");
    merian_nodes::VkTLASInHandle con_tlas = merian_nodes::VkTLASIn::compute_read("tlas");
    merian_nodes::ManagedVkImageInHandle con_mv =
        merian_nodes::ManagedVkImageIn::compute_read("mv");
    merian_nodes::ManagedVkBufferInHandle con_prev_ssmc =
        merian_nodes::ManagedVkBufferIn::compute_read("prev_ssmc", 1);

    merian_nodes::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian_nodes::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian_nodes::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian_nodes::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");

    merian_nodes::ManagedVkImageOutHandle con_irradiance;
    merian_nodes::ManagedVkImageOutHandle con_moments;

    merian_nodes::ManagedVkBufferOutHandle con_ssmc;

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
    uint ml_max_n = 1024;
    float ml_min_alpha = 0.01;
    uint smis_group_size = 5;

    uint32_t seed = 0;
    bool randomize_seed = true;
};
