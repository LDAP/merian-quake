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

class RendererMarkovChain : public merian_nodes::Node {
  public:
    // Per-frame data and updates
    struct FrameData {
        merian::PipelineHandle pipe;
        merian::PipelineHandle clear_pipe;
        merian::PipelineHandle volume_pipe;
        merian::PipelineHandle volume_forward_project_pipe;
    };

  public:
    RendererMarkovChain(const merian::ContextHandle& context,
                        const merian::ResourceAllocatorHandle& allocator);

    ~RendererMarkovChain();

    // -----------------------------------------------------

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs(const merian_nodes::NodeIOLayout& io_layout) override;

    NodeStatusFlags
    on_connected(const merian_nodes::NodeIOLayout& io_layout,
                 const merian::DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(merian_nodes::GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian_nodes::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
    const merian::ContextHandle context;
    const merian::ResourceAllocatorHandle allocator;

    merian::ShaderModuleHandle rt_shader;
    merian::ShaderModuleHandle clear_shader;
    merian::ShaderModuleHandle volume_shader;
    merian::ShaderModuleHandle volume_forward_project_shader;

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
    merian_nodes::ManagedVkImageInHandle con_prev_volume_depth =
        merian_nodes::ManagedVkImageIn::compute_read("prev_volume_depth", 1);
    merian_nodes::ManagedVkImageInHandle con_mv =
        merian_nodes::ManagedVkImageIn::transfer_src("mv");

    merian_nodes::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian_nodes::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian_nodes::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian_nodes::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");

    merian_nodes::ManagedVkImageOutHandle con_irradiance;
    merian_nodes::ManagedVkImageOutHandle con_debug;
    merian_nodes::ManagedVkImageOutHandle con_moments;
    merian_nodes::ManagedVkImageOutHandle con_volume;
    merian_nodes::ManagedVkImageOutHandle con_volume_moments;
    merian_nodes::ManagedVkImageOutHandle con_volume_depth;
    merian_nodes::ManagedVkImageOutHandle con_volume_mv;

    merian_nodes::ManagedVkBufferOutHandle con_markovchain;
    merian_nodes::ManagedVkBufferOutHandle con_lightcache;
    merian_nodes::ManagedVkBufferOutHandle con_volume_distancemc;

    //-----------------------------------------------------

    merian::DescriptorSetLayoutHandle graph_desc_set_layout;
    merian::PipelineLayoutHandle pipe_layout;

    merian::PipelineHandle pipe;
    merian::PipelineHandle clear_pipe;
    merian::PipelineHandle volume_pipe;
    merian::PipelineHandle volume_forward_project_pipe;

    // ----------------------------------------------------

    // Spec constants
    // https://gpuopen.com/learn/rdna-performance-guide/ recommends 8x4
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;
    int32_t spp = 1;
    int32_t volume_spp = 0;
    int32_t max_path_length = 3;
    VkBool32 use_light_cache_tail = false;
    int32_t adaptive_sampling = 0;

    VkBool32 volume_use_light_cache = 0;
    float volume_particle_size_um = 25.0;

    bool dump_mc = false;

    int32_t mc_samples = 5;
    float mc_samples_adaptive_prob = 0.7;
    int32_t distance_mc_samples = 3;

    VkBool32 mc_fast_recovery = true;

    float light_cache_levels = 32.0;
    float light_cache_tan_alpha_half = 0.002;
    uint32_t light_cache_buffer_size = 4000000;

    uint32_t mc_adaptive_buffer_size = 32777259;
    uint32_t mc_static_buffer_size = 800009;
    float mc_adaptive_grid_tan_alpha_half = 0.003;
    float mc_static_grid_width = 25.3;
    int32_t mc_adaptive_grid_levels = 10;

    int32_t distance_mc_grid_width = 25;

    bool volume_forward_project = true;

    float surf_bsdf_p = 0.15;
    float volume_phase_p = 0.3;
    float dir_guide_prior = 0.2;
    float dist_guide_p = 0.0;

    uint32_t distance_mc_vertex_state_count = 10;

    uint32_t seed = 0;
    bool randomize_seed = true;
    int debug_output_selector = 0;
};
