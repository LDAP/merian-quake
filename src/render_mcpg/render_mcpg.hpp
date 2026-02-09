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
#include "merian/shader/entry_point.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

class RendererMarkovChain : public merian::Node {
  public:
    // Per-frame data and updates
    struct FrameData {
        merian::PipelineHandle pipe;
        merian::PipelineHandle clear_pipe;
        merian::PipelineHandle volume_pipe;
        merian::PipelineHandle volume_forward_project_pipe;
    };

  public:
    RendererMarkovChain();

    virtual void initialize(const merian::ContextHandle& context,
                            const merian::ResourceAllocatorHandle& allocator) override;

    ~RendererMarkovChain();

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
    merian::EntryPointHandle volume_shader;
    merian::EntryPointHandle volume_forward_project_shader;

    merian::VkBufferInHandle con_vtx = merian::VkBufferIn::compute_read("vtx");
    merian::VkBufferInHandle con_prev_vtx = merian::VkBufferIn::compute_read("prev_vtx");
    merian::VkBufferInHandle con_idx = merian::VkBufferIn::compute_read("idx");
    merian::VkBufferInHandle con_ext = merian::VkBufferIn::compute_read("ext");
    merian::GBufferInHandle con_gbuffer = merian::GBufferIn::compute_read("gbuffer");
    merian::VkBufferInHandle con_hits = merian::VkBufferIn::compute_read("hits");

    merian::VkSampledImageInHandle con_textures =
        merian::VkSampledImageIn::compute_read("textures");
    merian::VkTLASInHandle con_tlas = merian::VkTLASIn::compute_read("tlas");
    merian::VkSampledImageInHandle con_prev_volume_depth =
        merian::VkSampledImageIn::compute_read("prev_volume_depth", 1);
    merian::VkImageInHandle con_mv = merian::VkImageIn::transfer_src("mv");

    merian::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");

    merian::ManagedVkImageOutHandle con_irradiance;
    merian::ManagedVkImageOutHandle con_debug;
    merian::ManagedVkImageOutHandle con_volume;
    merian::ManagedVkImageOutHandle con_volume_depth;
    merian::ManagedVkImageOutHandle con_volume_mv;

    merian::ManagedVkBufferOutHandle con_markovchain;
    merian::ManagedVkBufferOutHandle con_lightcache;
    merian::ManagedVkBufferOutHandle con_volume_distancemc;

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

    bool reference_mode = false;

    int32_t spp = 1;
    int32_t volume_spp = 0;
    int32_t max_path_length = 3;
    VkBool32 use_light_cache_tail = VK_FALSE;

    VkBool32 volume_use_light_cache = 0;
    float volume_particle_size_um = 25.0;

    bool dump_mc = false;
    std::atomic_bool dumping = false;

    int32_t mc_samples = 5;
    float mc_samples_adaptive_prob = 0.7;
    int32_t distance_mc_samples = 3;

    VkBool32 mc_fast_recovery = VK_TRUE;

    int lc_grid_type = 0;
    uint32_t lc_buffer_size = 4000000;
    float lc_grid_steps_per_unit_size = 6.0;
    float lc_grid_tan_alpha_half = 0.002;
    float lc_grid_min_width = 0.01;
    float lc_grid_power = 2.0;

    int mc_adaptive_grid_type = 0;
    uint32_t mc_adaptive_buffer_size = 32777259;
    float mc_adaptive_grid_tan_alpha_half = 0.003;
    float mc_adaptive_grid_min_width = .01;
    float mc_adaptive_grid_power = 4.;
    float mc_adaptive_grid_steps_per_unit_size = 6.0;

    uint32_t mc_static_buffer_size = 800009;
    float mc_static_grid_width = 25.3;

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
