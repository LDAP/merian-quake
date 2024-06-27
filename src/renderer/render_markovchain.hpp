#pragma once

#include "merian-nodes/connectors/any_in.hpp"
#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"
#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_buffer_array_in.hpp"
#include "merian-nodes/connectors/vk_texture_array_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "merian-nodes/graph/node.hpp"

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
    RendererMarkovChain(const merian::SharedContext& context,
                        const merian::ResourceAllocatorHandle& allocator);

    ~RendererMarkovChain();

    // -----------------------------------------------------

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs(const merian_nodes::ConnectorIOMap& output_for_input) override;

    NodeStatusFlags
    on_connected(const merian::DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(merian_nodes::GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian_nodes::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
    const merian::SharedContext context;
    const merian::ResourceAllocatorHandle allocator;

    merian::ShaderModuleHandle rt_shader;
    merian::ShaderModuleHandle clear_shader;
    merian::ShaderModuleHandle volume_shader;
    merian::ShaderModuleHandle volume_forward_project_shader;

    merian_nodes::ManagedVkImageInHandle con_blue_noise =
        merian_nodes::ManagedVkImageIn::compute_read("blue_noise", 0);
    merian_nodes::ManagedVkImageInHandle con_prev_filtered =
        merian_nodes::ManagedVkImageIn::compute_read("prev_filtered", 1);
    merian_nodes::ManagedVkImageInHandle con_prev_volume_depth =
        merian_nodes::ManagedVkImageIn::compute_read("prev_volume_depth", 1);
    merian_nodes::ManagedVkBufferInHandle con_prev_gbuf =
        merian_nodes::ManagedVkBufferIn::compute_read("prev_gbuf", 1);
    merian_nodes::AnyInHandle con_render_info = merian_nodes::AnyIn::create("render_info");
    merian_nodes::VkTextureArrayInHandle con_textures =
        merian_nodes::VkTextureArrayIn::compute_read("textures");
    merian_nodes::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian_nodes::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian_nodes::VkBufferArrayInHandle con_vtx =
        merian_nodes::VkBufferArrayIn::compute_read("vtx");
    merian_nodes::VkBufferArrayInHandle con_prev_vtx =
        merian_nodes::VkBufferArrayIn::compute_read("prev_vtx");
    merian_nodes::VkBufferArrayInHandle con_idx =
        merian_nodes::VkBufferArrayIn::compute_read("idx");
    merian_nodes::VkBufferArrayInHandle con_ext =
        merian_nodes::VkBufferArrayIn::compute_read("ext");
    merian_nodes::VkTLASInHandle con_tlas = merian_nodes::VkTLASIn::compute_read("tlas");

    merian_nodes::ManagedVkImageOutHandle con_irradiance;
    merian_nodes::ManagedVkImageOutHandle con_albedo;
    merian_nodes::ManagedVkImageOutHandle con_mv;
    merian_nodes::ManagedVkImageOutHandle con_debug;
    merian_nodes::ManagedVkImageOutHandle con_moments;
    merian_nodes::ManagedVkImageOutHandle con_volume;
    merian_nodes::ManagedVkImageOutHandle con_volume_moments;
    merian_nodes::ManagedVkImageOutHandle con_volume_depth;
    merian_nodes::ManagedVkImageOutHandle con_volume_mv;

    merian_nodes::ManagedVkBufferOutHandle con_markovchain;
    merian_nodes::ManagedVkBufferOutHandle con_lightcache;
    merian_nodes::ManagedVkBufferOutHandle con_gbuffer;
    merian_nodes::ManagedVkBufferOutHandle con_volume_distancemc;

    //-----------------------------------------------------

    merian::DescriptorSetLayoutHandle graph_desc_set_layout;
    merian::PipelineLayoutHandle pipe_layout;

    merian::PipelineHandle pipe;
    merian::PipelineHandle clear_pipe;
    merian::PipelineHandle volume_pipe;
    merian::PipelineHandle volume_forward_project_pipe;

    // ----------------------------------------------------

    uint64_t frame = 0;
    double prev_cl_time = 0;

    // ----------------------------------------------------

    // Spec constants
    // https://gpuopen.com/learn/rdna-performance-guide/ recommends 8x4
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;
    int32_t spp = 1;
    int32_t volume_spp = 0;
    int32_t max_path_length = 3;
    int32_t use_light_cache_tail = 0;
    int32_t adaptive_sampling = 0;

    int32_t volume_use_light_cache = 0;
    float volume_particle_size_um = 25.0;
    float volume_max_t = 1000.;

    bool dump_mc = false;

    int32_t mc_samples = 5;
    float mc_samples_adaptive_prob = 0.7;
    int32_t distance_mc_samples = 3;

    int32_t mc_fast_recovery = 1;

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
};
