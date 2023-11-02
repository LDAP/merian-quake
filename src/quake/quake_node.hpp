#pragma once

#include "glm/ext/vector_float4.hpp"
#include "merian/utils/input_controller.hpp"
#include "merian/utils/sdl_audio_device.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/raytrace/blas_builder.hpp"
#include "merian/vk/raytrace/tlas_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "merian/vk/sync/ring_fences.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "quake/config.h"

#include <optional>
#include <queue>
#include <unordered_set>

extern "C" {
#include "quakedef.h"
}

class QuakeNode : public merian::Node {
  public:
    struct QuakeTexture {
        explicit QuakeTexture(gltexture_t* glt, uint32_t* data)
            : width(glt->width), height(glt->height), flags(glt->flags) {
            cpu_tex.resize(width * height);

            memcpy(cpu_tex.data(), data, sizeof(uint32_t) * cpu_tex.size());

            linear = false;
            linear |= merian::ends_with(glt->name, "_norm");
            linear |= merian::ends_with(glt->name, "_gloss");
        }

        uint32_t width;
        uint32_t height;
        // bitmask of TEXPREF_* flags in gl_texmgr
        uint32_t flags;
        // if true interpret linearly (Unorm) else as Srgb.
        bool linear;

        std::vector<uint32_t> cpu_tex{};

        // allocated and uploaded in cmd_process
        merian::TextureHandle gpu_tex{};
    };

    struct PlayerData {
        // see PLAYER_* in config.h
        unsigned char flags;
        unsigned char padding0;
        unsigned char padding1;
        unsigned char padding2;
    };

    struct RTConfig {
        unsigned char flags = 0;
        unsigned char padding0;
        unsigned char padding1;
        unsigned char padding2;
    };

    struct PushConstant {
        glm::vec4 cam_x_mu_t; // pos, and fog mu_t
        glm::vec4 cam_w;      // forward
        glm::vec4 cam_u;      // up

        glm::vec4 prev_cam_x_mu_sx;
        glm::vec4 prev_cam_w_mu_sy;
        glm::vec4 prev_cam_u_mu_sz;

        // The texnums for sky_rt, sky_bk, sky_lf, sky_ft, sky_up, sky_dn;
        std::array<uint16_t, 6> sky;

        // quake time
        float cl_time;
        uint32_t frame;

        PlayerData player;
        RTConfig rt_config;
    };

    struct VertexExtraData {
        // texnum and alpha in upper 4 bits
        // Alpha meaning: 0: use texture, [1,15] map to [0,1] where 15 is fully opaque and 1
        // transparent
        uint16_t texnum_alpha{};
        // 12 bit fullbright_texnum or 0 if not bright, 4 bit flags (most significant)
        // for flags see MAT_FLAGS_* in config.h
        uint16_t texnum_fb_flags{};

        // Normals encoded using encode_normal
        // or glossmap texnum and normalmap texnum
        // if n1_brush = ~0.
        uint32_t n0_gloss_norm{};
        // Marks as brush model if ~0, else second normal
        uint32_t n1_brush{};
        uint32_t n2{};

        // Texture coords, encoded using float_to_half
        uint16_t s_0{};
        uint16_t t_0{};
        uint16_t s_1{};
        uint16_t t_1{};
        uint16_t s_2{};
        uint16_t t_2{};
    };

    struct RTGeometry {
        merian::BufferHandle vtx_buffer{nullptr};
        merian::BufferHandle prev_vtx_buffer{nullptr};
        merian::BufferHandle idx_buffer{nullptr};
        merian::BufferHandle ext_buffer{nullptr};
        merian::AccelerationStructureHandle blas{nullptr};
        vk::BuildAccelerationStructureFlagsKHR blas_flags{};
        vk::GeometryInstanceFlagsKHR instance_flags{};

        // vtx.size() / 3
        uint32_t vtx_count = 0;
        // idx.size() / 3
        uint32_t primitive_count = 0;

        uint64_t last_rebuild = 0;
    };

  public:
    /* Path to the Quake basedir. This directory must contain the id1 directory.
     * It must be guaranteed that at most `ring_size` resources suffice. (A RingFence with that
     * RING_SIZE).
     */
    QuakeNode(const merian::SharedContext& context,
              const merian::ResourceAllocatorHandle& allocator,
              const std::shared_ptr<merian::InputController> controller,
              const uint32_t ring_size,
              const int quakespasm_argc = 0,
              const char** quakespasm_argv = nullptr);

    ~QuakeNode();

    virtual std::string name() override {
        return "Quake";
    }

    // Callbacks from quake -------------------------------

    // called each time a new map is (re)loaded
    void QS_worldspawn();

    // called when a texture should be loaded
    void QS_texture_load(gltexture_t* glt, uint32_t* data);

    // called from within qs
    void IN_Move(usercmd_t* cmd);

    // -----------------------------------------------------

    std::tuple<std::vector<merian::NodeInputDescriptorImage>,
               std::vector<merian::NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
               std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<merian::NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<merian::NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    void cmd_build(const vk::CommandBuffer& cmd,
                   const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                   const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                   const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                   const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs) override;

    void cmd_process(const vk::CommandBuffer& cmd,
                     merian::GraphRun& run,
                     const uint32_t set_index,
                     const std::vector<merian::ImageHandle>& image_inputs,
                     const std::vector<merian::BufferHandle>& buffer_inputs,
                     const std::vector<merian::ImageHandle>& image_outputs,
                     const std::vector<merian::BufferHandle>& buffer_outputs) override;

    void queue_command(std::string command) {
        pending_commands.push(command);
    }

    void get_configuration(merian::Configuration& config, bool& needs_rebuild) override;

  private:
    // Attemps to reuse the supplied old_geo (by update, buffer reuse).
    // Uses "flags" if building, else old_geo.flags.
    // If force_rebuild is false and an update is possible an update is queued instead if a rebuild.
    QuakeNode::RTGeometry get_rt_geometry(const vk::CommandBuffer& cmd,
                                          const std::vector<float>& vtx,
                                          const std::vector<float>& prev_vtx,
                                          const std::vector<uint32_t>& idx,
                                          const std::vector<QuakeNode::VertexExtraData>& ext,
                                          const std::unique_ptr<merian::BLASBuilder>& blas_builder,
                                          const QuakeNode::RTGeometry old_geometry,
                                          const bool force_rebuild,
                                          const vk::BuildAccelerationStructureFlagsKHR flags);
    // Optionally refreshes the geo and updates the current descriptor set if necessary
    void update_static_geo(const vk::CommandBuffer& cmd, const bool refresh_geo);
    // Refreshes the geo and updates the current descriptor set
    void update_dynamic_geo(const vk::CommandBuffer& cmd);
    // Builds the as and updates the current descriptor set
    void update_as(const vk::CommandBuffer& cmd, const merian::ProfilerHandle profiler);
    // processes the pending uploads and updates the current descriptor set
    void update_textures(const vk::CommandBuffer& cmd);

    void parse_worldspawn();

  private:
    const merian::SharedContext context;
    const merian::ResourceAllocatorHandle allocator;
    const std::shared_ptr<merian::InputController> controller;

    std::unique_ptr<merian::SDLAudioDevice> audio_device;

    merian::ShaderModuleHandle rt_shader;
    merian::ShaderModuleHandle clear_shader;
    merian::ShaderModuleHandle volume_shader;
    merian::ShaderModuleHandle volume_forward_project_shader;

    merian::DescriptorSetLayoutHandle graph_desc_set_layout;
    merian::DescriptorPoolHandle graph_pool;
    std::vector<merian::DescriptorSetHandle> graph_sets;
    std::vector<merian::TextureHandle> graph_textures;

    // Use this buffer in bindings when the real resource is not available
    merian::BufferHandle binding_dummy_buffer;
    merian::TextureHandle binding_dummy_image;

    //-----------------------------------------------------

    merian::DescriptorSetLayoutHandle quake_desc_set_layout;
    merian::DescriptorPoolHandle quake_pool;

    merian::PipelineHandle pipe;
    merian::PipelineHandle clear_pipe;
    merian::PipelineHandle volume_pipe;
    merian::PipelineHandle volume_forward_project_pipe;

    // ----------------------------------------------------
    // Params

    PushConstant pc;

    // Store some textures for custom patches
    uint32_t texnum_blood = 0;
    uint32_t texnum_explosion = 0;

    // ----------------------------------------------------
    // Per-frame data and updates

    struct FrameData {
        // Needed per frame because might reallocate scratch buffer
        std::unique_ptr<merian::BLASBuilder> blas_builder;
        std::unique_ptr<merian::TLASBuilder> tlas_builder;

        merian::DescriptorSetHandle quake_sets{nullptr};

        // keep copy of current_textures to keep them alive and to know which descriptors to update
        std::array<std::shared_ptr<QuakeTexture>, MAX_GLTEXTURES> textures{};

        std::vector<RTGeometry> static_geometries;
        std::vector<RTGeometry> dynamic_geometries;

        // TLAS
        merian::BufferHandle instances_buffer{nullptr};
        // Can be nullptr if there is not geometry
        merian::AccelerationStructureHandle tlas{nullptr};
        uint32_t last_instances_size{};
    };

    // Access using frame % frames.size()
    std::vector<FrameData> frames;
    uint64_t frame = 0;
    double prev_cl_time = 0;

    std::array<char, 512> startup_commands_buffer = {0};

    FrameData& current_frame() {
        assert(frames.size());
        return frames[frame % frames.size()];
    }

    // "Global" frame info
    // texnum -> texture
    std::array<std::shared_ptr<QuakeTexture>, MAX_GLTEXTURES> current_textures;
    std::unordered_set<uint32_t> pending_uploads;

    // Static geo
    std::vector<RTGeometry> current_static_geo;

    // Keep to avoid reallocation
    std::vector<float> dynamic_vtx;
    std::vector<float> dynamic_prev_vtx;
    std::vector<uint32_t> dynamic_idx;
    std::vector<QuakeNode::VertexExtraData> dynamic_ext;

    std::vector<float> static_vtx;
    std::vector<float> static_prev_vtx;
    std::vector<uint32_t> static_idx;
    std::vector<QuakeNode::VertexExtraData> static_ext;

    // ----------------------------------------------------
    // Gamestate

    std::queue<std::string> pending_commands;
    bool worldspawn = false;

    double old_time = 0;
    bool update_gamestate = true;
    bool sound = true;
    bool raw_mouse_was_enabled = false;
    int stop_after_worldspawn = -1;
    bool rebuild_after_stop = true;
    uint64_t last_worldspawn_frame = 0;
    float force_timediff = 0;

    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;

    int32_t render_width = 1920;
    int32_t render_height = 1080;

    // 0 None, 1 Gun, 2 Full
    int playermodel = 1;

    float fov = glm::radians(90.);
    float fov_tan_alpha_half = glm::tan(fov / 2);

    glm::vec3 quake_sun_dir{0};
    glm::vec3 quake_sun_col{0};

    bool overwrite_sun = false;
    glm::vec3 overwrite_sun_dir{0, 0, 1};
    glm::vec3 overwrite_sun_col{0};

    // Spec constants
    // https://gpuopen.com/learn/rdna-performance-guide/ recommends 8x4
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;
    int32_t spp = 1;
    int32_t volume_spp = 0;
    int32_t max_path_length = 3;
    int32_t use_light_cache_tail = 0;
    int32_t adaptive_sampling = 0;
    bool mu_t_s_overwrite = false;
    float mu_t = 0.;
    glm::vec3 mu_s_div_mu_t = glm::vec3(1);
    int32_t volume_use_light_cache = 0;
    float volume_particle_size_um = 25.0;
    float volume_max_t = 1000.;

    bool dump_mc = false;

    bool reproducible_renders = false;

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
    int32_t mc_static_vertex_state_count = 23;

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
