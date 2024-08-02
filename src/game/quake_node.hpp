#pragma once

#include "game/quake_helpers.hpp"
#include "glm/ext/vector_float4.hpp"

#include "merian-nodes/connectors/ptr_out.hpp"
#include "merian-nodes/connectors/special_static_out.hpp"
#include "merian-nodes/connectors/vk_buffer_array_out.hpp"
#include "merian-nodes/connectors/vk_texture_array_out.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/nodes/as_builder/device_as_builder.hpp"

#include "config.h"
#include "merian/utils/input_controller.hpp"
#include "merian/utils/input_controller_dummy.hpp"
#include "merian/utils/string.hpp"

#include <queue>
#include <set>

extern "C" {
#include "quakedef.h"
}

class QuakeNode : public merian_nodes::Node {
  public:
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

    struct UniformData {
        glm::vec4 cam_x_mu_t; // pos, and fog mu_t in alpha
        glm::vec4 cam_w;      // forward, and time_diff in alpha (set to 1. if 0.)
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

    struct ConstantData {
        glm::vec3 sun_color;
        glm::vec3 sun_direction;

        float fov;
        float fov_tan_alpha_half;
        float volume_max_t = 1000;
    };

    struct QuakeRenderInfo {
        // Can be used as push constant.
        // Updated every frame and only valid if render == true
        UniformData uniform;

        // Does only change if a new world is loaded or settings are changed
        ConstantData constant;

        // If this is false do not render, just clear your outputs.
        bool render;
        // Set if new constant data is available. For example, if a new map was loaded, maybe reset
        // stuff?
        bool constant_data_update = true;
    };

    struct QuakeTexture {
        explicit QuakeTexture(gltexture_t* glt, uint32_t* data)
            : width(glt->width), height(glt->height), flags(glt->flags), name(glt->name) {
            cpu_tex.resize(width * height);

            memcpy(cpu_tex.data(), data, sizeof(uint32_t) * cpu_tex.size());

            linear = false;
            linear |= merian::ends_with(glt->name, "_norm");
            linear |= merian::ends_with(glt->name, "_gloss");
        }

        const uint32_t width;
        const uint32_t height;
        // bitmask of TEXPREF_* flags in gl_texmgr
        const uint32_t flags;
        // if true interpret linearly (Unorm) else as Srgb.
        bool linear;

        std::vector<uint32_t> cpu_tex{};

        const std::string name;
    };

    struct RTGeometry {
        merian::BufferHandle vtx;
        merian::BufferHandle prev_vtx;
        merian::BufferHandle idx;
        merian::BufferHandle ext;

        std::shared_ptr<merian_nodes::DeviceASBuilder::BlasBuildInfo> blas_info;
        merian_nodes::DeviceASBuilder::BlasBuildInfo::GeometryHandle geo_handle;
        vk::GeometryInstanceFlagsKHR instance_flags;
    };

  public:
    QuakeNode(const merian::ContextHandle& context,
              const merian::ResourceAllocatorHandle allocator,
              const int quakespasm_argc,
              const char** quakespasm_argv);

    ~QuakeNode();

    void set_controller(const merian::InputControllerHandle& controller);

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout);

    NodeStatusFlags
    on_connected([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
                 [[maybe_unused]] const merian::DescriptorSetLayoutHandle& descriptor_set_layout);

    NodeStatusFlags pre_process(merian_nodes::GraphRun& run, const merian_nodes::NodeIO& io);

    void process(merian_nodes::GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian_nodes::NodeIO& io);

    NodeStatusFlags properties(merian::Properties& config);

    // -----------------------------------------------------

    // Called when the vid struct changed
    void VID_Changed_f(cvar_t* var);

    // called when a texture should be loaded
    void QS_texture_load(gltexture_t* glt, uint32_t* data);

    // called from within qs
    void IN_Move(usercmd_t* cmd);

    // called when Quake wants to render
    void R_RenderScene();

    // called each time a new is (re)loaded
    void QS_worldspawn();

    // -----------------------------------------------------

    void queue_command(std::string command) {
        pending_commands.push(command);
    }

  private:
    // processes the pending uploads and updates the current descriptor set
    void update_textures(const vk::CommandBuffer& cmd, const merian_nodes::NodeIO& io);

    void update_static_geo(const vk::CommandBuffer& cmd);
    void update_dynamic_geo(const vk::CommandBuffer& cmd, const merian::ProfilerHandle& profiler);
    void update_as(const vk::CommandBuffer& cmd, const merian_nodes::NodeIO& io);

  private:
    const merian::ContextHandle context;
    const merian::ResourceAllocatorHandle allocator;

    // Graph outputs
    // clang-format off
    merian_nodes::SpecialStaticOutHandle<vk::Extent3D> con_resolution;
    merian_nodes::PtrOutHandle<QuakeRenderInfo> con_render_info = merian_nodes::PtrOut<QuakeRenderInfo>::create("render_info");
    merian_nodes::VkTextureArrayOutHandle con_textures = merian_nodes::VkTextureArrayOut::create("textures", MAX_GLTEXTURES);

    merian_nodes::VkBufferArrayOutHandle con_vtx = merian_nodes::VkBufferArrayOut::create("vtx", MAX_GEOMETRIES);
    merian_nodes::VkBufferArrayOutHandle con_prev_vtx = merian_nodes::VkBufferArrayOut::create("prev_vtx", MAX_GEOMETRIES);
    merian_nodes::VkBufferArrayOutHandle con_idx = merian_nodes::VkBufferArrayOut::create("idx", MAX_GEOMETRIES);
    merian_nodes::VkBufferArrayOutHandle con_ext = merian_nodes::VkBufferArrayOut::create("ext", MAX_GEOMETRIES);

    merian_nodes::PtrOutHandle<merian_nodes::DeviceASBuilder::TlasBuildInfo> con_tlas_info = merian_nodes::PtrOut<merian_nodes::DeviceASBuilder::TlasBuildInfo>::create("tlas_info");
    // clang-format on

    // Game thread / synchronization
    std::thread game_thread;
    std::atomic_bool game_running = true;
    merian::ConcurrentQueue<bool> sync_gamestate;
    merian::ConcurrentQueue<bool> sync_render;

    // Game state
    double old_time = 0;
    float force_timediff = 0;
    bool update_gamestate = true;
    QuakeRenderInfo render_info;
    double server_fps = 0;
    uint64_t frame = 0;
    uint64_t last_worldspawn_frame = 0;

    // Input processing
    std::shared_ptr<merian::InputController> controller =
        std::make_shared<merian::DummyInputController>();
    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;
    bool raw_mouse_was_enabled = false;

    // Quake commands
    std::string startup_commands = {0};
    std::queue<std::string> pending_commands;

    // Helpers for image generation
    int stop_after_worldspawn = -1;
    bool rebuild_after_stop = true;

    // Textures
    std::map<uint32_t, QuakeTexture> pending_uploads;

    // Geometry
    std::vector<RTGeometry> static_geo;
    std::vector<RTGeometry> dynamic_geo;

    // keep on hand to prevent realloc and copy...
    std::vector<float> vtx;
    std::vector<float> prev_vtx;
    std::vector<uint32_t> idx;
    std::vector<VertexExtraData> ext;

    // Store some textures for custom patches
    uint32_t texnum_blood = 0;
    uint32_t texnum_explosion = 0;

    int default_filtering = 0;

    // Debug overwrites
    bool overwrite_sun = false;
    glm::vec3 overwrite_sun_dir{0, 0, 1};
    glm::vec3 overwrite_sun_col{0};
    // --
    bool mu_t_s_overwrite = false;
    float mu_t = 0.;
    glm::vec3 mu_s_div_mu_t = glm::vec3(1);
    // --
    // 0 None, 1 Gun, 2 Full
    int playermodel = 1;
    bool reproducible_renders = false;
};
