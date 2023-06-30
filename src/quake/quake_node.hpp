#pragma once

#include "glm/ext/vector_float4.hpp"
#include "merian/utils/input_controller.hpp"
#include "merian/utils/sdl_audio_device.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/raytrace/blas_builder.hpp"
#include "merian/vk/raytrace/tlas_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <queue>
#include <unordered_set>

extern "C" {
#include "quakedef.h"
}

class QuakeNode : public merian::Node {
  public:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct QuakeTexture {
        explicit QuakeTexture(gltexture_t* glt, uint32_t* data)
            : width(glt->width), height(glt->height), flags(glt->flags) {
            cpu_tex.resize(width * height);

            memcpy(cpu_tex.data(), data, sizeof(uint32_t) * cpu_tex.size());
        }

        uint32_t width;
        uint32_t height;
        // bitmask of TEXPREF_* flags in gl_texmgr
        uint32_t flags;

        std::vector<uint32_t> cpu_tex{};

        // allocated and uploaded in cmd_process
        merian::TextureHandle gpu_tex{};
    };

    struct PushConstant {
        glm::vec4 cam_x; // pos
        glm::vec4 cam_w; // forward
        glm::vec4 cam_u; // up

        glm::vec4 fog; // xyz: color, w: density

        int torch;
        int water; // player is underwater -> 1 means apply water effect

        // The texnums for sky_rt, sky_bk, sky_lf, sky_ft, sky_up, sky_dn;
        std::array<uint32_t, 6> sky;

        float cl_time; // quake time
        int ref{false};       // use reference sampling

        int health;
        int armor;

        int frame;
    };

    struct VertexExtraData {
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

        // texnum and alpha in upper 4 bits
        // Alpha meaning: 0: use texture, [1,15] map to [0,1] where 15 is fully opaque and 1 transparent
        uint16_t texnum_alpha{};
        // 12 bit fullbright_texnum or 0 if not bright, 4 bit flags (most significant)
        // for flags see MAT_FLAGS_* in config.h
        uint16_t texnum_fb_flags{};
    };

  public:
    /* Path to the Quake basedir. This directory must contain the id1 directory. */
    QuakeNode(const merian::SharedContext& context,
              const merian::ResourceAllocatorHandle& allocator,
              const std::shared_ptr<merian::InputController> controller,
              const char* base_dir = "./res/quake");

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

  private:
    void update_static_geo(const vk::CommandBuffer& cmd);
    void update_dynamic_geo(const vk::CommandBuffer& cmd);
    void update_as(const vk::CommandBuffer& cmd, const merian::ProfilerHandle profiler);
    // processes the pending uploads and updates the current descriptor set
    void update_textures(const vk::CommandBuffer& cmd);

  private:
    const merian::SharedContext context;
    const merian::ResourceAllocatorHandle allocator;

    std::unique_ptr<merian::SDLAudioDevice> audio_device;

    merian::BLASBuilder blas_builder;
    merian::TLASBuilder tlas_builder;

    merian::ShaderModuleHandle shader;
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
    merian::DescriptorSetHandle quake_sets;

    merian::PipelineHandle pipe;

    // ----------------------------------------------------
    // Params

    uint32_t width = 1920;
    uint32_t height = 1080;

    PushConstant pc;

    // ----------------------------------------------------
    // Per-frame info, TODO: what if multiple in flight?

    // Store some textures for custom patches
    uint32_t texnum_blood;
    uint32_t texnum_explosion;
    std::array<uint32_t, 6> texnum_skybox;

    // Textures
    // texnum -> texture
    std::unordered_map<uint32_t, std::shared_ptr<QuakeTexture>> textures;
    std::unordered_set<uint32_t> pending_uploads;

    // Static geo
    std::vector<float> static_vtx;
    std::vector<uint32_t> static_idx;
    std::vector<VertexExtraData> static_ext; // per primitive
    // Can be nullptr if idx is empty
    merian::BufferHandle static_vtx_buffer;
    merian::BufferHandle static_idx_buffer;
    merian::BufferHandle static_ext_buffer;
    // Can be nullptr if idx is empty
    merian::AccelerationStructureHandle static_blas;

    // Dynamic geo
    std::vector<float> dynamic_vtx;
    std::vector<uint32_t> dynamic_idx;
    std::vector<VertexExtraData> dynamic_ext;
    // Can be nullptr if idx is empty
    merian::BufferHandle dynamic_vtx_buffer;
    merian::BufferHandle dynamic_idx_buffer;
    merian::BufferHandle dynamic_ext_buffer;
    // Can be nullptr if idx is empty
    merian::AccelerationStructureHandle dynamic_blas;

    merian::BufferHandle instances_buffer;
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    // Can be nullptr if there is not geometry
    merian::AccelerationStructureHandle tlas;

    // ----------------------------------------------------

    std::queue<std::string> pending_commands;
    bool worldspawn = false;

    double old_time = 0;
    bool pause = false;
    bool sound = false;

    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;

    uint64_t frame = 0;
};
