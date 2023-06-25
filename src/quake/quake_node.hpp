#pragma once

#include "glm/ext/vector_float4.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <queue>

extern "C" {
#include "quakedef.h"
}

class QuakeNode : public merian::Node {
  private:
    // in gl_texmgr.c
    static constexpr uint32_t MAX_GLTEXTURES = 4096;

    struct QuakeTexture {
        explicit QuakeTexture(gltexture_t* glt, uint32_t* data)
            : width(glt->width), height(glt->height) {
            cpu_tex.resize(width * height);
            memcpy(cpu_tex.data(), data, sizeof(uint32_t) * cpu_tex.size());
        }

        QuakeTexture(const QuakeTexture& other) = delete;

        QuakeTexture(const QuakeTexture&& other)
            : width(other.width), height(other.height), cpu_tex(std::move(other.cpu_tex)),
              gpu_tex(other.gpu_tex) {}

        uint32_t width;
        uint32_t height;

        std::vector<uint32_t> cpu_tex{};

        // allocated and uploaded in cmd_process
        merian::TextureHandle gpu_tex{};
    };

    struct PushConstant {
        int frame;

        glm::vec4 cam_x; // pos
        glm::vec4 cam_w; // forward
        glm::vec4 cam_u; // up

        glm::vec4 fog; // xyz: color, w: density

        int torch;
        int water; // player is underwater -> 1 means apply water effect

        // The texnums for sky_rt, sky_bk, sky_lf, sky_ft, sky_up, sky_dn;
        std::array<uint32_t, 6> sky;

        float cl_time; // quake time
        int ref;       // use reference sampling

        int health;
        int armor;
    };

    struct VertexExtraData {
        // glossmap texnum if available else 0
        uint16_t gloss;
        // normal/displacemnt texnum if available else 0
        uint16_t norm;

        uint16_t _2;
        uint16_t _3;

        uint16_t _4;
        uint16_t _5;

        uint16_t _6;
        uint16_t _7;

        uint16_t _8;
        uint16_t _9;

        uint16_t _10;
        uint16_t _11;

        uint16_t texnum;
        // 12 bit fullbright_texnum or 0 if not bright, 4 bit flags (most significant)
        // Flags:
        // 0 -> None
        // 1 -> Lava
        // 2 -> Slime
        // 3 -> Tele
        // 4 -> Water
        // 5 -> Water, lower mark
        // 6 -> Sky (unused currently)
        // 7 -> Waterfall
        uint16_t texnum_fb_flags;
    };

  public:
    /* Path to the Quake basedir. This directory must contain the id1 directory. */
    QuakeNode(const merian::ResourceAllocatorHandle& allocator,
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

  private:
    const merian::ResourceAllocatorHandle allocator;

    uint32_t width = 1920;
    uint32_t height = 1080;

    PushConstant pc;

    // Store some textures for custom patches
    // TODO: Frame-awareness (like for textures...)
    uint32_t texnum_none;
    uint32_t texnum_blood;
    uint32_t texnum_explosion;
    std::array<uint32_t, 6> texnum_skybox;

    // texnum -> texture
    std::unordered_map<uint32_t, QuakeTexture> textures;
    std::queue<uint32_t> pending_uploads;

    std::queue<std::string> pending_commands;

    bool worldspawn = false;

    double old_time = 0;
    bool pause = false;

    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;

    uint64_t frame = 0;
};
