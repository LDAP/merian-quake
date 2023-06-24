#pragma once

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

  public:
    /* Path to the Quake basedir. This directory must contain the id1 directory. */
    QuakeNode(const merian::ResourceAllocatorHandle& allocator,
              const char* base_dir = "./res/quake");

    ~QuakeNode();

    virtual std::string name() {
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

    void cmd_build(const vk::CommandBuffer& cmd,
                   const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                   const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                   const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                   const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs);

    void cmd_process(const vk::CommandBuffer& cmd,
                     merian::GraphRun& run,
                     const uint32_t set_index,
                     const std::vector<merian::ImageHandle>& image_inputs,
                     const std::vector<merian::BufferHandle>& buffer_inputs,
                     const std::vector<merian::ImageHandle>& image_outputs,
                     const std::vector<merian::BufferHandle>& buffer_outputs);

  private:
    const merian::ResourceAllocatorHandle allocator;

    // Store some textures for custom patches
    uint32_t texnum_none;
    uint32_t texnum_blood;
    uint32_t texnum_explosion;
    std::array<uint32_t, 6> texnum_skybox;

    // texnum -> texture
    std::unordered_map<uint32_t, QuakeTexture> textures;
    std::queue<uint32_t> pending_uploads;

    bool worldspawn = false;

    double old_time = 0;
    bool pause = false;

    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;

    uint64_t frame = 0;
};
