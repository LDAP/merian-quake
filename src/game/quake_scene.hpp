#pragma once

#include "game/quake_material.hpp"

#include "merian-shaders/scene/scene.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/utils/concurrent/concurrent_queue.hpp"
#include "merian/utils/input_controller.hpp"
#include "merian/utils/input_controller_dummy.hpp"
#include "merian/utils/input_listener.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <atomic>
#include <queue>
#include <string>
#include <thread>

extern "C" {
#include "quakedef.h"
}

namespace merian_quake {

// QuakeScene owns Quake's global lifecycle: the QuakeSpasm init, the game
// thread, the input listener that pumps key/mouse events, the per-frame
// render info (camera/sun/fog/sky/uniform), and the texture upload pump.
//
// Quake itself is built on a stack of static globals; only one instance can
// exist at a time. The owning QuakeNode constructs the scene exactly once
// (lazily, after the FrameCachingShaderObjectAllocator has the right
// in-flight count) and tears it down with the node.
class QuakeScene : public merian::Scene {
  public:
    QuakeScene(const merian::ShaderCompileContextHandle& compile_context,
               const merian::ContextHandle& context,
               const merian::ResourceAllocatorHandle& allocator,
               const merian::ShaderObjectAllocatorHandle& obj_allocator,
               const merian::MaterialSystemHandle& material_system,
               uint32_t quakespasm_argc,
               const char** quakespasm_argv);

    ~QuakeScene() override;

    merian::float3 get_up() override {
        return merian::float3(0, 0, 1);
    }

    // Expose Quake's `cl.time` to the shader-side scene clock.
    float get_time(float time) override;

    merian::MaterialModelID get_quake_material_type_id() const {
        return quake_material_type_id;
    }

    // Resolution as reported by Quake (vid.width / vid.height). Updated when
    // VID_Changed_f fires.
    vk::Extent3D get_resolution() const {
        return resolution;
    }

    // Swap the input controller. Safe to call before or after game-thread
    // startup; the listener is rebound to the new controller. Pass a dummy
    // controller to detach.
    void set_controller(const merian::InputControllerHandle& controller);

    // Queue a Quake console command to be executed on the game thread.
    void queue_command(const std::string& command);

    // ------------------------------------------------------------------
    // Quake callback entry points (invoked from the QuakeSpasm thread via
    // the extern "C" thunks in quake_scene.cpp).

    void cb_VID_Changed();
    void cb_QS_texture_load(gltexture_t* glt, const uint32_t* data);
    void cb_IN_Move(usercmd_t* cmd);
    void cb_R_RenderScene();
    void cb_QS_worldspawn();

    // Called by QuakeNode::properties() to surface Quake-level settings.
    void properties(merian::Properties& config);

  protected:
    void
    on_update(const merian::CommandBufferHandle& cmd, float time, float time_diff, uint32_t frame)
        override;

  private:
    void register_input_listener(const merian::InputControllerHandle& controller);
    void drain_pending_uploads(const merian::CommandBufferHandle& cmd);
    void refresh_render_info(bool render_this_frame);
    void run_startup_commands_if_needed();
    void rebuild_static_world();
    void init_dynamic_meshes();
    void refresh_dynamic_meshes();
    void cycle_animated_materials();

  private:
    merian::MaterialModelID quake_material_type_id{};

    // Resolution: written from the Quake thread (cb_VID_Changed); read on the
    // graph thread. Quake calls this before any update; just keep it atomic-
    // ish (single uint32 reads on x86 are torn-free; we accept it).
    vk::Extent3D resolution{};

    // Game thread / synchronization.
    std::thread game_thread;
    std::atomic_bool game_running{true};
    merian::ConcurrentQueue<bool> sync_gamestate;
    merian::ConcurrentQueue<float> sync_render;

    // Set true by the game thread when QuakeSpasm hits R_RenderScene; reset
    // to false at the top of every on_update. Also gated against
    // scr_drawloading.
    bool render_this_frame = false;
    // Set on worldspawn / when sun overrides change. Renderers will read
    // this off Scene's get_constant_data_dirty() once that lands.
    bool constant_data_dirty = true;
    bool update_gamestate = true;
    uint64_t frame_counter = 0;
    uint64_t last_worldspawn_frame = 0;
    double server_fps = 0;

    // Sun + volume scattering state, surfaced from worldspawn parsing /
    // overrides. The renderer reads these via the camera + scene stash;
    // QuakeNode also exposes them through properties.
    merian::float3 sun_color{};
    merian::float3 sun_direction{0, 0, 1};
    float volume_max_t = 1000.F;

    // Static brush world: built once on the first worldspawn after
    // scene construction. Subsequent worldspawns are ignored for now
    // (Scene has no remove_mesh; map-change support is a follow-up).
    bool world_meshes_built = false;
    merian::NodeID world_node_id = merian::NODE_ID_INVALID;

    // Animated brush materials: (material_id, base_texture, fb/normal/gloss
    // texnums, surface_flags, alpha_mode). Per-frame, R_TextureAnimation
    // resolves the current member of the cycle for each base; if its texnum
    // differs from what the material currently holds, we re-pack and call
    // MaterialSystem::update_material so the geometry doesn't have to rebake.
    struct AnimatedBrushMaterial {
        merian::MaterialID material_id;
        texture_t* base_tex;
        merian::TextureID fb_texnum;
        merian::TextureID normal_texnum;
        merian::TextureID gloss_texnum;
        uint16_t surface_flags;
        uint8_t alpha_mode;
        merian::TextureID current_base_texnum;
    };
    std::vector<AnimatedBrushMaterial> animated_brush_materials;

    // Dynamic geometry: per-frame extraction of alias / brush-entity / sprite
    // and particle quake objects. The meshes are pre-allocated once (in
    // init_dynamic_meshes) so per-frame refills only mark mesh data dirty
    // — no add_mesh churn that would force the static world to rebuild.
    // For now everything (per type) shares a single material with no
    // texture bound; per-entity textures are a follow-up once material
    // updates are wired in.
    bool dynamic_meshes_built = false;
    merian::NodeID dynamic_node_id = merian::NODE_ID_INVALID;
    merian::MeshID entity_mesh_id = 0;
    merian::MeshID sprite_mesh_id = 0;
    merian::MeshID particle_mesh_id = 0;
    merian::MaterialID entity_material_id = 0;
    merian::MaterialID sprite_material_id = 0;
    merian::MaterialID particle_material_id = 0;
    double prev_cl_time = 0.0;

    // Input.
    merian::InputControllerHandle controller =
        std::make_shared<merian::DummyInputController>();
    std::shared_ptr<merian::InputListener> input_listener;
    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;
    bool raw_mouse_was_enabled = false;

    // Texture upload queue, drained per-frame inside on_update.
    struct PendingTexture {
        uint32_t texnum;
        uint32_t width;
        uint32_t height;
        uint32_t flags;
        bool linear;
        std::string name;
        std::vector<uint32_t> rgba;
    };
    std::vector<PendingTexture> pending_uploads;
    std::mutex pending_uploads_mutex;

    // Console commands queued from the graph/UI thread; executed on game
    // thread.
    std::queue<std::string> pending_commands;
    std::mutex pending_commands_mutex;

    // Properties / debug knobs (unchanged from old QuakeNode).
    int default_filtering = 0;
    std::string startup_commands{};
    bool startup_commands_dispatched = false;
    int stop_after_worldspawn = -1;
    bool rebuild_after_stop = true;
    bool overwrite_sun = false;
    merian::float3 overwrite_sun_dir{0, 0, 1};
    merian::float3 overwrite_sun_col{0};
    bool mu_t_s_overwrite = false;
    float mu_t = 0.0F;
    merian::float3 mu_s_div_mu_t{1};
    int playermodel = 1;
    bool reproducible_renders = false;

    // HACK texture ids stored once at load (used by future particle mesh).
    uint32_t texnum_blood = 0;
    uint32_t texnum_explosion = 0;

    merian::CameraID quake_camera;
};

using QuakeSceneHandle = std::shared_ptr<QuakeScene>;

} // namespace merian_quake
