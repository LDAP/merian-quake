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
    void on_update(const merian::CommandBufferHandle& cmd,
                   float time,
                   float time_diff,
                   uint32_t frame) override;

  private:
    void register_input_listener(const merian::InputControllerHandle& controller);

    void rebuild_static_world();
    void build_model_registries(const merian::CommandBufferHandle& cmd);
    void init_particle_batch();
    void refresh_entities(const merian::CommandBufferHandle& cmd);
    void retire_stale_entity_slots();
    void cycle_animated_materials();
    void teardown_world();

  private:
    merian::MaterialModelID quake_material_type_id{};

    vk::Extent3D resolution{};

    // Game thread / synchronization.
    std::thread game_thread;
    std::atomic_bool game_running{true};
    merian::ConcurrentQueue<bool> sync_gamestate;
    merian::ConcurrentQueue<float> sync_render;

    bool render_next = false;
    bool update_gamestate = true;
    uint64_t frame = 0;
    uint64_t last_worldspawn_frame = 0;
    double server_fps = 0;

    merian::float3 sun_color{};
    merian::float3 sun_direction{0, 0, 1};
    float volume_max_t = 1000.F;

    // Static brush world: rebuilt on every worldspawn. The previous map's
    // meshes / nodes are torn down first.
    bool world_meshes_built = false;
    merian::NodeID world_node_id = merian::NODE_ID_INVALID;
    std::vector<merian::MeshID> world_mesh_ids;

    // Partition key for static brush surfaces. Two surfaces sharing the same
    // (texture_t*, surf->flags) tuple share a material and a mesh.
    struct TexFlagsKey {
        texture_t* tex;
        int surf_flags;
        bool operator==(const TexFlagsKey& o) const noexcept {
            return tex == o.tex && surf_flags == o.surf_flags;
        }
        bool operator<(const TexFlagsKey& o) const noexcept {
            if (tex != o.tex)
                return tex < o.tex;
            return surf_flags < o.surf_flags;
        }
    };
    struct TexFlagsKeyHash {
        size_t operator()(const TexFlagsKey& k) const noexcept {
            return std::hash<texture_t*>()(k.tex) ^ (std::hash<int>()(k.surf_flags) << 1u);
        }
    };
    // Bits we care about for material partitioning. SURF_PLANEBACK is
    // per-vertex (geometry-level), SURF_DRAWTILED governs r_notexture
    // surfaces; both irrelevant. The rest distinguish material variants.
    static constexpr int SURF_INTERESTING_BITS =
        SURF_DRAWSKY | SURF_DRAWLAVA | SURF_DRAWSLIME | SURF_DRAWTELE | SURF_DRAWWATER;

    // Worldmodel + brush submodels share textures (loadmodel->textures[]),
    // so this single map covers both.
    std::unordered_map<TexFlagsKey, merian::MaterialID, TexFlagsKeyHash> material_id_for_tex;

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

    // Per-model info built at worldspawn; stable across frames.
    struct AliasModelInfo {
        merian::BufferHandle index_buffer; // device-local, uploaded once
        uint32_t vertex_count;
        uint32_t primitive_count;
        int numskins;
    };
    std::unordered_map<qmodel_t*, AliasModelInfo> alias_model_info;

    // One per (texture_t*, surf_flags) partition in a brush submodel.
    struct BrushSubmodelGeoPart {
        merian::BufferHandle vb; // device-local, model space
        merian::BufferHandle ib; // device-local
        uint32_t vertex_count;
        uint32_t primitive_count;
        merian::MaterialID material_id;
        bool has_alpha;
    };
    std::unordered_map<qmodel_t*, std::vector<BrushSubmodelGeoPart>> brush_submodel_geo;

    // Material registries built at worldspawn.
    struct AliasSkinKey {
        qmodel_t* model;
        int skin;
        bool operator==(const AliasSkinKey& o) const = default;
    };
    struct AliasSkinKeyHash {
        size_t operator()(const AliasSkinKey& k) const noexcept {
            return std::hash<qmodel_t*>()(k.model) ^ (std::hash<int>()(k.skin) << 1u);
        }
    };
    std::unordered_map<AliasSkinKey, merian::MaterialID, AliasSkinKeyHash>
        material_id_for_alias_skin;

    struct SpriteFrameKey {
        qmodel_t* model;
        int frame;
        bool operator==(const SpriteFrameKey& o) const = default;
    };
    struct SpriteFrameKeyHash {
        size_t operator()(const SpriteFrameKey& k) const noexcept {
            return std::hash<qmodel_t*>()(k.model) ^ (std::hash<int>()(k.frame) << 1u);
        }
    };
    std::unordered_map<SpriteFrameKey, merian::MaterialID, SpriteFrameKeyHash>
        material_id_for_sprite_frame;

    // Per-entity slot: one SceneNode + one or more MeshIDs.
    struct EntityMeshSlot {
        merian::NodeID node_id = merian::NODE_ID_INVALID;
        std::vector<merian::MeshID> mesh_ids;
        qmodel_t* model = nullptr;
        int kind = 0; // 0=alias, 1=brush, 2=sprite

        // Alias change detection: cached state that was last written.
        int cached_skinnum = -1;
        merian::TextureID cached_skin_texnum{};
        int cached_pose1 = -1;
        int cached_pose2 = -1;
        float cached_blend = -1.f;
        int cached_prev_pose1 = -1;
        int cached_prev_pose2 = -1;
        float cached_prev_blend = -1.f;
        vec3_t cached_origin = {};
        vec3_t cached_angles = {};
    };
    std::unordered_map<entity_t*, EntityMeshSlot> entity_slots;

    EntityMeshSlot& ensure_alias_slot(entity_t* ent);
    EntityMeshSlot& ensure_brush_slot(entity_t* ent, const merian::CommandBufferHandle& cmd);
    EntityMeshSlot& ensure_sprite_slot(entity_t* ent);
    void process_alias_model(EntityMeshSlot& slot, entity_t* ent);

    // Particle batch: single mesh, palette-encoded color.
    bool particle_mesh_built = false;
    merian::MeshID particle_mesh_id = 0;
    merian::NodeID particle_node_id = merian::NODE_ID_INVALID;
    merian::MaterialID particle_material_id = 0;
    double prev_cl_time = 0.0;

    // Input.
    merian::InputControllerHandle controller = std::make_shared<merian::DummyInputController>();
    std::shared_ptr<merian::InputListener> input_listener;
    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;
    bool raw_mouse_was_enabled = false;

    // Console commands queued from the graph/UI thread; executed on game
    // thread.
    std::queue<std::string> pending_commands;
    std::mutex pending_commands_mutex;

    // Properties.
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

    // HACK texture ids stored once at load.
    uint32_t texnum_blood = 0;
    uint32_t texnum_explosion = 0;

    merian::CameraID quake_camera;
};

using QuakeSceneHandle = std::shared_ptr<QuakeScene>;

} // namespace merian_quake
