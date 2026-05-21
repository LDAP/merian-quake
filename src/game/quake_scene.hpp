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

// One bit per geometry class, written into Mesh::instance_mask. Lets render
// passes include/exclude classes by ANDing with the TLAS trace mask.
enum class InstanceMask : uint8_t {
    WORLD = 1u << 0,        // worldspawn brushes
    BRUSH_ENTITY = 1u << 1, // moving/submodel brushes (doors, lifts, plats)
    ALIAS = 1u << 2,        // monsters, items, third-party players
    SPRITE = 1u << 3,       // sprite billboards
    PARTICLE = 1u << 4,     // particles
    VIEWENT = 1u << 5,      // first-person gun (cl.viewent)
    PLAYER_BODY = 1u << 6,  // local player's third-person body
};

constexpr uint8_t to_mask(InstanceMask m) {
    return static_cast<uint8_t>(m);
}

// Owns Quake's global lifecycle: QuakeSpasm init, the game thread, the input
// listener, the per-frame scene refresh, and the texture upload pump. Quake
// runs on static globals so only one instance can exist at a time.
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

    // Pass a dummy controller to detach.
    void set_controller(const merian::InputControllerHandle& controller);

    void queue_command(const std::string& command);

    // --- QuakeSpasm callbacks (invoked from the game thread via extern "C") ---
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

    // --- World lifecycle ---
    void unload_world();
    void load_world(const merian::CommandBufferHandle& cmd);
    void load_world_brushes();
    void register_alias_models(const merian::CommandBufferHandle& cmd);
    void register_sprite_models();
    void init_particle_batch();
    void update_sky();

    // --- Per-frame ---
    void sync_game_thread(float time_diff);
    void update_entities(const merian::CommandBufferHandle& cmd);
    void update_alias_entity(entity_t* ent,
                             const merian::CommandBufferHandle& cmd,
                             uint8_t instance_mask);
    void update_brush_entity(entity_t* ent,
                             const merian::CommandBufferHandle& cmd,
                             uint8_t instance_mask);
    void update_sprite_entity(entity_t* ent);
    void update_particles();
    void update_animated_materials();
    void update_camera_and_sun();

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

    // Static brush world: rebuilt on every worldspawn.
    bool world_meshes_built = false;
    merian::Scene::NodeID world_node_id = merian::Scene::NODE_ID_INVALID;
    std::vector<merian::Scene::MeshID> world_mesh_ids;

    // Surfaces sharing (texture, surf_flags) share a material and a mesh.
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
    // SURF_PLANEBACK is per-vertex, SURF_DRAWTILED selects r_notexture; the
    // rest carry meaningful material variants.
    static constexpr int SURF_INTERESTING_BITS =
        SURF_DRAWSKY | SURF_DRAWLAVA | SURF_DRAWSLIME | SURF_DRAWTELE | SURF_DRAWWATER;

    // Pre-upload CPU buffer for one (texture, surf_flags) partition.
    struct BrushSurfaceBucket {
        std::vector<merian::PackedVertexData> vertices;
        std::vector<merian::uint3> indices;
        texture_t* tex = nullptr;
        int surf_flags = 0;
    };
    std::unordered_map<TexFlagsKey, BrushSurfaceBucket, TexFlagsKeyHash>
    collect_brush_surfaces(qmodel_t* mod);
    // Lookup-or-create; only pushes onto animated_brush_materials on creation.
    merian::MaterialID register_brush_material(texture_t* tex, int surf_flags);

    // Shared by worldmodel and submodels (textures are model-private but the
    // same texture_t* gets re-used across submodels of the same map).
    std::unordered_map<TexFlagsKey, merian::MaterialID, TexFlagsKeyHash> material_id_for_tex;

    // R_TextureAnimation resolves the current frame each tick; we re-pack the
    // material when the resolved texnum changes so geometry isn't rebaked.
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
        merian::BufferHandle index_buffer;
        uint32_t vertex_count;
        uint32_t primitive_count;
        int numskins;
        // smooth per-pose normals; MDL's 162-direction quantization is too coarse for AD models
        std::vector<merian::float3> baked_normals;
    };
    std::unordered_map<qmodel_t*, AliasModelInfo> alias_model_info;

    // One uploaded VB/IB pair per (texture, surf_flags) partition.
    struct BrushSubmodelGeoPart {
        merian::BufferHandle vb;
        merian::BufferHandle ib;
        uint32_t vertex_count;
        uint32_t primitive_count;
        merian::MaterialID material_id;
        bool has_alpha;
    };
    std::unordered_map<qmodel_t*, std::vector<BrushSubmodelGeoPart>> brush_submodel_geo;

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

    // Shared mesh + material per mspriteframe_t*; orientation/scale live on the entity node.
    struct SpriteFrameInfo {
        merian::Scene::MeshID mesh_id;
        merian::MaterialID material_id;
    };
    std::unordered_map<mspriteframe_t*, SpriteFrameInfo> sprite_frame_info;

    // mod_alias / mod_brush own their per-entity meshes; mod_sprite shares from sprite_frame_info.
    struct EntityMeshSlot {
        merian::Scene::NodeID node_id = merian::Scene::NODE_ID_INVALID;
        merian::SmallVector<merian::Scene::MeshID, 1> mesh_ids;
        qmodel_t* model = nullptr;

        mspriteframe_t* cached_sprite_frame = nullptr;

        int cached_skinnum = -1;
        int cached_anim_frame = -1;
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
    // Holds last frame's slots until each surviving entity migrates back; what
    // remains belongs to entities that vanished (server-removed, culled, or
    // temp-entity slots recycled without nulling ent->model — e.g. expired
    // lightning beams) and is destroyed by release_unused_entities.
    std::unordered_map<entity_t*, EntityMeshSlot> previous_entity_slots;

    // Returns nullptr if the previous-frame slot is unusable (caller builds fresh).
    EntityMeshSlot* migrate_entity_slot(entity_t* ent);
    void release_unused_entities();
    void destroy_slot(EntityMeshSlot& slot);

    // Single particle mesh with palette-encoded color. The instance is only
    // attached while extraction yields non-empty geometry — Scene can't
    // upload empty meshes.
    merian::Scene::MeshID particle_mesh_id = 0;
    merian::Scene::NodeID particle_node_id = merian::Scene::NODE_ID_INVALID;
    merian::MaterialID particle_material_id = 0;
    bool particle_instance_attached = false;
    double prev_cl_time = 0.0;

    // Input.
    merian::InputControllerHandle controller = std::make_shared<merian::DummyInputController>();
    std::shared_ptr<merian::InputListener> input_listener;
    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;
    bool raw_mouse_was_enabled = false;

    // Queued from graph/UI thread, drained by the game thread.
    std::queue<std::string> pending_commands;
    std::mutex pending_commands_mutex;

    // Properties.
    int default_filtering = 0;
    std::string startup_commands{};
    int stop_after_worldspawn = -1;
    bool rebuild_after_stop = true;
    bool overwrite_sun = false;
    merian::float3 overwrite_sun_dir{0, 0, 1};
    merian::float3 overwrite_sun_col{0};
    bool mu_t_s_overwrite = false;
    float mu_t = 0.0F;
    merian::float3 mu_s_div_mu_t{1};
    bool reproducible_renders = false;

    merian::Scene::CameraID quake_camera;

    // Per-frame entity counters; newly_created bumps in update_*_entity when a slot is built.
    struct CategoryStats {
        uint32_t active = 0;
        uint32_t newly_created = 0;
    };
    struct EntityStats {
        CategoryStats alias;
        CategoryStats brush;
        CategoryStats sprite;
    };
    EntityStats current_entity_stats{};
    EntityStats last_frame_entity_stats{};
};

using QuakeSceneHandle = std::shared_ptr<QuakeScene>;

} // namespace merian_quake
