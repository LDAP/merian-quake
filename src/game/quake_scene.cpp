#include "game/quake_scene.hpp"

#include "../../res/shader/config.h"
#include "game/quake_extraction.hpp"
#include "game/quake_material.hpp"
#include "game/quake_meshes.hpp"
#include "merian/utils/audio/audio_device_provider.hpp"
#include "merian/utils/camera/camera.hpp"
#include "merian/utils/colors.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/string.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

extern "C" {
#include "bgmusic.h"
#include "quakedef.h"
#include "screen.h"

extern cvar_t cl_maxpitch;
extern cvar_t cl_minpitch;
extern qboolean scr_drawloading;
}

namespace merian_quake {

namespace {

// Quake's static globals make it a singleton. We mirror that on our side
// with a single QuakeData carrying the scene pointer plus the QuakeSpasm
// runtime state (params, audio, parsed worldspawn).
struct QuakeData {
    QuakeScene* quake_scene{nullptr};
    quakeparms_t params;
    merian::AudioDeviceHandle audio_device;

    merian::float3 current_sun_color{};
    merian::float3 current_sun_direction{};

    float timediff = 0;
};
QuakeData g_quake_data;

void init_quakespasm(const uint32_t quakespasm_argc, const char** quakespasm_argv) {
    std::vector<const char*> quakespasm_args = {"quakespasm"};
    if (quakespasm_argc > 0) {
        quakespasm_args.resize(1 + quakespasm_argc);
        std::memcpy(&quakespasm_args[1], quakespasm_argv,
                    quakespasm_argc * sizeof(quakespasm_argv[0]));
    }

    g_quake_data.params.argc = static_cast<int>(quakespasm_args.size());
    g_quake_data.params.argv = const_cast<char**>(quakespasm_args.data());
    g_quake_data.params.errstate = 0;
    g_quake_data.params.memsize = 256 * 1024 * 1024;
    g_quake_data.params.membase = malloc(g_quake_data.params.memsize);

    srand(1337);
    COM_InitArgv(g_quake_data.params.argc, g_quake_data.params.argv);
    Sys_Init();

    Sys_Printf("Quake %1.2f (c) id Software\n", VERSION);
    Sys_Printf("GLQuake %1.2f (c) id Software\n", GLQUAKE_VERSION);
    Sys_Printf("FitzQuake %1.2f (c) John Fitzgibbons\n", FITZQUAKE_VERSION);
    Sys_Printf("FitzQuake SDL port (c) SleepwalkR, Baker\n");
    Sys_Printf("QuakeSpasm " QUAKESPASM_VER_STRING " (c) Ozkan Sezer, Eric Wasylishen & others\n");

    Host_Init();

    key_dest = key_game;
    m_state = m_none;
}

void shutdown_quakespasm() {
    CL_Disconnect();
    Host_ShutdownServer(false);
    Host_Shutdown();

    free(g_quake_data.params.membase);
    g_quake_data.quake_scene = nullptr;
}

void parse_worldspawn() {
    merian::float3& quake_sun_col = g_quake_data.current_sun_color;
    merian::float3& quake_sun_dir = g_quake_data.current_sun_direction;

    std::map<std::string, std::string> worldspawn_props;
    char key[128];
    char value[4096];
    const char* data;

    data = COM_Parse(cl.worldmodel->entities);
    if (data == nullptr)
        return;
    if (com_token[0] != '{')
        return;
    while (true) {
        data = COM_Parse(data);
        if (data == nullptr)
            return;
        if (com_token[0] == '}')
            break;
        if (com_token[0] == '_')
            q_strlcpy(key, com_token + 1, sizeof(key));
        else
            q_strlcpy(key, com_token, sizeof(key));
        while ((key[0] != 0) && key[strlen(key) - 1] == ' ')
            key[strlen(key) - 1] = 0;
        data = COM_Parse(data);
        if (data == nullptr)
            return;
        q_strlcpy(value, com_token, sizeof(value));
        SPDLOG_DEBUG("{} {}", key, value);
        worldspawn_props[key] = value;
    }

    quake_sun_col = merian::float3(0);
    for (const std::string k : {"sunlight", "sunlight2", "sunlight3"}) {
        if (worldspawn_props.contains(k)) {
            merian::float3 col(0);
            if (worldspawn_props.contains(k + "_color")) {
                sscanf(worldspawn_props[k + "_color"].c_str(), "%f %f %f", &col.r, &col.g, &col.b);
            } else {
                col = merian::float3(1);
            }
            float intensity = std::stoi(worldspawn_props[k]);
            col *= intensity;
            col /= 4000.0F;
            if (merian::yuv_luminance(col) > merian::yuv_luminance(quake_sun_col)) {
                quake_sun_col = col;
            }
        }
    }

    if (worldspawn_props.contains("sun_mangle")) {
        float angles[3];
        sscanf(worldspawn_props["sun_mangle"].c_str(), "%f %f %f", &angles[1], &angles[0],
               &angles[2]);
        float right[3];
        float up[3];
        angles[1] -= 180;
        AngleVectors(angles, &quake_sun_dir.x, right, up);
    } else {
        quake_sun_dir = merian::float3(1, 1, 1);
    }

    if (worldspawn_props.contains("sky") && worldspawn_props["sky"] == "stormydays_") {
        quake_sun_dir = merian::float3(1, -1, 1);
        quake_sun_col = merian::float3(1.1, 1.0, 0.9);
        quake_sun_col *= 6.0;
    }

    const float max_col =
        merian::max(merian::max(quake_sun_col.r, quake_sun_col.g), quake_sun_col.b);
    if (max_col > MAX_SUN_COLOR)
        quake_sun_col = quake_sun_col / max_col * MAX_SUN_COLOR;
    quake_sun_dir = merian::normalize(quake_sun_dir);
}

} // namespace

// QuakeSpasm callbacks --------------------------------------------------------

extern "C" void VID_Changed_f(cvar_t* /*var*/) {
    if (g_quake_data.quake_scene)
        g_quake_data.quake_scene->cb_VID_Changed();
}

extern "C" void QS_worldspawn() {
    if (g_quake_data.quake_scene)
        g_quake_data.quake_scene->cb_QS_worldspawn();
}

extern "C" void QS_texture_load(gltexture_t* glt, uint32_t* data) {
    if (g_quake_data.quake_scene)
        g_quake_data.quake_scene->cb_QS_texture_load(glt, data);
}

extern "C" void IN_Move(usercmd_t* cmd) {
    if (g_quake_data.quake_scene)
        g_quake_data.quake_scene->cb_IN_Move(cmd);
}

extern "C" void R_RenderScene() {
    if (g_quake_data.quake_scene)
        g_quake_data.quake_scene->cb_R_RenderScene();
}

extern "C" qboolean SNDDMA_Init(dma_t* dma) {
    if (!g_quake_data.audio_device)
        return false;

    const auto callback = [](uint8_t* stream, int len) {
        int buffersize = shm->samples * (shm->samplebits / 8);
        int pos, tobufend;
        int len1, len2;

        if (!shm) {
            memset(stream, 0, len);
            return;
        }
        pos = (shm->samplepos * (shm->samplebits / 8));
        if (pos >= buffersize)
            shm->samplepos = pos = 0;
        tobufend = buffersize - pos;
        len1 = len;
        len2 = 0;
        if (len1 > tobufend) {
            len1 = tobufend;
            len2 = len - len1;
        }
        memcpy(stream, shm->buffer + pos, len1);
        if (len2 <= 0) {
            shm->samplepos += (len1 / (shm->samplebits / 8));
        } else {
            memcpy(stream + len1, shm->buffer, len2);
            shm->samplepos = (len2 / (shm->samplebits / 8));
        }
        if (shm->samplepos >= buffersize)
            shm->samplepos = 0;
    };

    merian::AudioDevice::AudioSpec desired = {
        merian::AudioDevice::FORMAT_S16_LSB,
        1024,
        static_cast<int>(snd_mixspeed.value),
        2,
    };
    if (desired.samplerate <= 11025)
        desired.buffersize = 256;
    else if (desired.samplerate <= 22050)
        desired.buffersize = 512;
    else if (desired.samplerate <= 44100)
        desired.buffersize = 1024;
    else if (desired.samplerate <= 56000)
        desired.buffersize = 2048;
    else
        desired.buffersize = 4096;

    auto actual = g_quake_data.audio_device->open_device(desired, callback);
    if (!actual)
        return false;

    memset(static_cast<void*>(dma), 0, sizeof(dma_t));
    shm = dma;
    shm->samplebits = (actual->format & 0xFF);
    shm->signed8 = (actual->format == merian::AudioDevice::FORMAT_S8);
    shm->speed = actual->samplerate;
    shm->channels = actual->channels;
    int tmp = (actual->buffersize * actual->channels) * 10;
    if (tmp & (tmp - 1)) {
        int val = 1;
        while (val < tmp)
            val <<= 1;
        tmp = val;
    }
    shm->samples = tmp;
    shm->samplepos = 0;
    shm->submission_chunk = 1;

    size_t buffersize = shm->samples * (shm->samplebits / 8);
    shm->buffer = static_cast<unsigned char*>(calloc(1, buffersize));

    g_quake_data.audio_device->unpause_audio();
    return 1;
}

extern "C" int SNDDMA_GetDMAPos(void) {
    if (shm != nullptr)
        return shm->samplepos;
    return 0;
}

extern "C" void SNDDMA_Shutdown(void) {
    if (shm != nullptr) {
        if (shm->buffer != nullptr)
            free(shm->buffer);
        shm->buffer = nullptr;
        shm = nullptr;
    }
    g_quake_data.audio_device.reset();
}

extern "C" void SNDDMA_LockBuffer(void) {
    if (g_quake_data.audio_device)
        g_quake_data.audio_device->lock_device();
}
extern "C" void SNDDMA_Submit(void) {
    if (g_quake_data.audio_device)
        g_quake_data.audio_device->unlock_device();
}
extern "C" void SNDDMA_BlockSound(void) {
    if (g_quake_data.audio_device)
        g_quake_data.audio_device->pause_audio();
}
extern "C" void SNDDMA_UnblockSound(void) {
    if (g_quake_data.audio_device)
        g_quake_data.audio_device->unpause_audio();
}

// QuakeScene ------------------------------------------------------------------

QuakeScene::QuakeScene(const merian::ShaderCompileContextHandle& compile_context,
                       const merian::ContextHandle& context,
                       const merian::ResourceAllocatorHandle& allocator,
                       const merian::ShaderObjectAllocatorHandle& obj_allocator,
                       const merian::MaterialSystemHandle& material_system,
                       const uint32_t quakespasm_argc,
                       const char** quakespasm_argv)
    : merian::Scene(compile_context, context, allocator, obj_allocator, material_system) {

    if (g_quake_data.quake_scene != nullptr) {
        throw std::runtime_error{"only one QuakeScene can exist (Quake uses static globals)"};
    }
    g_quake_data.quake_scene = this;

    quake_material_type_id = material_system->register_material_type(
        QUAKE_MATERIAL_SLANG_TYPE_NAME, QUAKE_MATERIAL_SLANG_MODULE_PATH);

    auto cam = std::make_shared<merian::Camera>(merian::float3(1, 0, 0), merian::float3(0, 0, 0),
                                                get_up(), 90.F, 16.F / 9.F, 0.01F, 1e5f);
    quake_camera = add_camera(std::move(cam));

    if (const auto audio_provider = context->find_provider<merian::AudioDeviceProvider>(true)) {
        g_quake_data.audio_device = audio_provider->create_audio_device();
    } else {
        g_quake_data.audio_device = nullptr;
    }
    host_parms = &g_quake_data.params;

    init_quakespasm(quakespasm_argc, quakespasm_argv);

    game_thread = std::thread([this] {
        merian::Stopwatch sw;
        while (game_running.load()) {
            {
                std::lock_guard<std::mutex> lock(pending_commands_mutex);
                if (!pending_commands.empty()) {
                    Cmd_ExecuteString(pending_commands.front().c_str(), src_command);
                    pending_commands.pop();
                }
            }
            try {
                render_this_frame = false;
                Host_Frame(g_quake_data.timediff);
                if (!render_this_frame) {
                    sync_gamestate.push(true, 1);
                    if (!game_running.load()) {
                        throw std::runtime_error{"quit"};
                    }
                    g_quake_data.timediff = sync_render.pop();
                }
            } catch (const std::runtime_error&) {
                // game quit; loop checks game_running and exits.
            }
            server_fps = 1 / sw.seconds();
            sw.reset();
        }
    });
    sync_gamestate.pop();

    resolution =
        vk::Extent3D{static_cast<uint32_t>(vid.width), static_cast<uint32_t>(vid.height), 1U};
}

QuakeScene::~QuakeScene() {
    game_running.store(false);
    // unblock the game thread if it's waiting on either queue
    sync_render.push(0);
    sync_render.push(0);
    if (game_thread.joinable())
        game_thread.join();

    shutdown_quakespasm();
}

float QuakeScene::get_time(const float /*time*/) {
    return static_cast<float>(cl.time);
}

void QuakeScene::set_controller(const merian::InputControllerHandle& controller) {
    if (input_listener && this->controller) {
        // detach existing listener so we don't double-register on the new
        // controller.
        this->controller->add_listener(input_listener, 0);
    }
    this->controller = controller ? controller
                                  : std::static_pointer_cast<merian::InputController>(
                                        std::make_shared<merian::DummyInputController>());
    register_input_listener(this->controller);
}

void QuakeScene::queue_command(const std::string& command) {
    std::lock_guard<std::mutex> lock(pending_commands_mutex);
    pending_commands.push(command);
}

void QuakeScene::register_input_listener(const merian::InputControllerHandle& controller) {
    struct QuakeInputListener : merian::InputListener {
        QuakeScene* scene;
        explicit QuakeInputListener(QuakeScene* s) : scene(s) {}

        bool on_key(merian::InputController& /*c*/,
                    merian::InputController::Key key,
                    merian::InputController::KeyStatus action,
                    int /*mods*/) override {
            using K = merian::InputController::Key;
            const int ki = static_cast<int>(key);
            const int A = static_cast<int>(K::A);
            const int N0 = static_cast<int>(K::NUM_0);
            int qkey = 0;
            if (ki >= A && ki <= static_cast<int>(K::Z))
                qkey = 'a' + (ki - A);
            else if (ki >= N0 && ki <= static_cast<int>(K::NUM_9))
                qkey = '0' + (ki - N0);
            else {
                // clang-format off
                static const std::unordered_map<merian::InputController::Key, int> keymap = {
                    {K::TAB,        K_TAB},
                    {K::ENTER,      K_ENTER},
                    {K::ESCAPE,     K_ESCAPE},
                    {K::SPACE,      K_SPACE},
                    {K::BACKSPACE,  K_BACKSPACE},
                    {K::UP,         K_UPARROW},
                    {K::DOWN,       K_DOWNARROW},
                    {K::LEFT,       K_LEFTARROW},
                    {K::RIGHT,      K_RIGHTARROW},
                    {K::LEFT_ALT,   K_ALT},
                    {K::LEFT_CTRL,  K_CTRL},
                    {K::LEFT_SHIFT, K_SHIFT},
                    {K::F1,  K_F1},  {K::F2,  K_F2},  {K::F3,  K_F3},
                    {K::F4,  K_F4},  {K::F5,  K_F5},  {K::F6,  K_F6},
                    {K::F7,  K_F7},  {K::F8,  K_F8},  {K::F9,  K_F9},
                    {K::F10, K_F10}, {K::F11, K_F11}, {K::F12, K_F12},
                };
                // clang-format on
                if (const auto it = keymap.find(key); it != keymap.end())
                    qkey = it->second;
            }
            if (qkey == 0)
                return true;
            using KS = merian::InputController::KeyStatus;
            if (action == KS::PRESS)
                Key_Event(qkey, true);
            else if (action == KS::RELEASE)
                Key_Event(qkey, false);
            return true;
        }

        bool on_cursor(merian::InputController& c, double xpos, double ypos) override {
            const bool raw = c.is_mouse_grabbed();
            if (raw) {
                scene->mouse_x = xpos;
                scene->mouse_y = ypos;
            }
            if (raw != scene->raw_mouse_was_enabled || !raw) {
                scene->mouse_x = scene->mouse_oldx = xpos;
                scene->mouse_y = scene->mouse_oldy = ypos;
            }
            scene->raw_mouse_was_enabled = raw;
            return true;
        }

        bool on_mouse_button(merian::InputController& /*c*/,
                             merian::InputController::MouseButton button,
                             merian::InputController::KeyStatus status) override {
            using MB = merian::InputController::MouseButton;
            using KS = merian::InputController::KeyStatus;
            if (button == MB::UNKNOWN)
                return true;
            const int remap[] = {K_MOUSE1, K_MOUSE2, K_MOUSE3, K_MOUSE4, K_MOUSE5};
            Key_Event(remap[static_cast<int>(button)], status == KS::PRESS);
            return true;
        }

        bool on_scroll(merian::InputController& /*c*/, double xoffset, double yoffset) override {
            if (yoffset > 0) {
                Key_Event(K_MWHEELUP, true);
                Key_Event(K_MWHEELUP, false);
            } else if (xoffset < 0) {
                Key_Event(K_MWHEELDOWN, true);
                Key_Event(K_MWHEELDOWN, false);
            }
            return true;
        }
    };

    input_listener = std::make_shared<QuakeInputListener>(this);
    controller->add_listener(input_listener, 0);
}

// QuakeSpasm callback bodies --------------------------------------------------

void QuakeScene::cb_VID_Changed() {
    resolution =
        vk::Extent3D{static_cast<uint32_t>(vid.width), static_cast<uint32_t>(vid.height), 1U};
}

void QuakeScene::cb_QS_worldspawn() {
    SPDLOG_DEBUG("worldspawn");
    parse_worldspawn();
    last_worldspawn_frame = frame_counter;
    constant_data_dirty = true;
}

void QuakeScene::cb_IN_Move(usercmd_t* cmd) {
    SPDLOG_TRACE("move");
    int dmx = (mouse_x - mouse_oldx) * sensitivity.value;
    int dmy = (mouse_y - mouse_oldy) * sensitivity.value;
    mouse_oldx = mouse_x;
    mouse_oldy = mouse_y;

    if ((in_strafe.state & 1) || (lookstrafe.value && (in_mlook.state & 1)))
        cmd->sidemove += m_side.value * dmx;
    else
        cl.viewangles[YAW] -= m_yaw.value * dmx;

    if (in_mlook.state & 1) {
        if (dmx || dmy)
            V_StopPitchDrift();
    }

    if ((in_mlook.state & 1) && !(in_strafe.state & 1)) {
        cl.viewangles[PITCH] += m_pitch.value * dmy;
        cl.viewangles[PITCH] = std::min(cl.viewangles[PITCH], cl_maxpitch.value);
        cl.viewangles[PITCH] = std::max(cl.viewangles[PITCH], cl_minpitch.value);
    } else {
        if ((in_strafe.state & 1) && noclip_anglehack)
            cmd->upmove -= m_forward.value * dmy;
        else
            cmd->forwardmove -= m_forward.value * dmy;
    }
}

void QuakeScene::cb_R_RenderScene() {
    if (!game_running.load()) {
        throw std::runtime_error{"quit"};
    }
    render_this_frame = true;
    sync_gamestate.push(true, 1);
    g_quake_data.timediff = sync_render.pop();
}

void QuakeScene::cb_QS_texture_load(gltexture_t* glt, const uint32_t* data) {
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
    const std::string source = strcmp(glt->source_file, "") == 0 ? "memory" : glt->source_file;
    SPDLOG_DEBUG("texture_load {} {} {}x{} from {}, frame: {}", glt->texnum, glt->name, glt->width,
                 glt->height, source, glt->visframe);
#endif
    if (glt->width == 0 || glt->height == 0) {
        SPDLOG_WARN("image extent was 0. skipping");
        return;
    }

    if (strcmp(glt->name, "progs/gib_1.mdl:frame0") == 0)
        texnum_blood = glt->texnum;
    if (strcmp(glt->name, "progs/s_exp_big.spr:frame10") == 0)
        texnum_explosion = glt->texnum;

    PendingTexture pt;
    pt.texnum = glt->texnum;
    pt.width = glt->width;
    pt.height = glt->height;
    pt.flags = glt->flags;
    pt.name = glt->name;
    pt.linear = merian::ends_with(glt->name, "_norm") || merian::ends_with(glt->name, "_gloss");
    pt.rgba.assign(data, data + (pt.width * pt.height));

    std::lock_guard<std::mutex> lock(pending_uploads_mutex);
    // last write wins: drop any prior pending upload for the same slot.
    pending_uploads.erase(
        std::remove_if(pending_uploads.begin(), pending_uploads.end(),
                       [&](const PendingTexture& other) { return other.texnum == pt.texnum; }),
        pending_uploads.end());
    pending_uploads.push_back(std::move(pt));
}

// per-frame --------------------------------------------------------------------

void QuakeScene::on_update(const merian::CommandBufferHandle& cmd,
                           const float /*time*/,
                           const float time_diff,
                           const uint32_t /*frame*/) {
    if (update_gamestate) {
        sync_render.push(time_diff, 1);
        sync_gamestate.pop();
    }

    drain_pending_uploads(cmd);

    render_this_frame = render_this_frame && (scr_drawloading == 0);

    if ((cl.worldmodel != nullptr) && frame_counter == last_worldspawn_frame) {
        // First frame after a (re)load: pin gamestate to the live game and
        // drop any UI/menu state QuakeSpasm raised during loading.
        key_dest = key_game;
        m_state = m_none;
        sv_player = nullptr;
        // First worldspawn after construction: bake the BSP into static
        // brush meshes (one per material partition). Subsequent
        // worldspawns are ignored — proper map-change teardown lands
        // when Scene::remove_mesh is added.
        if (!world_meshes_built) {
            rebuild_static_world();
            world_meshes_built = true;
        }
        if (!dynamic_meshes_built) {
            init_dynamic_meshes();
            dynamic_meshes_built = true;
        }
    }

    if (dynamic_meshes_built && (cl.worldmodel != nullptr)) {
        refresh_dynamic_meshes();
    }

    if (world_meshes_built) {
        cycle_animated_materials();
    }

    refresh_render_info(render_this_frame);

    const bool in_game = update_gamestate && key_dest == key_game;
    controller->set_mouse_grabbed(in_game);
    if (input_listener)
        controller->add_listener(input_listener, in_game ? 100 : 0);

    if (stop_after_worldspawn >= 0 &&
        (frame_counter - last_worldspawn_frame) == static_cast<uint64_t>(stop_after_worldspawn)) {
        update_gamestate = false;
    }

    if (constant_data_dirty)
        constant_data_dirty = false;
    frame_counter++;

    run_startup_commands_if_needed();
}

void QuakeScene::drain_pending_uploads(const merian::CommandBufferHandle& cmd) {
    std::vector<PendingTexture> uploads;
    {
        std::lock_guard<std::mutex> lock(pending_uploads_mutex);
        uploads.swap(pending_uploads);
    }
    if (uploads.empty())
        return;

    const auto& texture_manager = get_texture_manager();
    auto& alloc = const_cast<merian::ResourceAllocatorHandle&>(get_allocator());
    (void)alloc;
    for (const auto& tex : uploads) {
        SPDLOG_DEBUG("uploading texture {}", tex.texnum);

        vk::Filter mag_filter;
        if (default_filtering == 0) {
            mag_filter =
                ((tex.flags & TEXPREF_LINEAR) != 0U) ? vk::Filter::eLinear : vk::Filter::eNearest;
        } else {
            mag_filter =
                ((tex.flags & TEXPREF_NEAREST) != 0U) ? vk::Filter::eNearest : vk::Filter::eLinear;
        }
        const bool srgb = !tex.linear;
        const bool generate_mipmaps = (tex.flags & TEXPREF_MIPMAP) != 0U;

        texture_manager->set_texture_from_rgba8(static_cast<merian::TextureID>(tex.texnum), cmd,
                                                tex.rgba.data(), tex.width, tex.height,
                                                vk::SamplerAddressMode::eRepeat, mag_filter,
                                                vk::Filter::eLinear, srgb, generate_mipmaps);
    }
}

void QuakeScene::refresh_render_info(const bool render_this_frame_) {
    if (constant_data_dirty) {
        if (overwrite_sun) {
            sun_color = overwrite_sun_col;
            sun_direction = overwrite_sun_dir;
        } else {
            sun_color = g_quake_data.current_sun_color;
            sun_direction = g_quake_data.current_sun_direction;
        }
        if (merian::length(sun_direction) > 0)
            sun_direction = merian::normalize(sun_direction);
    }

    if (!render_this_frame_)
        return;

    const auto cam = get_camera(quake_camera);
    assert(cam);

    float fwd[3];
    float rgt[3];
    float up[3];
    AngleVectors(r_refdef.viewangles, fwd, rgt, up);

    const merian::float3 pos = merian::as_float3(r_refdef.vieworg);
    const merian::float3 fwd_v(fwd[0], fwd[1], fwd[2]);
    const float aspect =
        (resolution.height > 0)
            ? (static_cast<float>(resolution.width) / static_cast<float>(resolution.height))
            : (16.F / 9.F);
    cam->look_at(pos, pos + fwd_v, get_up(), r_refdef.fov_x);
    cam->set_aspect_ratio(aspect);
}

namespace {

// Compact key identifying a per-surface material partition for static
// brush geometry. Surfaces with identical keys share a single QuakeBrushMesh
// (and therefore a single BLAS entry) inside the worldspawn instance.
struct BrushMaterialKey {
    uint32_t base_texnum;
    uint32_t fb_texnum;
    uint32_t normal_texnum;
    uint32_t gloss_texnum;
    uint16_t surface_flags;
    uint8_t alpha_mode;
    bool has_alpha;

    bool operator<(const BrushMaterialKey& o) const noexcept {
        return std::tie(base_texnum, fb_texnum, normal_texnum, gloss_texnum, surface_flags,
                        alpha_mode, has_alpha) <
               std::tie(o.base_texnum, o.fb_texnum, o.normal_texnum, o.gloss_texnum,
                        o.surface_flags, o.alpha_mode, o.has_alpha);
    }
};

uint16_t classify_surface_flags(const msurface_t* surf, const texture_t* base_tex) {
    if ((surf->flags & SURF_DRAWSKY) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Sky);
    if ((surf->flags & SURF_DRAWLAVA) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Lava);
    if ((surf->flags & SURF_DRAWSLIME) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Slime);
    if ((surf->flags & SURF_DRAWTELE) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Tele);
    if ((surf->flags & SURF_DRAWWATER) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Water);
    if ((base_tex != nullptr) && (base_tex->gltexture != nullptr) &&
        (strstr(base_tex->gltexture->name, "wfall") != nullptr))
        return static_cast<uint16_t>(QuakeSurfaceFlags::Waterfall);
    return static_cast<uint16_t>(QuakeSurfaceFlags::None);
}

} // namespace

void QuakeScene::rebuild_static_world() {
    if (cl.worldmodel == nullptr)
        return;

    qmodel_t* world = cl.worldmodel;

    if (world_node_id == merian::NODE_ID_INVALID) {
        merian::SceneNode root{};
        root.name = "worldspawn";
        world_node_id = add_node(std::move(root));
    }

    struct BrushBucket {
        std::vector<merian::PackedVertexData> vertices;
        std::vector<merian::uint3> indices;
        BrushMaterialKey key;
    };
    std::map<BrushMaterialKey, BrushBucket> buckets;

    const auto& material_system = get_material_system();

    for (int i = 0; i < world->nummodelsurfaces; i++) {
        msurface_t* surf = &world->surfaces[world->firstmodelsurface + i];
        if (surf->texinfo == nullptr || surf->texinfo->texture == nullptr)
            continue;

        texture_t* base_tex = surf->texinfo->texture;
        if (strcmp(base_tex->name, "skip") == 0)
            continue;

        gltexture_t* gltex = base_tex->gltexture;

        BrushMaterialKey key{};
        key.base_texnum = (gltex != nullptr) ? gltex->texnum : 0u;
        key.fb_texnum = (base_tex->fullbright != nullptr) ? base_tex->fullbright->texnum : 0u;
        key.normal_texnum = (base_tex->norm != nullptr) ? base_tex->norm->texnum : 0u;
        key.gloss_texnum = (base_tex->gloss != nullptr) ? base_tex->gloss->texnum : 0u;
        key.surface_flags = classify_surface_flags(surf, base_tex);
        key.has_alpha = (gltex != nullptr) && ((gltex->flags & TEXPREF_ALPHA) != 0u);
        key.alpha_mode = key.has_alpha ? 0u : 15u;

        auto& bucket = buckets[key];
        bucket.key = key;

        // Surface plane normal (already in world space; no model transform
        // for the worldspawn entity).
        merian::float3 plane_n = merian::as_float3(surf->plane->normal);
        if ((surf->flags & SURF_PLANEBACK) != 0)
            plane_n = -plane_n;
        const uint32_t enc_n = merian::encode_normal(merian::normalize(plane_n));

        for (glpoly_t* p = surf->polys; p != nullptr; p = nullptr) {
            const uint32_t base_vertex = static_cast<uint32_t>(bucket.vertices.size());
            for (int v = 0; v < p->numverts; v++) {
                merian::PackedVertexData pv{};
                pv.position = merian::as_float3(p->verts[v]);
                pv.encoded_normal = enc_n;
                pv.uv = merian::half2(p->verts[v][3], p->verts[v][4]);
                // Tangent: leave a zero placeholder for now (path tracer
                // recomputes from triangle edges). Sign bit zero.
                pv.encoded_tangent = 0;
                bucket.vertices.push_back(pv);
            }

            for (int v = 2; v < p->numverts; v++) {
                bucket.indices.push_back(merian::uint3(base_vertex, base_vertex + uint32_t(v) - 1u,
                                                       base_vertex + uint32_t(v)));
            }
        }
    }

    // Map base_texnum back to the underlying texture_t* so we can record the
    // animation chain alongside the material we are about to allocate. The
    // bucket key only carries the texnum.
    std::unordered_map<uint32_t, texture_t*> base_tex_by_texnum;
    for (int i = 0; i < world->nummodelsurfaces; i++) {
        msurface_t* surf = &world->surfaces[world->firstmodelsurface + i];
        if (surf->texinfo == nullptr || surf->texinfo->texture == nullptr)
            continue;
        texture_t* base_tex = surf->texinfo->texture;
        if (base_tex->gltexture != nullptr)
            base_tex_by_texnum.emplace(base_tex->gltexture->texnum, base_tex);
    }

    animated_brush_materials.clear();

    for (auto& [key, bucket] : buckets) {
        if (bucket.indices.empty())
            continue;

        QuakeMaterial mat;
        mat.payload.base_tex = static_cast<merian::TextureID>(key.base_texnum);
        mat.payload.fullbright_tex = (key.fb_texnum != 0u)
                                         ? static_cast<merian::TextureID>(key.fb_texnum)
                                         : QUAKE_NO_TEXTURE;
        mat.payload.normal_tex = (key.normal_texnum != 0u)
                                     ? static_cast<merian::TextureID>(key.normal_texnum)
                                     : QUAKE_NO_TEXTURE;
        mat.payload.gloss_tex = (key.gloss_texnum != 0u)
                                    ? static_cast<merian::TextureID>(key.gloss_texnum)
                                    : QUAKE_NO_TEXTURE;
        mat.payload.surface_flags = key.surface_flags;
        mat.payload.alpha_mode = key.alpha_mode;

        const merian::MaterialID material_id =
            material_system->add_material(quake_material_type_id, mat);

        auto mesh = std::make_unique<QuakeBrushMesh>();
        mesh->name = fmt::format("worldspawn:tex{}", key.base_texnum);
        mesh->material_id = material_id;
        // Static, opaque, CCW-front (Quake convention; alpha-test on world
        // brushes lands in a follow-up gbuffer pass).
        mesh->flags =
            merian::GeometryFlags::IsOpaque | merian::GeometryFlags::FrontCounterClockwise;
        mesh->vertices = std::move(bucket.vertices);
        mesh->indices = std::move(bucket.indices);

        const merian::MeshID mesh_id = add_mesh(std::move(mesh));
        add_mesh_instance(mesh_id, world_node_id);

        auto it = base_tex_by_texnum.find(key.base_texnum);
        if (it != base_tex_by_texnum.end() && it->second->anim_total > 0) {
            animated_brush_materials.push_back(AnimatedBrushMaterial{
                material_id, it->second, mat.payload.fullbright_tex, mat.payload.normal_tex,
                mat.payload.gloss_tex, key.surface_flags, key.alpha_mode,
                static_cast<merian::TextureID>(key.base_texnum)});
        }
    }

    SPDLOG_DEBUG("static world: {} brush partitions, {} surfaces, {} animated materials",
                 buckets.size(), world->nummodelsurfaces, animated_brush_materials.size());
}

namespace {

// Seed each pre-allocated dynamic mesh with one degenerate triangle so the
// vertex/index buffers are non-empty before the first refresh. Scene::update's
// upload path skips zero-sized buffers, and the BLAS builder requires a
// non-zero maxVertex; this avoids the special-case in both places.
void seed_with_degenerate_triangle(QuakeBrushMesh& mesh) {
    mesh.vertices.assign(3, merian::PackedVertexData{});
    mesh.indices.assign(1, merian::uint3(0u, 1u, 2u));
}

void ensure_non_empty(QuakeBrushMesh& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        seed_with_degenerate_triangle(mesh);
    }
}

QuakeMaterial make_default_dynamic_material() {
    QuakeMaterial m;
    // Leave all texture refs as QUAKE_NO_TEXTURE; v1 dynamic content is
    // rendered untextured. Material upload still needs a payload size and
    // a slot id.
    m.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::None);
    m.payload.alpha_mode = 15; // fully opaque
    return m;
}

QuakeMaterial make_default_sprite_material() {
    QuakeMaterial m;
    m.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::Sprite);
    m.payload.alpha_mode = 0; // use texture alpha
    return m;
}

QuakeMaterial make_default_particle_material() {
    QuakeMaterial m;
    m.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::Solid);
    m.payload.alpha_mode = 15; // opaque billboards
    return m;
}

} // namespace

void QuakeScene::init_dynamic_meshes() {
    if (dynamic_node_id == merian::NODE_ID_INVALID) {
        merian::SceneNode root{};
        root.name = "dynamic";
        dynamic_node_id = add_node(std::move(root));
    }

    const auto& material_system = get_material_system();
    entity_material_id =
        material_system->add_material(quake_material_type_id, make_default_dynamic_material());
    sprite_material_id =
        material_system->add_material(quake_material_type_id, make_default_sprite_material());
    particle_material_id =
        material_system->add_material(quake_material_type_id, make_default_particle_material());

    auto add_dynamic = [&](const char* name, merian::MaterialID mid,
                           merian::GeometryFlags flags) -> merian::MeshID {
        auto mesh = std::make_unique<QuakeBrushMesh>();
        mesh->name = name;
        mesh->material_id = mid;
        mesh->flags = flags;
        seed_with_degenerate_triangle(*mesh);
        const merian::MeshID id = add_mesh(std::move(mesh));
        add_mesh_instance(id, dynamic_node_id);
        return id;
    };

    const auto dyn_flags =
        merian::GeometryFlags::IsDynamic | merian::GeometryFlags::FrontCounterClockwise;
    entity_mesh_id = add_dynamic("dynamic:entities", entity_material_id, dyn_flags);
    sprite_mesh_id = add_dynamic("dynamic:sprites", sprite_material_id, dyn_flags);
    particle_mesh_id = add_dynamic("dynamic:particles", particle_material_id, dyn_flags);
}

void QuakeScene::refresh_dynamic_meshes() {
    // The vector is private/const-accessor, but each unique_ptr::get() returns
    // a non-const Mesh* regardless — so we can mutate the held meshes safely.
    const auto& meshes = get_meshes();
    auto& entity_mesh = static_cast<QuakeBrushMesh&>(*meshes[entity_mesh_id]);
    auto& sprite_mesh = static_cast<QuakeBrushMesh&>(*meshes[sprite_mesh_id]);
    auto& particle_mesh = static_cast<QuakeBrushMesh&>(*meshes[particle_mesh_id]);

    entity_mesh.vertices.clear();
    entity_mesh.indices.clear();
    sprite_mesh.vertices.clear();
    sprite_mesh.indices.clear();
    particle_mesh.vertices.clear();
    particle_mesh.indices.clear();

    // Prev-frame world positions, parallel to each mesh's vertices vector.
    // Storage lives here until the Mesh subclass / Scene plumbing is wired
    // up to consume them as a motion-vector stream.
    std::vector<merian::float3> entity_prev_positions;
    std::vector<merian::float3> sprite_prev_positions;
    std::vector<merian::float3> particle_prev_positions;

    auto append_entity = [&](entity_t* ent) {
        if (ent == nullptr || ent->model == nullptr)
            return;
        if (ent->model->type == mod_sprite) {
            extract_sprite_geo(ent, sprite_mesh.vertices, sprite_prev_positions,
                               sprite_mesh.indices);
        } else {
            extract_entity_geo(ent, entity_mesh.vertices, entity_prev_positions,
                               entity_mesh.indices);
        }
    };

    // Player viewmodel (gun) and optional first-person body.
    if (playermodel == 1) {
        append_entity(&cl.viewent);
    } else if (playermodel == 2) {
        append_entity(&cl.viewent);
        if (cl.viewentity > 0 && cl.viewentity < cl_max_edicts && (cl_entities != nullptr))
            append_entity(&cl_entities[cl.viewentity]);
    }

    // Visible mobile entities and brush statics culled into cl_visedicts.
    for (int i = 0; i < cl_numvisedicts; i++) {
        append_entity(cl_visedicts[i]);
    }
    // Static (worldspawn-attached) entities — these are not added to
    // cl_visedicts and must be walked separately.
    for (int i = 0; i < cl.num_statics; i++) {
        append_entity(&cl_static_entities[i]);
    }

    extract_particle_geo(particle_mesh.vertices, particle_prev_positions, particle_mesh.indices,
                         reproducible_renders, prev_cl_time);
    prev_cl_time = cl.time;

    ensure_non_empty(entity_mesh);
    ensure_non_empty(sprite_mesh);
    ensure_non_empty(particle_mesh);

    mark_mesh_dirty(entity_mesh_id);
    mark_mesh_dirty(sprite_mesh_id);
    mark_mesh_dirty(particle_mesh_id);
}

void QuakeScene::cycle_animated_materials() {
    if (animated_brush_materials.empty())
        return;

    const auto& material_system = get_material_system();

    for (auto& entry : animated_brush_materials) {
        // Worldspawn brushes use frame=0; entity-driven alternate anims are not
        // applicable for static world geometry.
        const texture_t* current = R_TextureAnimation(entry.base_tex, 0);
        const merian::TextureID current_texnum =
            (current->gltexture != nullptr)
                ? static_cast<merian::TextureID>(current->gltexture->texnum)
                : entry.current_base_texnum;

        if (current_texnum == entry.current_base_texnum)
            continue;

        QuakeMaterial mat;
        mat.payload.base_tex = current_texnum;
        mat.payload.fullbright_tex = entry.fb_texnum;
        mat.payload.normal_tex = entry.normal_texnum;
        mat.payload.gloss_tex = entry.gloss_texnum;
        mat.payload.surface_flags = entry.surface_flags;
        mat.payload.alpha_mode = entry.alpha_mode;
        material_system->update_material(entry.material_id, mat);
        entry.current_base_texnum = current_texnum;
    }
}

void QuakeScene::run_startup_commands_if_needed() {
    if (startup_commands_dispatched || frame_counter <= 1 || startup_commands.empty())
        return;
    startup_commands_dispatched = true;
    merian::split(startup_commands, "\n", [&](const std::string& cmd) {
        if (!cmd.starts_with("#"))
            queue_command(cmd);
    });
}

void QuakeScene::properties(merian::Properties& config) {
    const bool old_overwrite_sun = overwrite_sun;
    const merian::float3 old_overwrite_sun_dir = overwrite_sun_dir;
    const merian::float3 old_overwrite_sun_col = overwrite_sun_col;

    config.st_separate("General");
    config.config_bool("gamestate update", update_gamestate);
    update_gamestate = update_gamestate || frame_counter == 0;

    std::string cmd;
    if (config.config_text("command", cmd, true)) {
        queue_command(cmd);
        if (!update_gamestate) {
            SPDLOG_WARN("command unpaused gamestate update");
            update_gamestate = true;
        }
    }
    bool changed = config.config_text_multiline(
        "startup commands", startup_commands, false,
        "multiple commands separated by newline, lines starting with # are ignored");
    if (changed && frame_counter == 0) {
        startup_commands_dispatched = false;
    }

    config.config_options("filtering", default_filtering, {"nearest", "linear"},
                          merian::Properties::OptionsStyle::COMBO,
                          "requires a level reload to show any effect.");

    config.st_separate("Reproducibility");
    config.config_int("stop after worldspawn", stop_after_worldspawn,
                      "Can be used for reference renders.");
    config.config_bool("rebuild after stop", rebuild_after_stop);
    config.config_bool("reproducible renders", reproducible_renders,
                       "e.g. disables random behavior");

    config.st_separate("Debug / Info");
    config.config_bool("overwrite sun", overwrite_sun);
    if (overwrite_sun) {
        config.config_vec("sun dir", overwrite_sun_dir);
        config.config_vec("sun col", overwrite_sun_col);
    }
    if (config.config_float("volume max t", volume_max_t)) {
        constant_data_dirty = true;
    }
    config.config_bool("overwrite mu_t/s", mu_t_s_overwrite);
    if (mu_t_s_overwrite) {
        config.config_float("mu_t", mu_t, "", 0.000001);
        config.config_vec("mu_s / mu_t", mu_s_div_mu_t);
    } else {
        const float fog_t = std::pow(Fog_GetDensity(), 2.F) * 0.1F;
        const float* fog_color = Fog_GetColor();
        config.output_text(fmt::format("mu_t: {}\nmu_s: ({}, {}, {})", fog_t,
                                       std::pow(fog_color[0], 1.F / 1.2F) * fog_t,
                                       std::pow(fog_color[1], 1.F / 1.2F) * fog_t,
                                       std::pow(fog_color[2], 1.F / 1.2F) * fog_t));
    }
    config.output_text(fmt::format("sun direction: ({}, {}, {})\nsun color: ({}, {}, {})",
                                   sun_direction.x, sun_direction.y, sun_direction.z, sun_color.r,
                                   sun_color.g, sun_color.b));
    config.output_text(fmt::format("view angles {} {} {}", r_refdef.viewangles[0],
                                   r_refdef.viewangles[1], r_refdef.viewangles[2]));
    config.output_text(fmt::format("server fps: {}", server_fps));
    config.config_options("player model", playermodel, {"none", "gun only", "full"});

    if (old_overwrite_sun != overwrite_sun || (old_overwrite_sun_dir != overwrite_sun_dir) ||
        (old_overwrite_sun_col != overwrite_sun_col)) {
        constant_data_dirty = true;
    }

    config.st_separate("Scene");

    Scene::properties(config);
}

} // namespace merian_quake
