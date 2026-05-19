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
#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

extern "C" {
#include "bgmusic.h"
#include "quakedef.h"
#include "screen.h"

extern cvar_t cl_maxpitch;
extern cvar_t cl_minpitch;
extern cvar_t scr_fov;
extern cvar_t cl_gun_fovscale;
extern qboolean scr_drawloading;
}

namespace merian_quake {

namespace {

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

    // ---------------
    // Scene

    const auto& tm = get_texture_manager();

    tm->resize(MAX_GLTEXTURES + 2);
    quake_material_type_id = material_system->register_material_type(
        QUAKE_MATERIAL_SLANG_TYPE_NAME, QUAKE_MATERIAL_SLANG_MODULE_PATH);

    auto cam = std::make_shared<merian::Camera>(merian::float3(1, 0, 0), merian::float3(0, 0, 0),
                                                get_up(), 90.F, 16.F / 9.F, 0.01F, 1e5f);
    quake_camera = add_camera(std::move(cam));

    // ---------------
    // Quake

    if (const auto audio_provider = context->find_provider<merian::AudioDeviceProvider>(true)) {
        g_quake_data.audio_device = audio_provider->create_audio_device();
    } else {
        g_quake_data.audio_device = nullptr;
    }
    host_parms = &g_quake_data.params;

    init_quakespasm(quakespasm_argc, quakespasm_argv);

    // Upload palette textures AFTER Quake init so d_8to24table is populated.
    tm->set_texture_from_rgba8(static_cast<merian::TextureID>(MAX_GLTEXTURES), d_8to24table, 256, 1,
                               vk::SamplerAddressMode::eClampToEdge, vk::Filter::eNearest,
                               vk::Filter::eNearest, true, false);
    // hack for rocket trails and explosions
    std::array<uint32_t, 256> fb_palette{};
    std::memcpy(fb_palette.data(), d_8to24table_fbright, sizeof(fb_palette));
    for (uint32_t i = 96; i <= 111; i++)
        fb_palette[i] = d_8to24table[i];
    tm->set_texture_from_rgba8(static_cast<merian::TextureID>(MAX_GLTEXTURES + 1),
                               fb_palette.data(), 256, 1, vk::SamplerAddressMode::eClampToEdge,
                               vk::Filter::eNearest, vk::Filter::eNearest, true, false);

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
                render_next = false;
                Host_Frame(g_quake_data.timediff);
                if (!render_next) {
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
    // unblock the game thread
    sync_render.push(0);
    sync_render.push(0);
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
    last_worldspawn_frame = frame;
}

void QuakeScene::cb_IN_Move(usercmd_t* cmd) {
    SPDLOG_TRACE("move");
    // pretty much a copy from in_sdl.c:

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
    render_next = true;
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

    // HACK: for blood patch
    if (strcmp(glt->name, "progs/gib_1.mdl:frame0") == 0)
        texnum_blood = glt->texnum;
    // HACK: for sparks and for emissive rocket particle trails
    if (strcmp(glt->name, "progs/s_exp_big.spr:frame10") == 0)
        texnum_explosion = glt->texnum;

    const bool linear =
        merian::ends_with(glt->name, "_norm") || merian::ends_with(glt->name, "_gloss");

    vk::Filter mag_filter;
    if (default_filtering == 0) {
        mag_filter =
            ((glt->flags & TEXPREF_LINEAR) != 0u) ? vk::Filter::eLinear : vk::Filter::eNearest;
    } else {
        mag_filter =
            ((glt->flags & TEXPREF_NEAREST) != 0u) ? vk::Filter::eNearest : vk::Filter::eLinear;
    }
    const bool generate_mipmaps = (glt->flags & TEXPREF_MIPMAP) != 0U;

    get_texture_manager()->set_texture_from_rgba8(static_cast<merian::TextureID>(glt->texnum), data,
                                                  glt->width, glt->height,
                                                  vk::SamplerAddressMode::eRepeat, mag_filter,
                                                  vk::Filter::eLinear, !linear, generate_mipmaps);
}

// per-frame --------------------------------------------------------------------

void QuakeScene::on_update(const merian::CommandBufferHandle& cmd,
                           const float /*time*/,
                           const float time_diff,
                           const uint32_t /*frame*/) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "QuakeScene::on_update");

    if (update_gamestate) {
        {
            MERIAN_PROFILE_SCOPE("game thread sync");

            sync_render.push(time_diff, 1);
            sync_gamestate.pop();
        }

        render_next = render_next && (scr_drawloading == 0);

        if ((cl.worldmodel != nullptr) && this->frame == last_worldspawn_frame) {
            MERIAN_PROFILE_SCOPE_GPU(cmd, "worldspawn");
            key_dest = key_game;
            m_state = m_none;
            sv_player = nullptr;

            if (world_meshes_built || particle_mesh_built) {
                MERIAN_PROFILE_SCOPE("teardown_world");
                teardown_world();
            }
            {
                MERIAN_PROFILE_SCOPE("rebuild_static_world");
                rebuild_static_world();
            }
            {
                MERIAN_PROFILE_SCOPE_GPU(cmd, "build_model_registries");
                build_model_registries(cmd);
            }
            world_meshes_built = true;
            {
                MERIAN_PROFILE_SCOPE("init_particle_batch");
                init_particle_batch();
            }
            particle_mesh_built = true;
            {
                MERIAN_PROFILE_SCOPE("update_sky");
                update_sky();
            }
        }

        if (!render_next) {
            this->frame++;
            return;
        }

        if (world_meshes_built && (cl.worldmodel != nullptr)) {
            MERIAN_PROFILE_SCOPE_GPU(cmd, "update_dynamic");
            update_dynamic(cmd);
        }

        if (world_meshes_built) {
            MERIAN_PROFILE_SCOPE("cycle_animated_materials");
            cycle_animated_materials();
        }

        {
            MERIAN_PROFILE_SCOPE("camera & sun");
            {
                const auto cam = get_camera(quake_camera);
                assert(cam);

                float fwd[3];
                float rgt[3];
                float up[3];
                AngleVectors(r_refdef.viewangles, fwd, rgt, up);

                const merian::float3 pos = merian::as_float3(r_refdef.vieworg);
                const merian::float3 fwd_v(fwd[0], fwd[1], fwd[2]);
                const merian::float3 up_v(up[0], up[1], up[2]);
                const float aspect = (resolution.height > 0)
                                         ? (static_cast<float>(resolution.width) /
                                            static_cast<float>(resolution.height))
                                         : (16.F / 9.F);
                cam->look_at(pos, pos + fwd_v, up_v, r_refdef.fov_x);
                cam->set_aspect_ratio(aspect);
            }

            if (overwrite_sun) {
                sun_color = overwrite_sun_col;
                sun_direction = overwrite_sun_dir;
            } else {
                sun_color = g_quake_data.current_sun_color;
                sun_direction = g_quake_data.current_sun_direction;
            }
            if (merian::length(sun_direction) > 0) {
                sun_direction = merian::normalize(sun_direction);
            }

            // if (!render_info.render) {
            //     render_info.uniform.sky.fill(notexture->texnum);
            // } else if (skybox_name[0] != 0) {
            //     for (int i = 0; i < 6; i++)
            //         render_info.uniform.sky[i] = skybox_textures[i]->texnum;
            // } else if (solidskytexture != nullptr) {
            //     render_info.uniform.sky[0] = solidskytexture->texnum;
            //     render_info.uniform.sky[1] = alphaskytexture->texnum;
            //     render_info.uniform.sky[2] = static_cast<uint16_t>(-1u);
            // }

            // if (mu_t_s_overwrite) {
            //     render_info.uniform.cam_x_mu_t.a = mu_t;
            //     render_info.uniform.prev_cam_x_mu_sx.a = mu_s_div_mu_t.r * mu_t;
            //     render_info.uniform.prev_cam_w_mu_sy.a = mu_s_div_mu_t.g * mu_t;
            //     render_info.uniform.prev_cam_u_mu_sz.a = mu_s_div_mu_t.b * mu_t;
            // } else {
            //     render_info.uniform.cam_x_mu_t.a = std::pow(Fog_GetDensity(), 2.f) * 0.1f;

            //     const float* fog_color = Fog_GetColor();
            //     render_info.uniform.prev_cam_x_mu_sx.a =
            //         std::pow(fog_color[0], 1.f / 1.2f) * render_info.uniform.cam_x_mu_t.a;
            //     render_info.uniform.prev_cam_w_mu_sy.a =
            //         std::pow(fog_color[1], 1.f / 1.2f) * render_info.uniform.cam_x_mu_t.a;
            //     render_info.uniform.prev_cam_u_mu_sz.a =
            //         std::pow(fog_color[2], 1.f / 1.2f) * render_info.uniform.cam_x_mu_t.a;
            // }
        }

        const bool in_game = update_gamestate && key_dest == key_game;
        controller->set_mouse_grabbed(in_game);
        if (input_listener)
            controller->add_listener(input_listener, in_game ? 100 : 0);
    }

    if (stop_after_worldspawn >= 0 &&
        (this->frame - last_worldspawn_frame) == static_cast<uint64_t>(stop_after_worldspawn)) {
        update_gamestate = false;
    }

    this->frame++;
}

namespace {

static std::mutex quake_cache_mutex;

// Map SURF_DRAW* bits (set by Quake's BSP loader from the texture name into
// surf->flags at gl_model.c:1351-1395) to QuakeSurfaceFlags values. No
// texture-name parsing — we trust Quake's already-parsed bits.
uint16_t convert_surf_flags(int surf_flags) {
    if ((surf_flags & SURF_DRAWSKY) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Sky);
    if ((surf_flags & SURF_DRAWLAVA) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Lava);
    if ((surf_flags & SURF_DRAWSLIME) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Slime);
    if ((surf_flags & SURF_DRAWTELE) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Tele);
    if ((surf_flags & SURF_DRAWWATER) != 0)
        return static_cast<uint16_t>(QuakeSurfaceFlags::Water);
    return static_cast<uint16_t>(QuakeSurfaceFlags::None);
}

QuakeMaterial make_brush_material_for(texture_t* tex, int surf_flags) {
    QuakeMaterial m;
    m.header.alpha_texture_id = (tex->gltexture != nullptr)
                                    ? static_cast<merian::TextureID>(tex->gltexture->texnum)
                                    : QUAKE_NO_TEXTURE;
    m.payload.fullbright_tex = (tex->fullbright != nullptr)
                                   ? static_cast<merian::TextureID>(tex->fullbright->texnum)
                                   : QUAKE_NO_TEXTURE;
    m.payload.normal_tex = (tex->norm != nullptr)
                               ? static_cast<merian::TextureID>(tex->norm->texnum)
                               : QUAKE_NO_TEXTURE;
    m.payload.gloss_tex = (tex->gloss != nullptr)
                              ? static_cast<merian::TextureID>(tex->gloss->texnum)
                              : QUAKE_NO_TEXTURE;
    m.payload.surface_flags = convert_surf_flags(surf_flags);
    const bool has_alpha =
        (tex->gltexture != nullptr) && ((tex->gltexture->flags & TEXPREF_ALPHA) != 0u);
    m.payload.alpha_mode = has_alpha ? 0u : 15u;
    // hack for ad_tears emissive waterfalls
    if ((tex->gltexture != nullptr) && (strstr(tex->gltexture->name, "wfall") != nullptr)) {
        m.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::Waterfall);
    }
    // Teleporters glow the full surface; if no fullbright was authored, use the base texture.
    if (m.payload.surface_flags == static_cast<uint16_t>(QuakeSurfaceFlags::Tele) &&
        m.payload.fullbright_tex == QUAKE_NO_TEXTURE) {
        m.payload.fullbright_tex = m.header.alpha_texture_id;
    }
    return m;
}

} // namespace

void QuakeScene::teardown_world() {
    for (const merian::MeshID id : world_mesh_ids)
        remove_mesh(id);
    world_mesh_ids.clear();

    // Alias and Brush slots own their per-entity mesh; Sprite slots only
    // instance a shared sprite-frame mesh which we drop below. remove_node
    // detaches all instances.
    for (auto& [_, slot] : entity_slots) {
        if (slot.kind == EntityKind::Alias || slot.kind == EntityKind::Brush) {
            for (const merian::MeshID id : slot.mesh_ids)
                remove_mesh(id);
        }
        if (slot.node_id != merian::NODE_ID_INVALID)
            remove_node(slot.node_id);
    }
    entity_slots.clear();
    previous_entity_slots.clear();

    for (const auto& [_, info] : sprite_frame_info) {
        if (info.mesh_id != merian::MeshID{})
            remove_mesh(info.mesh_id);
    }
    sprite_frame_info.clear();

    if (particle_mesh_built) {
        remove_mesh(particle_mesh_id);
        particle_mesh_id = merian::MeshID{};
    }
    if (particle_node_id != merian::NODE_ID_INVALID) {
        remove_node(particle_node_id);
        particle_node_id = merian::NODE_ID_INVALID;
    }

    if (world_node_id != merian::NODE_ID_INVALID) {
        remove_node(world_node_id);
        world_node_id = merian::NODE_ID_INVALID;
    }

    material_id_for_tex.clear();
    material_id_for_alias_skin.clear();
    animated_brush_materials.clear();

    for (auto& [_, info] : alias_model_info)
        defer_buffer_release(std::move(info.index_buffer));
    alias_model_info.clear();

    for (auto& [_, parts] : brush_submodel_geo) {
        for (auto& part : parts) {
            defer_buffer_release(std::move(part.vb));
            defer_buffer_release(std::move(part.ib));
        }
    }
    brush_submodel_geo.clear();

    get_material_system()->clear();

    world_meshes_built = false;
    particle_mesh_built = false;
}

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
        texture_t* tex = nullptr;
        int surf_flags = 0;
    };
    std::map<TexFlagsKey, BrushBucket> buckets;

    const auto& material_system = get_material_system();

    for (int i = 0; i < world->nummodelsurfaces; i++) {
        msurface_t* surf = &world->surfaces[world->firstmodelsurface + i];
        if (surf->texinfo == nullptr || surf->texinfo->texture == nullptr)
            continue;

        texture_t* base_tex = surf->texinfo->texture;
        if (strcmp(base_tex->name, "skip") == 0)
            continue;

        const TexFlagsKey key{base_tex, surf->flags & SURF_INTERESTING_BITS};
        auto& bucket = buckets[key];
        bucket.tex = base_tex;
        bucket.surf_flags = key.surf_flags;

        // Surface plane normal (already in world space; no model transform
        // for the worldspawn entity).
        merian::float3 plane_n = merian::as_float3(surf->plane->normal);
        if ((surf->flags & SURF_PLANEBACK) != 0)
            plane_n = -plane_n;
        const uint32_t enc_n = merian::encode_normal(merian::normalize(plane_n));

        for (glpoly_t* p = surf->polys; p != nullptr; p = nullptr) {
            const uint32_t base = static_cast<uint32_t>(bucket.vertices.size());
            for (int v = 0; v < p->numverts; v++) {
                merian::PackedVertexData pv{};
                pv.position = merian::as_float3(p->verts[v]);
                pv.encoded_normal = enc_n;
                pv.uv = merian::half2(p->verts[v][3], p->verts[v][4]);
                pv.encoded_tangent = 0;
                bucket.vertices.push_back(pv);
            }
            // Fan-triangulate p->numverts verts into (numverts-2) triangles.
            for (int v = 2; v < p->numverts; v++) {
                bucket.indices.push_back(merian::uint3(base, base + static_cast<uint32_t>(v) - 1u,
                                                       base + static_cast<uint32_t>(v)));
            }
        }
    }

    material_id_for_tex.clear();
    animated_brush_materials.clear();

    for (auto& [key, bucket] : buckets) {
        if (bucket.indices.empty())
            continue;

        const QuakeMaterial mat = make_brush_material_for(bucket.tex, bucket.surf_flags);
        const merian::MaterialID material_id =
            material_system->add_material(quake_material_type_id, mat);
        material_id_for_tex.emplace(key, material_id);

        auto mesh = std::make_unique<QuakeBrushMesh>();
        mesh->name =
            fmt::format("worldspawn:{}", bucket.tex->name[0] != 0 ? bucket.tex->name : "unnamed");
        mesh->material_id = material_id;
        mesh->flags = merian::MeshFlags::FrontCounterClockwise;
        if ((key.tex->gltexture != nullptr) &&
            ((key.tex->gltexture->flags & TEXPREF_ALPHA) == 0u)) {
            mesh->flags = mesh->flags | merian::MeshFlags::IsOpaque;
        }
        mesh->vertices = std::move(bucket.vertices);
        mesh->indices = std::move(bucket.indices);

        const merian::MeshID mesh_id = add_mesh(std::move(mesh));
        add_mesh_instance(mesh_id, world_node_id);
        world_mesh_ids.push_back(mesh_id);

        if (bucket.tex->anim_total > 0) {
            animated_brush_materials.push_back(AnimatedBrushMaterial{
                material_id, bucket.tex, mat.payload.fullbright_tex, mat.payload.normal_tex,
                mat.payload.gloss_tex, mat.payload.surface_flags, mat.payload.alpha_mode,
                mat.header.alpha_texture_id});
        }
    }

    SPDLOG_DEBUG("static world: {} brush partitions, {} surfaces, {} animated materials",
                 buckets.size(), world->nummodelsurfaces, animated_brush_materials.size());
}

void QuakeScene::update_sky() {
    const merian::float3 active_sun_dir =
        overwrite_sun ? overwrite_sun_dir : g_quake_data.current_sun_direction;
    const merian::float3 active_sun_color =
        overwrite_sun ? overwrite_sun_col : g_quake_data.current_sun_color;

    std::string source = "import shader.quake_sky;\n";
    source += "namespace merian {\n";
    source += fmt::format(
        "export static const float3 sun_dir = float3({:.6f}, {:.6f}, {:.6f});\n",
        active_sun_dir.x, active_sun_dir.y, active_sun_dir.z);
    source += fmt::format(
        "export static const float3 sun_color = float3({:.6f}, {:.6f}, {:.6f});\n",
        active_sun_color.r, active_sun_color.g, active_sun_color.b);

    if (skybox_name[0] != 0) {
        const auto t = [](int i) -> uint32_t {
            return skybox_textures[i] != nullptr ? skybox_textures[i]->texnum : 0u;
        };
        source += fmt::format("export static const QuakeSky sky = CubemapSky("
                              "TextureID({}), TextureID({}), TextureID({}), "
                              "TextureID({}), TextureID({}), TextureID({}));\n",
                              t(0), t(1), t(2), t(3), t(4), t(5));
    } else if (solidskytexture != nullptr) {
        const uint32_t solid = solidskytexture->texnum;
        const uint32_t alpha = alphaskytexture != nullptr ? alphaskytexture->texnum : 0u;
        source += fmt::format(
            "export static const QuakeSky sky = ClassicSky(TextureID({}), TextureID({}));\n",
            solid, alpha);
    } else {
        source += "export static const QuakeSky sky = BlackSky();\n";
    }
    source += "}\n";

    get_material_system()->get_composition()->add_module_from_string("merian_quake_scene_spec",
                                                                     source);
}

namespace {

void seed_with_degenerate_triangle(QuakeHostDynamicMesh& mesh) {
    mesh.vertices.assign(3, merian::PackedVertexData{});
    mesh.prev_vertices.assign(3, merian::PackedPrevVertexData{});
    if (mesh.has_indices()) {
        mesh.indices.assign(1, merian::uint3(0u, 1u, 2u));
    } else {
        mesh.indices.clear();
    }
}

void ensure_non_empty(QuakeHostDynamicMesh& mesh) {
    if (mesh.vertices.empty() || (mesh.has_indices() && mesh.indices.empty()))
        seed_with_degenerate_triangle(mesh);
}

merian::float4x4 entity_transform(entity_t* ent) {
    std::array<float, 3> a = {-ent->angles[0], ent->angles[1], ent->angles[2]};
    merian::float4x4 m = merian::identity();
    AngleVectors(a.data(), &m[0].x, &m[1].x, &m[2].x);
    m[1] *= -1;
    m[3] = merian::float4(ent->origin[0], ent->origin[1], ent->origin[2], 1.f);
    return merian::transpose(m);
}

QuakeMaterial make_alias_material(aliashdr_t* hdr, int skin, int fm = 0) {
    if (hdr->numskins <= 0)
        return {};
    skin = std::clamp(skin, 0, hdr->numskins - 1);
    fm &= 3;
    QuakeMaterial m;
    if (hdr->gltextures[skin][fm] != nullptr)
        m.header.alpha_texture_id =
            static_cast<merian::TextureID>(hdr->gltextures[skin][fm]->texnum);
    if (hdr->fbtextures[skin][fm] != nullptr)
        m.payload.fullbright_tex =
            static_cast<merian::TextureID>(hdr->fbtextures[skin][fm]->texnum);
    if (hdr->nmtextures[skin][fm] != nullptr)
        m.payload.normal_tex = static_cast<merian::TextureID>(hdr->nmtextures[skin][fm]->texnum);
    if (hdr->gstextures[skin][fm] != nullptr)
        m.payload.gloss_tex = static_cast<merian::TextureID>(hdr->gstextures[skin][fm]->texnum);
    m.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::None);
    m.payload.alpha_mode = 15;
    return m;
}

QuakeMaterial make_sprite_frame_material(mspriteframe_t* frame) {
    QuakeMaterial m;
    if (frame->gltexture != nullptr)
        m.header.alpha_texture_id = static_cast<merian::TextureID>(frame->gltexture->texnum);
    m.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::Sprite);
    m.payload.alpha_mode = 0;
    m.payload.fullbright_tex = m.header.alpha_texture_id;
    return m;
}

} // namespace

void QuakeScene::build_model_registries(const merian::CommandBufferHandle& cmd) {
    const auto& ms = get_material_system();
    const auto& alloc = get_allocator();
    const auto buf_usage = vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst |
                           vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                           vk::BufferUsageFlagBits::eShaderDeviceAddress;

    // model_precache[0] is the worldmodel, handled separately by
    // rebuild_static_world. Skip it.
    for (int i = 1; i < MAX_MODELS; i++) {
        qmodel_t* mod = cl.model_precache[i];
        if (mod == nullptr)
            break;

        if (mod->type == mod_alias) {
            auto* hdr = (aliashdr_t*)Mod_Extradata(mod);
            if (hdr == nullptr)
                continue;

            const auto* indexes = (const int16_t*)((uint8_t*)hdr + hdr->indexes);
            const auto* desc = (const aliasmesh_t*)((uint8_t*)hdr + hdr->meshdesc);
            const auto* trivertexes = (const trivertx_t*)((uint8_t*)hdr + hdr->vertexes);
            const uint32_t prim_count = static_cast<uint32_t>(hdr->numindexes / 3);
            const uint32_t vert_count = static_cast<uint32_t>(hdr->numverts_vbo);
            const uint32_t numverts = static_cast<uint32_t>(hdr->numverts);
            const uint32_t numposes = static_cast<uint32_t>(hdr->numposes);

            // Quake mdl indices are int16 with non-negative values: same bit pattern as uint16.
            merian::BufferHandle ib = alloc->create_buffer(
                cmd, sizeof(int16_t) * prim_count * 3, buf_usage, indexes,
                merian::MemoryMappingType::NONE, fmt::format("alias_ib:{}", mod->name));

            // Bake per-pose smooth normals (one float3 per (pose, original
            // vertex)). For Quake's CW-wound triangles, the outward face
            // normal is cross(v2-v0, v1-v0); we accumulate at adjacent
            // vertices and normalize. Positions stay in raw byte-coord space:
            // the per-axis scale cancels at normalize-time after the inverse-
            // transposed instance transform is applied in the shader.
            std::vector<merian::float3> baked_normals(
                static_cast<size_t>(numposes) * numverts, merian::float3(0.f));
            for (uint32_t pose = 0; pose < numposes; pose++) {
                merian::float3* pose_normals = baked_normals.data() + pose * numverts;
                const trivertx_t* pose_verts = trivertexes + pose * numverts;
                for (uint32_t t = 0; t < prim_count; t++) {
                    const int vi0 = desc[indexes[t * 3 + 0]].vertindex;
                    const int vi1 = desc[indexes[t * 3 + 1]].vertindex;
                    const int vi2 = desc[indexes[t * 3 + 2]].vertindex;
                    const merian::float3 p0(pose_verts[vi0].v[0], pose_verts[vi0].v[1],
                                            pose_verts[vi0].v[2]);
                    const merian::float3 p1(pose_verts[vi1].v[0], pose_verts[vi1].v[1],
                                            pose_verts[vi1].v[2]);
                    const merian::float3 p2(pose_verts[vi2].v[0], pose_verts[vi2].v[1],
                                            pose_verts[vi2].v[2]);
                    const merian::float3 face_n = merian::cross(p2 - p0, p1 - p0);
                    pose_normals[vi0] += face_n;
                    pose_normals[vi1] += face_n;
                    pose_normals[vi2] += face_n;
                }
                for (uint32_t v = 0; v < numverts; v++) {
                    const float len2 = merian::dot(pose_normals[v], pose_normals[v]);
                    pose_normals[v] = (len2 > 0.f) ? pose_normals[v] / std::sqrt(len2)
                                                   : merian::float3(0.f, 0.f, 1.f);
                }
            }

            alias_model_info[mod] = AliasModelInfo{std::move(ib),
                                                   vert_count,
                                                   prim_count,
                                                   hdr->numskins,
                                                   std::move(baked_normals)};

            for (int s = 0; s < hdr->numskins; s++) {
                const QuakeMaterial mat = make_alias_material(hdr, s);
                material_id_for_alias_skin[{mod, s}] =
                    ms->add_material(quake_material_type_id, mat);
            }
        } else if (mod->type == mod_sprite) {
            auto* spr = (msprite_t*)mod->cache.data;
            if (spr == nullptr)
                continue;

            auto register_frame = [&](mspriteframe_t* frame, int debug_idx) {
                if (frame == nullptr || sprite_frame_info.contains(frame))
                    return;
                const QuakeMaterial mat = make_sprite_frame_material(frame);
                const merian::MaterialID material_id =
                    ms->add_material(quake_material_type_id, mat);

                auto sprite_mesh = std::make_unique<QuakeSpriteFrameMesh>();
                sprite_mesh->name = fmt::format("sprite:{}:{}", mod->name, debug_idx);
                sprite_mesh->material_id = material_id;
                sprite_mesh->flags = merian::MeshFlags::TwoSided;

                const uint32_t enc_n = merian::encode_normal(merian::float3(1, 0, 0));
                const float smax = frame->smax;
                const float tmax = frame->tmax;
                auto push = [&](float y, float z, float u, float v) {
                    merian::PackedVertexData pv{};
                    pv.position = merian::float3(0.f, y, z);
                    pv.encoded_normal = enc_n;
                    pv.uv = merian::half2(u, v);
                    pv.encoded_tangent = 0;
                    sprite_mesh->vertices.push_back(pv);
                };
                push(frame->left, frame->down, 0.f, tmax);
                push(frame->left, frame->up, 0.f, 0.f);
                push(frame->right, frame->up, smax, 0.f);
                push(frame->left, frame->down, 0.f, tmax);
                push(frame->right, frame->up, smax, 0.f);
                push(frame->right, frame->down, smax, tmax);

                const merian::MeshID mesh_id = add_mesh(std::move(sprite_mesh));
                sprite_frame_info[frame] = SpriteFrameInfo{mesh_id, material_id};
            };

            for (int f = 0; f < spr->numframes; f++) {
                if (spr->frames[f].type == SPR_SINGLE) {
                    register_frame(spr->frames[f].frameptr, f);
                } else {
                    auto* group = (mspritegroup_t*)spr->frames[f].frameptr;
                    if (group == nullptr)
                        continue;
                    for (int g = 0; g < group->numframes; g++)
                        register_frame(group->frames[g], (f << 8) | g);
                }
            }
        }
        // mod_brush submodels keep their lazy build path inside
        // ensure_brush_slot — sharing isn't enabled here.
    }
}

void QuakeScene::init_particle_batch() {
    QuakeMaterial particle_mat;
    particle_mat.header.alpha_texture_id = static_cast<merian::TextureID>(MAX_GLTEXTURES);
    particle_mat.payload.fullbright_tex = static_cast<merian::TextureID>(MAX_GLTEXTURES + 1);
    particle_mat.payload.surface_flags = static_cast<uint16_t>(QuakeSurfaceFlags::Solid);
    particle_mat.payload.alpha_mode = 15;
    particle_material_id =
        get_material_system()->add_material(quake_material_type_id, particle_mat);

    merian::SceneNode node;
    node.name = "particles";
    particle_node_id = add_node(std::move(node));

    auto mesh = std::make_unique<QuakeHostDynamicMesh>();
    mesh->name = "particles";
    mesh->material_id = particle_material_id;
    mesh->flags = merian::MeshFlags::IsMorphed | merian::MeshFlags::HasVariableTopology |
                  merian::MeshFlags::FrontCounterClockwise;
    seed_with_degenerate_triangle(*mesh);
    particle_mesh_id = add_mesh(std::move(mesh));
    add_mesh_instance(particle_mesh_id, particle_node_id);
}

// -----------------------------------------------------------------------
// Per-entity slot management
// -----------------------------------------------------------------------

void QuakeScene::destroy_slot(EntityMeshSlot& slot) {
    if (slot.kind == EntityKind::Alias) {
        for (const merian::MeshID id : slot.mesh_ids)
            remove_mesh(id);
    } else if (slot.kind == EntityKind::Sprite) {
        if (!slot.mesh_ids.empty() && slot.node_id != merian::NODE_ID_INVALID)
            remove_mesh_instance(slot.mesh_ids[0], slot.node_id);
    }
    if (slot.node_id != merian::NODE_ID_INVALID)
        remove_node(slot.node_id);
}

QuakeScene::EntityMeshSlot QuakeScene::build_alias_slot(entity_t* ent) {
    auto info_it = alias_model_info.find(ent->model);
    if (info_it == alias_model_info.end() || info_it->second.numskins <= 0)
        return {};

    const AliasModelInfo& info = info_it->second;
    const auto& alloc = get_allocator();
    const vk::DeviceSize vb_size = info.vertex_count * sizeof(merian::PackedVertexData);
    const vk::DeviceSize prev_vb_size = info.vertex_count * sizeof(merian::PackedPrevVertexData);
    const auto staging_usage = vk::BufferUsageFlagBits::eTransferSrc |
                               vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress;

    auto vb = alloc->create_buffer(vb_size, staging_usage,
                                   merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                   fmt::format("alias_vb:{}", ent->model->name));
    auto prev_vb = alloc->create_buffer(prev_vb_size, staging_usage,
                                        merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                        fmt::format("alias_prev_vb:{}", ent->model->name));

    const int skin = std::clamp(ent->skinnum, 0, info.numskins - 1);
    auto mat_it = material_id_for_alias_skin.find({ent->model, skin});
    const merian::MaterialID material_id =
        (mat_it != material_id_for_alias_skin.end()) ? mat_it->second : merian::MaterialID{};

    merian::float4x4 scale_col;
    {
        std::lock_guard<std::mutex> lock(quake_cache_mutex);
        const auto* hdr = (aliashdr_t*)Mod_Extradata(ent->model);
        scale_col = merian::mul(merian::translation(merian::as_float3(hdr->scale_origin)),
                                merian::scale(merian::as_float3(hdr->scale)));
    }

    merian::SceneNode node;
    node.name = fmt::format("alias:{}", ent->model->name);
    node.is_animated = true;
    node.local_transform = merian::mul(entity_transform(ent), scale_col);
    const merian::NodeID nid = add_node(std::move(node));

    auto mesh = std::make_unique<AliasInstanceMesh>();
    mesh->name = fmt::format("alias:{}", ent->model->name);
    mesh->material_id = material_id;
    mesh->flags = merian::MeshFlags::IsMorphed | merian::MeshFlags::FrontCounterClockwise;
    mesh->vb_staging = std::move(vb);
    mesh->prev_vb_staging = std::move(prev_vb);
    mesh->vb_mapped = mesh->vb_staging->get_memory()->map_as<merian::PackedVertexData>();
    mesh->prev_vb_mapped =
        mesh->prev_vb_staging->get_memory()->map_as<merian::PackedPrevVertexData>();
    mesh->ib_shared = info.index_buffer;
    mesh->vertex_count = info.vertex_count;
    mesh->primitive_count = info.primitive_count;

    const merian::MeshID mesh_id = add_mesh(std::move(mesh));
    add_mesh_instance(mesh_id, nid);

    EntityMeshSlot slot;
    slot.node_id = nid;
    slot.mesh_ids = {mesh_id};
    slot.model = ent->model;
    slot.kind = EntityKind::Alias;
    slot.cached_skinnum = ent->skinnum;
    return slot;
}

QuakeScene::EntityMeshSlot QuakeScene::build_brush_slot(entity_t* ent,
                                                        const merian::CommandBufferHandle& cmd) {
    // Lazily build submodel geometry on first reference.
    auto geo_it = brush_submodel_geo.find(ent->model);
    if (geo_it == brush_submodel_geo.end()) {
        qmodel_t* mod = ent->model;
        std::unordered_map<TexFlagsKey,
                           std::pair<std::vector<merian::PackedVertexData>,
                                     std::vector<merian::uint3>>,
                           TexFlagsKeyHash>
            parts;

        for (int i = 0; i < mod->nummodelsurfaces; i++) {
            msurface_t* surf = &mod->surfaces[mod->firstmodelsurface + i];
            if (surf->texinfo == nullptr || surf->texinfo->texture == nullptr)
                continue;
            texture_t* tex = surf->texinfo->texture;
            const TexFlagsKey key{tex, surf->flags & SURF_INTERESTING_BITS};
            auto& [verts, idxs] = parts[key];

            merian::float3 plane_n = merian::as_float3(surf->plane->normal);
            if ((surf->flags & SURF_PLANEBACK) != 0)
                plane_n = -plane_n;
            const uint32_t enc_n = merian::encode_normal(merian::normalize(plane_n));

            for (glpoly_t* p = surf->polys; p != nullptr; p = nullptr) {
                const uint32_t base = static_cast<uint32_t>(verts.size());
                for (int v = 0; v < p->numverts; v++) {
                    merian::PackedVertexData pv{};
                    pv.position = merian::as_float3(p->verts[v]);
                    pv.encoded_normal = enc_n;
                    pv.uv = merian::half2(p->verts[v][3], p->verts[v][4]);
                    pv.encoded_tangent = 0;
                    verts.push_back(pv);
                }
                for (int v = 2; v < p->numverts; v++) {
                    idxs.push_back(merian::uint3(base, base + static_cast<uint32_t>(v) - 1u,
                                                 base + static_cast<uint32_t>(v)));
                }
            }
        }

        const auto& alloc = get_allocator();
        const auto& ms = get_material_system();
        const auto buf_usage =
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress;

        auto& geo_parts = brush_submodel_geo[mod];
        for (auto& [key, data] : parts) {
            auto& [verts, idxs] = data;
            if (idxs.empty())
                continue;

            auto mat_it = material_id_for_tex.find(key);
            merian::MaterialID material_id;
            if (mat_it != material_id_for_tex.end()) {
                material_id = mat_it->second;
            } else {
                const QuakeMaterial mat = make_brush_material_for(key.tex, key.surf_flags);
                material_id = ms->add_material(quake_material_type_id, mat);
                material_id_for_tex.emplace(key, material_id);
                if (key.tex->anim_total > 0) {
                    animated_brush_materials.push_back(AnimatedBrushMaterial{
                        material_id, key.tex, mat.payload.fullbright_tex,
                        mat.payload.normal_tex, mat.payload.gloss_tex,
                        mat.payload.surface_flags, mat.payload.alpha_mode,
                        mat.header.alpha_texture_id});
                }
            }

            auto vb =
                alloc->create_buffer(cmd, verts, buf_usage, fmt::format("brush_vb:{}", mod->name));
            auto ib =
                alloc->create_buffer(cmd, idxs, buf_usage, fmt::format("brush_ib:{}", mod->name));

            const bool has_alpha =
                key.tex->gltexture != nullptr && (key.tex->gltexture->flags & TEXPREF_ALPHA) != 0;
            geo_parts.push_back(BrushSubmodelGeoPart{
                std::move(vb), std::move(ib), static_cast<uint32_t>(verts.size()),
                static_cast<uint32_t>(idxs.size()), material_id, has_alpha});
        }
        geo_it = brush_submodel_geo.find(mod);
    }

    if (geo_it == brush_submodel_geo.end() || geo_it->second.empty())
        return {};

    merian::SceneNode node;
    node.name = fmt::format("brush:{}", ent->model->name);
    node.is_animated = true;
    node.local_transform = entity_transform(ent);
    const merian::NodeID nid = add_node(std::move(node));

    EntityMeshSlot slot;
    slot.node_id = nid;
    slot.model = ent->model;
    slot.kind = EntityKind::Brush;

    for (const auto& part : geo_it->second) {
        auto mesh = std::make_unique<BrushEntityMesh>();
        mesh->name = fmt::format("brush:{}:{}", ent->model->name, part.material_id);
        mesh->material_id = part.material_id;
        mesh->flags = merian::MeshFlags::FrontCounterClockwise;
        if (!part.has_alpha)
            mesh->flags = mesh->flags | merian::MeshFlags::IsOpaque;
        mesh->vb = part.vb;
        mesh->ib = part.ib;
        mesh->vertex_count = part.vertex_count;
        mesh->primitive_count = part.primitive_count;

        const merian::MeshID mesh_id = add_mesh(std::move(mesh));
        add_mesh_instance(mesh_id, nid);
        slot.mesh_ids.push_back(mesh_id);
    }

    return slot;
}

QuakeScene::EntityMeshSlot QuakeScene::build_sprite_slot(entity_t* ent) {
    mspriteframe_t* frame = R_GetSpriteFrame(ent);
    auto info_it = sprite_frame_info.find(frame);
    if (info_it == sprite_frame_info.end())
        return {};

    merian::SceneNode node;
    node.name = fmt::format("sprite:{}", ent->model->name);
    node.is_animated = true;
    const merian::NodeID nid = add_node(std::move(node));

    add_mesh_instance(info_it->second.mesh_id, nid);

    EntityMeshSlot slot;
    slot.node_id = nid;
    slot.mesh_ids = {info_it->second.mesh_id};
    slot.model = ent->model;
    slot.kind = EntityKind::Sprite;
    slot.cached_sprite_frame = frame;
    return slot;
}

QuakeScene::EntityMeshSlot* QuakeScene::acquire_slot(entity_t* ent,
                                                    const merian::CommandBufferHandle& cmd) {
    EntityKind kind;
    switch (ent->model->type) {
    case mod_alias:  kind = EntityKind::Alias;  break;
    case mod_brush:  kind = EntityKind::Brush;  break;
    case mod_sprite: kind = EntityKind::Sprite; break;
    default: return nullptr;
    }

    // Migrate intact when previous frame's slot still matches this entity.
    auto node = previous_entity_slots.extract(ent);
    if (!node.empty()) {
        if (node.mapped().model == ent->model && node.mapped().kind == kind) {
            auto [it, _, __] = entity_slots.insert(std::move(node));
            return &it->second;
        }
        destroy_slot(node.mapped());
    }

    EntityMeshSlot fresh;
    switch (kind) {
    case EntityKind::Alias:  fresh = build_alias_slot(ent);       break;
    case EntityKind::Brush:  fresh = build_brush_slot(ent, cmd);  break;
    case EntityKind::Sprite: fresh = build_sprite_slot(ent);      break;
    }
    if (fresh.node_id == merian::NODE_ID_INVALID)
        return nullptr;

    switch (kind) {
    case EntityKind::Alias:  current_entity_stats.alias.newly_created++;  break;
    case EntityKind::Brush:  current_entity_stats.brush.newly_created++;  break;
    case EntityKind::Sprite: current_entity_stats.sprite.newly_created++; break;
    }

    auto [it, _] = entity_slots.emplace(ent, std::move(fresh));
    return &it->second;
}

void QuakeScene::refresh_alias(QuakeScene::EntityMeshSlot& slot, entity_t* ent) {
    if (slot.mesh_ids.empty())
        return;

    auto& mesh = static_cast<AliasInstanceMesh&>(*get_mesh_infos()[slot.mesh_ids[0]].mesh);

    std::lock_guard<std::mutex> lock(quake_cache_mutex);

    auto* hdr = (aliashdr_t*)Mod_Extradata(ent->model);

    if (hdr->numskins > 0) {
        const int skin = std::clamp(ent->skinnum, 0, hdr->numskins - 1);
        const int fm = ((int)(cl.time * 10)) & 3;
        const auto* skin_tex = hdr->gltextures[skin][fm];
        const merian::TextureID current_texnum =
            skin_tex ? static_cast<merian::TextureID>(skin_tex->texnum) : merian::TextureID{};

        if (ent->skinnum != slot.cached_skinnum) {
            auto mat_it = material_id_for_alias_skin.find({ent->model, skin});
            if (mat_it != material_id_for_alias_skin.end())
                mesh.material_id = mat_it->second;
            slot.cached_skinnum = ent->skinnum;
        }

        if (current_texnum != slot.cached_skin_texnum && mesh.material_id != merian::MaterialID{}) {
            get_material_system()->update_material(mesh.material_id,
                                                   make_alias_material(hdr, skin, fm));
            slot.cached_skin_texnum = current_texnum;
        }
    }

    lerpdata_t lerpdata;
    R_SetupAliasFrame(ent, hdr, ent->frame, &lerpdata);
    R_SetupEntityTransform(ent, &lerpdata);

    const int prev_pose1 = (slot.cached_pose1 >= 0) ? slot.cached_pose1 : lerpdata.pose1;
    const int prev_pose2 = (slot.cached_pose2 >= 0) ? slot.cached_pose2 : lerpdata.pose2;
    const float prev_blend = (slot.cached_blend >= 0.f) ? slot.cached_blend : lerpdata.blend;

    const bool pose_changed =
        lerpdata.pose1 != slot.cached_pose1 || lerpdata.pose2 != slot.cached_pose2 ||
        lerpdata.blend != slot.cached_blend || prev_pose1 != slot.cached_prev_pose1 ||
        prev_pose2 != slot.cached_prev_pose2 || prev_blend != slot.cached_prev_blend;

    if (pose_changed) {
        const auto info_it = alias_model_info.find(ent->model);
        const merian::float3* baked_normals =
            (info_it != alias_model_info.end()) ? info_it->second.baked_normals.data() : nullptr;
        lerp_alias_vertices(hdr, baked_normals, lerpdata.pose1, lerpdata.pose2, lerpdata.blend,
                            prev_pose1, prev_pose2, prev_blend, mesh.vb_mapped,
                            mesh.prev_vb_mapped);

        slot.cached_pose1 = lerpdata.pose1;
        slot.cached_pose2 = lerpdata.pose2;
        slot.cached_blend = lerpdata.blend;
        slot.cached_prev_pose1 = prev_pose1;
        slot.cached_prev_pose2 = prev_pose2;
        slot.cached_prev_blend = prev_blend;

        get_mesh_infos()[slot.mesh_ids[0]].mesh->vertices_dirty = true;
    }

    const merian::float3 hdr_scale = merian::as_float3(hdr->scale);
    const merian::float3 hdr_scale_origin = merian::as_float3(hdr->scale_origin);

    const bool transform_changed = !VectorCompare(lerpdata.origin, slot.cached_origin) ||
                                   !VectorCompare(lerpdata.angles, slot.cached_angles);

    if (transform_changed) {
        merian::float3 fovscale(1.f);
        if (ent == &cl.viewent && scr_fov.value > 90.f && cl_gun_fovscale.value != 0.f) {
            const float t = std::tan(scr_fov.value * static_cast<float>(0.5 * M_PI / 180.0));
            fovscale.y = t;
            fovscale.z = t;
        }

        const merian::float4x4 scale_col = merian::mul(
            merian::translation(hdr_scale_origin * fovscale), merian::scale(hdr_scale * fovscale));

        std::array<float, 3> a = {-lerpdata.angles[0], lerpdata.angles[1], lerpdata.angles[2]};
        merian::float4x4 rt = merian::identity();
        AngleVectors(a.data(), &rt[0].x, &rt[1].x, &rt[2].x);
        rt[1] *= -1;
        rt[3] = merian::float4(merian::as_float3(lerpdata.origin), 1.f);

        update_node(slot.node_id, merian::mul(merian::transpose(rt), scale_col));

        VectorCopy(lerpdata.origin, slot.cached_origin);
        VectorCopy(lerpdata.angles, slot.cached_angles);
    }
}

void QuakeScene::refresh_brush(EntityMeshSlot& slot, entity_t* ent) {
    update_node(slot.node_id, entity_transform(ent));
}

void QuakeScene::refresh_sprite(EntityMeshSlot& slot, entity_t* ent) {
    mspriteframe_t* sprite_frame = R_GetSpriteFrame(ent);
    if (sprite_frame != slot.cached_sprite_frame) {
        auto info_it = sprite_frame_info.find(sprite_frame);
        if (info_it != sprite_frame_info.end()) {
            remove_mesh_instance(slot.mesh_ids[0], slot.node_id);
            add_mesh_instance(info_it->second.mesh_id, slot.node_id);
            slot.mesh_ids[0] = info_it->second.mesh_id;
            slot.cached_sprite_frame = sprite_frame;
        }
    }

    auto* psprite = (msprite_t*)ent->model->cache.data;
    merian::float3 s_up;
    merian::float3 s_right;
    if (!sprite_world_basis(ent, psprite, s_up, s_right))
        return;

    const float scale = ENTSCALE_DECODE(ent->scale);
    const merian::float3 origin = merian::as_float3(ent->origin);
    // Map local (1,0,0)/(0,1,0)/(0,0,1) -> (cross(s_right, s_up), s_right, s_up).
    // Local quads live in the y-z plane so column 0 only needs to be a
    // sane basis vector for determinant sign.
    const merian::float3 n = merian::normalize(merian::cross(s_right, s_up));
    merian::float4x4 m = merian::identity();
    m[0] = merian::float4(n * scale, 0.f);
    m[1] = merian::float4(s_right * scale, 0.f);
    m[2] = merian::float4(s_up * scale, 0.f);
    m[3] = merian::float4(origin, 1.f);
    update_node(slot.node_id, merian::transpose(m));

    VectorCopy(ent->origin, ent->mv_prev_origin);
}

void QuakeScene::process_entity(entity_t* ent, const merian::CommandBufferHandle& cmd) {
    if (ent == nullptr || ent->model == nullptr)
        return;

    EntityMeshSlot* slot = acquire_slot(ent, cmd);
    if (slot == nullptr)
        return;

    switch (slot->kind) {
    case EntityKind::Alias:  refresh_alias(*slot, ent);  break;
    case EntityKind::Brush:  refresh_brush(*slot, ent);  break;
    case EntityKind::Sprite: refresh_sprite(*slot, ent); break;
    }
}

void QuakeScene::update_dynamic(const merian::CommandBufferHandle& cmd) {
    // Swap: every slot from last frame starts in previous_entity_slots.
    // acquire_slot migrates each one back as its entity is visited. Anything
    // left in previous_entity_slots after the visit pass belonged to entities
    // that vanished this frame (server removed them, culling dropped them,
    // or — most importantly for lightning beams — a temp-entity slot in
    // cl_temp_entities was recycled without nulling ent->model when the
    // beam expired). Those slots get destroyed in one sweep at the end.
    previous_entity_slots = std::move(entity_slots);
    entity_slots.clear();
    current_entity_stats = {};

    if (playermodel == 1) {
        process_entity(&cl.viewent, cmd);
    } else if (playermodel == 2) {
        process_entity(&cl.viewent, cmd);
        if (cl.viewentity > 0 && cl.viewentity < cl_max_edicts && (cl_entities != nullptr))
            process_entity(&cl_entities[cl.viewentity], cmd);
    }

    for (int i = 0; i < cl_numvisedicts; i++)
        process_entity(cl_visedicts[i], cmd);

    for (int i = 0; i < cl.num_statics; i++)
        process_entity(&cl_static_entities[i], cmd);

    for (auto& [_, slot] : previous_entity_slots)
        destroy_slot(slot);
    previous_entity_slots.clear();

    // Particle batch: one shared mesh, fully re-extracted every frame.
    {
        auto& mesh = static_cast<QuakeHostDynamicMesh&>(*get_mesh_infos()[particle_mesh_id].mesh);
        mesh.vertices.clear();
        mesh.prev_vertices.clear();
        mesh.indices.clear();

        std::vector<merian::float3> prev_pos;
        extract_particle_geo(mesh.vertices, prev_pos, mesh.indices, reproducible_renders,
                             prev_cl_time);
        prev_cl_time = cl.time;

        mesh.prev_vertices.resize(prev_pos.size());
        for (size_t i = 0; i < prev_pos.size(); ++i)
            mesh.prev_vertices[i].position = prev_pos[i];

        ensure_non_empty(mesh);
        mesh.vertices_dirty = true;
        mesh.indices_dirty = true;
    }

    for (const auto& [_, slot] : entity_slots) {
        switch (slot.kind) {
        case EntityKind::Alias:  current_entity_stats.alias.active++;  break;
        case EntityKind::Brush:  current_entity_stats.brush.active++;  break;
        case EntityKind::Sprite: current_entity_stats.sprite.active++; break;
        }
    }
    last_frame_entity_stats = current_entity_stats;
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
        mat.header.alpha_texture_id = current_texnum;
        mat.payload.fullbright_tex = entry.fb_texnum;
        mat.payload.normal_tex = entry.normal_texnum;
        mat.payload.gloss_tex = entry.gloss_texnum;
        mat.payload.surface_flags = entry.surface_flags;
        mat.payload.alpha_mode = entry.alpha_mode;
        material_system->update_material(entry.material_id, mat);
        entry.current_base_texnum = current_texnum;
    }
}

void QuakeScene::properties(merian::Properties& config) {
    config.st_separate("General");
    config.config_bool("gamestate update", update_gamestate);
    update_gamestate = update_gamestate || frame == 0;

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
    if (changed && frame == 0) {
        merian::split(startup_commands, "\n", [&](const std::string& cmd) {
            if (!cmd.starts_with("#"))
                queue_command(cmd);
        });
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
    config.config_float("volume max t", volume_max_t);
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

    config.st_separate("Entity counts");
    const auto& s = last_frame_entity_stats;
    config.output_text(fmt::format(
        "alias:  {} active (+{} new)\n"
        "brush:  {} active (+{} new)\n"
        "sprite: {} active (+{} new)\n"
        "sprite frames: {}\n"
        "brush submodels: {}",
        s.alias.active, s.alias.newly_created,
        s.brush.active, s.brush.newly_created,
        s.sprite.active, s.sprite.newly_created,
        sprite_frame_info.size(), brush_submodel_geo.size()));

    config.st_separate("Scene");

    Scene::properties(config);
}

} // namespace merian_quake
