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
    tm->set_texture_from_rgba8(static_cast<merian::TextureID>(MAX_GLTEXTURES + 1),
                               d_8to24table_fbright, 256, 1, vk::SamplerAddressMode::eClampToEdge,
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
                           const uint32_t frame) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "QuakeScene::on_update");

    this->frame = frame;

    if (update_gamestate) {
        {
            MERIAN_PROFILE_SCOPE("game thread sync");

            sync_render.push(time_diff, 1);
            sync_gamestate.pop();
        }

        render_next = render_next && (scr_drawloading == 0);

        if ((cl.worldmodel != nullptr) && frame == last_worldspawn_frame) {
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
        }

        if (world_meshes_built && (cl.worldmodel != nullptr)) {
            MERIAN_PROFILE_SCOPE_GPU(cmd, "refresh_entities");
            refresh_entities(cmd);
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
                const float aspect = (resolution.height > 0)
                                         ? (static_cast<float>(resolution.width) /
                                            static_cast<float>(resolution.height))
                                         : (16.F / 9.F);
                cam->look_at(pos, pos + fwd_v, get_up(), r_refdef.fov_x);
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
        (frame - last_worldspawn_frame) == static_cast<uint64_t>(stop_after_worldspawn)) {
        update_gamestate = false;
    }
}

namespace {

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
    return m;
}

} // namespace

void QuakeScene::teardown_world() {
    for (const merian::MeshID id : world_mesh_ids)
        remove_mesh(id);
    world_mesh_ids.clear();

    for (auto& [_, slot] : entity_slots) {
        for (const merian::MeshID id : slot.mesh_ids)
            remove_mesh(id);
        if (slot.node_id != merian::NODE_ID_INVALID)
            remove_node(slot.node_id);
    }
    entity_slots.clear();

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
    material_id_for_sprite_frame.clear();
    animated_brush_materials.clear();
    alias_model_info.clear();
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
            const uint32_t base_vertex = static_cast<uint32_t>(bucket.vertices.size());
            for (int v = 0; v < p->numverts; v++) {
                merian::PackedVertexData pv{};
                pv.position = merian::as_float3(p->verts[v]);
                pv.encoded_normal = enc_n;
                pv.uv = merian::half2(p->verts[v][3], p->verts[v][4]);
                pv.encoded_tangent = 0;
                bucket.vertices.push_back(pv);
            }
            for (int v = 2; v < p->numverts; v++) {
                bucket.indices.push_back(merian::uint3(base_vertex, base_vertex + uint32_t(v) - 1u,
                                                       base_vertex + uint32_t(v)));
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
        // Static, opaque, CCW-front (Quake convention; alpha-test on world
        // brushes lands in a follow-up gbuffer pass).
        mesh->flags = merian::MeshFlags::IsOpaque | merian::MeshFlags::FrontCounterClockwise;
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

namespace {

void seed_with_degenerate_triangle(QuakeHostDynamicMesh& mesh) {
    mesh.vertices.assign(3, merian::PackedVertexData{});
    mesh.prev_vertices.assign(3, merian::PackedPrevVertexData{});
    mesh.indices.assign(1, merian::uint3(0u, 1u, 2u));
}

void ensure_non_empty(QuakeHostDynamicMesh& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty())
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

QuakeMaterial make_alias_material(aliashdr_t* hdr, int skin) {
    if (hdr->numskins <= 0)
        return {};
    skin = std::clamp(skin, 0, hdr->numskins - 1);
    QuakeMaterial m;
    if (hdr->gltextures[skin][0] != nullptr)
        m.header.alpha_texture_id =
            static_cast<merian::TextureID>(hdr->gltextures[skin][0]->texnum);
    if (hdr->fbtextures[skin][0] != nullptr)
        m.payload.fullbright_tex = static_cast<merian::TextureID>(hdr->fbtextures[skin][0]->texnum);
    if (hdr->nmtextures[skin][0] != nullptr)
        m.payload.normal_tex = static_cast<merian::TextureID>(hdr->nmtextures[skin][0]->texnum);
    if (hdr->gstextures[skin][0] != nullptr)
        m.payload.gloss_tex = static_cast<merian::TextureID>(hdr->gstextures[skin][0]->texnum);
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
    return m;
}

} // namespace

void QuakeScene::build_model_registries(const merian::CommandBufferHandle& cmd) {
    const auto& ms = get_material_system();
    const auto& alloc = get_allocator();
    const auto buf_usage = vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferDst |
                           vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                           vk::BufferUsageFlagBits::eShaderDeviceAddress;

    for (int i = 1; i < MAX_MODELS; i++) {
        qmodel_t* mod = cl.model_precache[i];
        if (mod == nullptr)
            break;

        if (mod->type == mod_alias) {
            aliashdr_t* hdr = (aliashdr_t*)Mod_Extradata(mod);
            if (hdr == nullptr)
                continue;

            int16_t* indexes = (int16_t*)((uint8_t*)hdr + hdr->indexes);
            const uint32_t prim_count = static_cast<uint32_t>(hdr->numindexes / 3);
            const uint32_t vert_count = static_cast<uint32_t>(hdr->numverts_vbo);

            std::vector<merian::uint3> tris(prim_count);
            for (uint32_t t = 0; t < prim_count; t++) {
                tris[t] = merian::uint3(static_cast<uint32_t>(indexes[(t * 3) + 0]),
                                        static_cast<uint32_t>(indexes[(t * 3) + 1]),
                                        static_cast<uint32_t>(indexes[(t * 3) + 2]));
            }

            merian::BufferHandle ib =
                alloc->create_buffer(cmd, tris, buf_usage, fmt::format("alias_ib:{}", mod->name));

            alias_model_info[mod] = AliasModelInfo{hdr, std::move(ib), vert_count, prim_count};

            for (int s = 0; s < hdr->numskins; s++) {
                const QuakeMaterial mat = make_alias_material(hdr, s);
                material_id_for_alias_skin[{mod, s}] =
                    ms->add_material(quake_material_type_id, mat);
            }
        } else if (mod->type == mod_sprite) {
            msprite_t* spr = (msprite_t*)mod->cache.data;
            if (spr == nullptr)
                continue;
            for (int f = 0; f < spr->numframes; f++) {
                mspriteframe_t* frame = spr->frames[f].frameptr;
                if (frame == nullptr)
                    continue;
                const QuakeMaterial mat = make_sprite_frame_material(frame);
                material_id_for_sprite_frame[{mod, f}] =
                    ms->add_material(quake_material_type_id, mat);
            }
        }
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
    mesh->flags = merian::MeshFlags::IsMorphed | merian::MeshFlags::FrontCounterClockwise;
    seed_with_degenerate_triangle(*mesh);
    particle_mesh_id = add_mesh(std::move(mesh));
    add_mesh_instance(particle_mesh_id, particle_node_id);
}

// -----------------------------------------------------------------------
// Per-entity slot management
// -----------------------------------------------------------------------

QuakeScene::EntityMeshSlot& QuakeScene::ensure_alias_slot(entity_t* ent) {
    auto it = entity_slots.find(ent);
    if (it != entity_slots.end() && it->second.model == ent->model && it->second.kind == 0)
        return it->second;

    // Model changed or first time — tear down old slot if any.
    if (it != entity_slots.end()) {
        for (const merian::MeshID id : it->second.mesh_ids)
            remove_mesh(id);
        if (it->second.node_id != merian::NODE_ID_INVALID)
            remove_node(it->second.node_id);
        entity_slots.erase(it);
    }

    auto info_it = alias_model_info.find(ent->model);
    if (info_it == alias_model_info.end() || info_it->second.hdr->numskins <= 0)
        return entity_slots[ent]; // empty slot; caller will skip

    const AliasModelInfo& info = info_it->second;
    const auto& alloc = get_allocator();

    const vk::DeviceSize vb_size = info.vertex_count * sizeof(merian::PackedVertexData);
    const vk::DeviceSize prev_vb_size = info.vertex_count * sizeof(merian::PackedPrevVertexData);
    const auto staging_usage =
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer;

    auto vb = alloc->create_buffer(vb_size, staging_usage,
                                   merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                   fmt::format("alias_vb:{}", ent->model->name));
    auto prev_vb = alloc->create_buffer(prev_vb_size, staging_usage,
                                        merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                        fmt::format("alias_prev_vb:{}", ent->model->name));

    const int skin = std::clamp(ent->skinnum, 0, info.hdr->numskins - 1);
    auto mat_it = material_id_for_alias_skin.find({ent->model, skin});
    merian::MaterialID mid =
        (mat_it != material_id_for_alias_skin.end()) ? mat_it->second : merian::MaterialID{};

    merian::SceneNode node;
    node.name = fmt::format("alias:{}", ent->model->name);
    node.is_animated = true;
    node.local_transform = entity_transform(ent);
    const merian::NodeID nid = add_node(std::move(node));

    auto mesh = std::make_unique<AliasInstanceMesh>();
    mesh->name = fmt::format("alias:{}", ent->model->name);
    mesh->material_id = mid;
    mesh->flags = merian::MeshFlags::IsMorphed | merian::MeshFlags::FrontCounterClockwise;
    mesh->vb_staging = std::move(vb);
    mesh->prev_vb_staging = std::move(prev_vb);
    mesh->ib_shared = info.index_buffer;
    mesh->vertex_count = info.vertex_count;
    mesh->primitive_count = info.primitive_count;

    const merian::MeshID mesh_id = add_mesh(std::move(mesh));
    add_mesh_instance(mesh_id, nid);

    auto& slot = entity_slots[ent];
    slot.node_id = nid;
    slot.mesh_ids = {mesh_id};
    slot.model = ent->model;
    slot.kind = 0;
    return slot;
}

QuakeScene::EntityMeshSlot& QuakeScene::ensure_brush_slot(entity_t* ent,
                                                          const merian::CommandBufferHandle& cmd) {
    auto it = entity_slots.find(ent);
    if (it != entity_slots.end() && it->second.model == ent->model && it->second.kind == 1)
        return it->second;

    if (it != entity_slots.end()) {
        for (const merian::MeshID id : it->second.mesh_ids)
            remove_mesh(id);
        if (it->second.node_id != merian::NODE_ID_INVALID)
            remove_node(it->second.node_id);
        entity_slots.erase(it);
    }

    // Lazily build submodel geometry on first reference.
    auto geo_it = brush_submodel_geo.find(ent->model);
    if (geo_it == brush_submodel_geo.end()) {
        qmodel_t* mod = ent->model;
        std::map<TexFlagsKey,
                 std::pair<std::vector<merian::PackedVertexData>, std::vector<merian::uint3>>>
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
                for (int v = 2; v < p->numverts; v++)
                    idxs.push_back(
                        merian::uint3(base, base + uint32_t(v) - 1u, base + uint32_t(v)));
            }
        }

        const auto& alloc = get_allocator();
        const auto& ms = get_material_system();
        const auto buf_usage =
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress;

        auto& geo_parts = brush_submodel_geo[mod];
        for (auto& [key, data] : parts) {
            auto& [verts, idxs] = data;
            if (idxs.empty())
                continue;

            // Find or create material for this (texture, flags) combo.
            auto mat_it = material_id_for_tex.find(key);
            merian::MaterialID mid;
            if (mat_it != material_id_for_tex.end()) {
                mid = mat_it->second;
            } else {
                const QuakeMaterial mat = make_brush_material_for(key.tex, key.surf_flags);
                mid = ms->add_material(quake_material_type_id, mat);
                material_id_for_tex.emplace(key, mid);
            }

            auto vb =
                alloc->create_buffer(cmd, verts, buf_usage, fmt::format("brush_vb:{}", mod->name));
            auto ib =
                alloc->create_buffer(cmd, idxs, buf_usage, fmt::format("brush_ib:{}", mod->name));

            geo_parts.push_back(BrushSubmodelGeoPart{std::move(vb), std::move(ib),
                                                     static_cast<uint32_t>(verts.size()),
                                                     static_cast<uint32_t>(idxs.size()), mid});
        }
        geo_it = brush_submodel_geo.find(mod);
    }

    if (geo_it == brush_submodel_geo.end() || geo_it->second.empty())
        return entity_slots[ent];

    merian::SceneNode node;
    node.name = fmt::format("brush:{}", ent->model->name);
    node.is_animated = true;
    node.local_transform = entity_transform(ent);
    const merian::NodeID nid = add_node(std::move(node));

    auto& slot = entity_slots[ent];
    slot.node_id = nid;
    slot.model = ent->model;
    slot.kind = 1;

    for (const auto& part : geo_it->second) {
        auto mesh = std::make_unique<BrushEntityMesh>();
        mesh->name = fmt::format("brush:{}:{}", ent->model->name, part.material_id);
        mesh->material_id = part.material_id;
        mesh->flags = merian::MeshFlags::IsOpaque | merian::MeshFlags::FrontCounterClockwise;
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

QuakeScene::EntityMeshSlot& QuakeScene::ensure_sprite_slot(entity_t* ent) {
    auto it = entity_slots.find(ent);
    if (it != entity_slots.end() && it->second.model == ent->model && it->second.kind == 2)
        return it->second;

    if (it != entity_slots.end()) {
        for (const merian::MeshID id : it->second.mesh_ids)
            remove_mesh(id);
        if (it->second.node_id != merian::NODE_ID_INVALID)
            remove_node(it->second.node_id);
        entity_slots.erase(it);
    }

    merian::SceneNode node;
    node.name = fmt::format("sprite:{}", ent->model->name);
    const merian::NodeID nid = add_node(std::move(node));

    int frame = std::max(ent->frame, 0);
    auto mat_it = material_id_for_sprite_frame.find({ent->model, frame});
    merian::MaterialID mid =
        (mat_it != material_id_for_sprite_frame.end()) ? mat_it->second : merian::MaterialID{};

    auto mesh = std::make_unique<QuakeHostDynamicMesh>();
    mesh->name = fmt::format("sprite:{}", ent->model->name);
    mesh->material_id = mid;
    mesh->flags = merian::MeshFlags::IsMorphed | merian::MeshFlags::FrontCounterClockwise;
    seed_with_degenerate_triangle(*mesh);

    const merian::MeshID mesh_id = add_mesh(std::move(mesh));
    add_mesh_instance(mesh_id, nid);

    auto& slot = entity_slots[ent];
    slot.node_id = nid;
    slot.mesh_ids = {mesh_id};
    slot.model = ent->model;
    slot.kind = 2;
    return slot;
}

void QuakeScene::fill_alias_pose(QuakeScene::EntityMeshSlot& slot, entity_t* ent) {
    if (slot.mesh_ids.empty())
        return;

    const auto& meshes = get_meshes();
    auto& mesh = static_cast<AliasInstanceMesh&>(*meshes[slot.mesh_ids[0]]);

    auto* vb = mesh.vb_staging->get_memory()->map_as<merian::PackedVertexData>();
    auto* prev_vb = mesh.prev_vb_staging->get_memory()->map_as<merian::PackedPrevVertexData>();

    merian::float4x4 lerped_transform;
    compute_alias_lerped(ent, vb, prev_vb, &lerped_transform);

    mesh.vb_staging->get_memory()->unmap();
    mesh.prev_vb_staging->get_memory()->unmap();

    update_node(slot.node_id, lerped_transform);

    get_meshes()[slot.mesh_ids[0]]->vertices_dirty = true;
}

void QuakeScene::retire_stale_entity_slots() {
    constexpr uint64_t STALE_THRESHOLD = 8;
    for (auto it = entity_slots.begin(); it != entity_slots.end();) {
        if (frame - it->second.last_seen_frame > STALE_THRESHOLD) {
            for (const merian::MeshID id : it->second.mesh_ids)
                remove_mesh(id);
            if (it->second.node_id != merian::NODE_ID_INVALID)
                remove_node(it->second.node_id);
            it = entity_slots.erase(it);
        } else {
            ++it;
        }
    }
}

void QuakeScene::refresh_entities(const merian::CommandBufferHandle& cmd) {
    auto process_entity = [&](entity_t* ent) {
        if (ent == nullptr || ent->model == nullptr)
            return;

        switch (ent->model->type) {
        case mod_alias: {
            auto& slot = ensure_alias_slot(ent);
            if (!slot.mesh_ids.empty()) {
                fill_alias_pose(slot, ent);
                slot.last_seen_frame = frame;
            }
            break;
        }
        case mod_brush: {
            auto& slot = ensure_brush_slot(ent, cmd);
            if (!slot.mesh_ids.empty()) {
                update_node(slot.node_id, entity_transform(ent));
                slot.last_seen_frame = frame;
            }
            break;
        }
        case mod_sprite: {
            auto& slot = ensure_sprite_slot(ent);
            if (!slot.mesh_ids.empty()) {
                auto& mesh = static_cast<QuakeHostDynamicMesh&>(*get_meshes()[slot.mesh_ids[0]]);
                mesh.vertices.clear();
                mesh.prev_vertices.clear();
                mesh.indices.clear();

                std::vector<merian::float3> prev_pos;
                extract_sprite_geo(ent, mesh.vertices, prev_pos, mesh.indices);
                mesh.prev_vertices.resize(prev_pos.size());
                for (size_t i = 0; i < prev_pos.size(); ++i)
                    mesh.prev_vertices[i].position = prev_pos[i];

                ensure_non_empty(mesh);

                int frame = std::max(ent->frame, 0);
                auto mat_it = material_id_for_sprite_frame.find({ent->model, frame});
                if (mat_it != material_id_for_sprite_frame.end())
                    mesh.material_id = mat_it->second;

                get_meshes()[slot.mesh_ids[0]]->vertices_dirty = true;
                get_meshes()[slot.mesh_ids[0]]->indices_dirty = true;
                slot.last_seen_frame = frame;
            }
            break;
        }
        default:
            break;
        }
    };

    if (playermodel == 1) {
        process_entity(&cl.viewent);
    } else if (playermodel == 2) {
        process_entity(&cl.viewent);
        if (cl.viewentity > 0 && cl.viewentity < cl_max_edicts && (cl_entities != nullptr))
            process_entity(&cl_entities[cl.viewentity]);
    }

    for (int i = 0; i < cl_numvisedicts; i++)
        process_entity(cl_visedicts[i]);

    for (int i = 0; i < cl.num_statics; i++)
        process_entity(&cl_static_entities[i]);

    // Particle batch.
    {
        auto& mesh = static_cast<QuakeHostDynamicMesh&>(*get_meshes()[particle_mesh_id]);
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

    retire_stale_entity_slots();
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

    config.st_separate("Scene");

    Scene::properties(config);
}

} // namespace merian_quake
