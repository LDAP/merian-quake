#include "quake_node.hpp"

#include "merian/utils/audio/sdl_audio_device.hpp"
#include "merian/utils/colors.hpp"

#include <GLFW/glfw3.h>

extern "C" {
#include "bgmusic.h"
#include "quakedef.h"
#include "screen.h"

extern cvar_t cl_maxpitch; // johnfitz -- variable pitch clamping
extern cvar_t cl_minpitch; // johnfitz -- variable pitch clamping
}

struct QuakeData {
    RendererMarkovChain* render_node{nullptr};
    QuakeNode* quake_node{nullptr};
    quakeparms_t params;
    std::unique_ptr<merian::SDLAudioDevice> audio_device;
};
// Quake uses lots of static global variables,
// so we need to do that to
// e.g. to make sure there is only one QuakeNode.
static QuakeData quake_data;

// CALLBACKS from within Quake --------------------------------------------------------------------

// called each time a new map is (re)loaded
extern "C" void QS_worldspawn() {
    quake_data.quake_node->QS_worldspawn();
}

// called when a texture should be loaded
extern "C" void QS_texture_load(gltexture_t* glt, uint32_t* data) {
    quake_data.quake_node->QS_texture_load(glt, data);
}

// called from within qs, pretty much a copy from in_sdl.c:
extern "C" void IN_Move(usercmd_t* cmd) {
    quake_data.quake_node->IN_Move(cmd);
}

// called when rendering
extern "C" void R_RenderScene() {
    quake_data.quake_node->R_RenderScene();
}

extern "C" qboolean SNDDMA_Init(dma_t* dma) {
    quake_data.audio_device = std::make_unique<merian::SDLAudioDevice>();

    const auto callback = [](uint8_t* stream, int len) {
        // from
        // https://github.com/sezero/quakespasm/blob/70df2b661e9c632d04825b259e63ad58c29c01ac/Quake/snd_sdl.c#L156
        int buffersize = shm->samples * (shm->samplebits / 8);
        int pos, tobufend;
        int len1, len2;

        if (!shm) { /* shouldn't happen, but just in case */
            memset(stream, 0, len);
            return;
        }

        pos = (shm->samplepos * (shm->samplebits / 8));
        if (pos >= buffersize)
            shm->samplepos = pos = 0;

        tobufend = buffersize - pos; /* bytes to buffer's end. */
        len1 = len;
        len2 = 0;

        if (len1 > tobufend) {
            len1 = tobufend;
            len2 = len - len1;
        }

        memcpy(stream, shm->buffer + pos, len1);

        if (len2 <= 0) {
            shm->samplepos += (len1 / (shm->samplebits / 8));
        } else { /* wraparound? */
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
        desired.buffersize = 2048; /* for 48 kHz */
    else
        desired.buffersize = 4096; /* for 96 kHz */

    std::optional<merian::AudioDevice::AudioSpec> actual =
        quake_data.audio_device->open_device(desired, callback);

    if (!actual) {
        return false;
    }

    memset((void*)dma, 0, sizeof(dma_t));
    shm = dma;
    /* Fill the audio DMA information block */
    shm->samplebits = (actual->format & 0xFF); /* first byte of format is bits */
    shm->signed8 = (actual->format == merian::AudioDevice::FORMAT_S8);
    shm->speed = actual->samplerate;
    shm->channels = actual->channels;
    int tmp = (actual->buffersize * actual->channels) * 10;
    /* make it a power of two */
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
    shm->buffer = (unsigned char*)calloc(1, buffersize);

    quake_data.audio_device->unpause_audio();
    return true;
}

extern "C" int SNDDMA_GetDMAPos(void) {
    if (shm)
        return shm->samplepos;
    else
        return 0;
}

extern "C" void SNDDMA_Shutdown(void) {
    if (shm) {
        if (shm->buffer)
            free(shm->buffer);
        shm->buffer = NULL;
        shm = NULL;
    }
    quake_data.audio_device.reset();
}
extern "C" void SNDDMA_LockBuffer(void) {
    if (!quake_data.audio_device)
        return;
    quake_data.audio_device->lock_device();
}
extern "C" void SNDDMA_Submit(void) {
    if (!quake_data.audio_device)
        return;
    quake_data.audio_device->unlock_device();
}
extern "C" void SNDDMA_BlockSound(void) {
    if (!quake_data.audio_device)
        return;
    quake_data.audio_device->pause_audio();
}
extern "C" void SNDDMA_UnblockSound(void) {
    if (!quake_data.audio_device)
        return;
    quake_data.audio_device->unpause_audio();
}

void parse_worldspawn(QuakeNode::QuakeRenderInfo& renderer_info) {
    glm::vec3& quake_sun_col = renderer_info.sun_color;
    glm::vec3& quake_sun_dir = renderer_info.sun_direction;

    std::map<std::string, std::string> worldspawn_props;
    char key[128], value[4096];
    const char* data;

    data = COM_Parse(cl.worldmodel->entities);
    if (!data)
        return; // error
    if (com_token[0] != '{')
        return; // error
    while (1) {
        data = COM_Parse(data);
        if (!data)
            return; // error
        if (com_token[0] == '}')
            break; // end of worldspawn
        if (com_token[0] == '_')
            q_strlcpy(key, com_token + 1, sizeof(key));
        else
            q_strlcpy(key, com_token, sizeof(key));
        while (key[0] && key[strlen(key) - 1] == ' ') // remove trailing spaces
            key[strlen(key) - 1] = 0;
        data = COM_Parse(data);
        if (!data)
            return; // error
        q_strlcpy(value, com_token, sizeof(value));

        SPDLOG_DEBUG("{} {}", key, value);
        worldspawn_props[key] = value;
    }

    quake_sun_col = glm::vec3(0);
    for (const std::string k : {"sunlight", "sunlight2", "sunlight3"}) {
        if (worldspawn_props.contains(k)) {
            glm::vec3 col(0);

            if (worldspawn_props.contains(k + "_color")) {
                sscanf(worldspawn_props[k + "_color"].c_str(), "%f %f %f", &col.r, &col.g, &col.b);
            } else {
                col = glm::vec3(1);
            }

            float intensity = std::stoi(worldspawn_props[k]);
            col *= intensity;
            col /= 4000.;

            if (merian::yuv_luminance(col) > merian::yuv_luminance(quake_sun_col)) {
                quake_sun_col = col;
            }
        }
    }

    if (worldspawn_props.contains("sun_mangle")) {
        float angles[3];
        sscanf(worldspawn_props["sun_mangle"].c_str(), "%f %f %f", &angles[1], &angles[0],
               &angles[2]);
        // This seems wrong.. But works on ad_azad
        float right[3], up[3];
        angles[1] -= 180;
        AngleVectors(angles, &quake_sun_dir.x, right, up);
    } else {
        quake_sun_dir = glm::vec3(1, 1, 1);
    }

    // Some patches for maps
    if (worldspawn_props.contains("sky") && worldspawn_props["sky"] == "stormydays_") {
        // ad_tears
        quake_sun_dir = glm::vec3(1, -1, 1);
        quake_sun_col = glm::vec3(1.1, 1.0, 0.9);
        quake_sun_col *= 6.0;
    }

    // prevent float16 overflow
    const float max_col = std::max(std::max(quake_sun_col.r, quake_sun_col.g), quake_sun_col.b);
    if (max_col > MAX_SUN_COLOR)
        quake_sun_col = quake_sun_col / max_col * MAX_SUN_COLOR;
    quake_sun_dir = glm::normalize(quake_sun_dir);
}

QuakeNode::QuakeNode([[maybe_unused]] const merian::SharedContext& context,
                     const merian::ResourceAllocatorHandle allocator,
                     const std::shared_ptr<merian::InputController>& controller,
                     const int quakespasm_argc,
                     const char** quakespasm_argv,
                     RendererMarkovChain* renderer)
    : Node("Quake"), allocator(allocator), controller(controller) {

    // clang-format off
    controller->set_key_event_callback([&](merian::InputController&, int key, int, merian::InputController::KeyStatus action, int){
        static const std::map<int, int> keymap = {
            {GLFW_KEY_TAB, K_TAB},
            {GLFW_KEY_ENTER, K_ENTER},
            {GLFW_KEY_ESCAPE, K_ESCAPE},
            {GLFW_KEY_SPACE, K_SPACE},

            {GLFW_KEY_BACKSPACE, K_BACKSPACE},
            {GLFW_KEY_UP, K_UPARROW},
            {GLFW_KEY_DOWN, K_DOWNARROW},
            {GLFW_KEY_LEFT, K_LEFTARROW},
            {GLFW_KEY_RIGHT, K_RIGHTARROW},

            {GLFW_KEY_LEFT_ALT, K_ALT},
            {GLFW_KEY_LEFT_CONTROL, K_CTRL},
            {GLFW_KEY_LEFT_SHIFT, K_SHIFT},
            {GLFW_KEY_F1, K_F1},
            {GLFW_KEY_F2, K_F2},
            {GLFW_KEY_F3, K_F3},
            {GLFW_KEY_F4, K_F4},
            {GLFW_KEY_F5, K_F5},
            {GLFW_KEY_F6, K_F6},
            {GLFW_KEY_F7, K_F7},
            {GLFW_KEY_F8, K_F8},
            {GLFW_KEY_F9, K_F9},
            {GLFW_KEY_F10, K_F10},
            {GLFW_KEY_F11, K_F11},
            {GLFW_KEY_F12, K_F12},
        };

        // normal keys sould be passed as lowercased ascii
        if (key >= 65 && key <= 90) key |= 32;
        else if (keymap.contains(key)) key = keymap.at(key);

        if (action == merian::InputController::PRESS) {
            Key_Event(key, true);
        } else if (action == merian::InputController::RELEASE) {
            Key_Event(key, false);
        }
    });
    controller->set_mouse_cursor_callback([&](merian::InputController& controller, double xpos, double ypos){
        const bool raw = controller.get_raw_mouse_input();

        if (raw) {
            this->mouse_x = xpos;
            this->mouse_y = ypos;
        }

        if (raw != raw_mouse_was_enabled || !raw) {
            this->mouse_x = this->mouse_oldx = xpos;
            this->mouse_y = this->mouse_oldy = ypos;
        }
   
        raw_mouse_was_enabled = raw;
    });
    controller->set_mouse_button_callback([&](merian::InputController&, merian::InputController::MouseButton button, merian::InputController::KeyStatus status, int){
        const int remap[] = {K_MOUSE1, K_MOUSE2, K_MOUSE3, K_MOUSE4, K_MOUSE5};
        Key_Event(remap[button], status == merian::InputController::PRESS);
    });
    controller->set_scroll_event_callback([&](merian::InputController&, double xoffset, double yoffset){
        if (yoffset > 0) {
            Key_Event(K_MWHEELUP, true);
            Key_Event(K_MWHEELUP, false);
        } else if (xoffset < 0) {
            Key_Event(K_MWHEELDOWN, true);
            Key_Event(K_MWHEELDOWN, false);
        }
    });
    // clang-format on

    // INIT QUAKE
    if (quake_data.quake_node) {
        throw std::runtime_error{"Only one quake node can be created."};
    }
    quake_data.quake_node = this;
    quake_data.render_node = renderer;
    host_parms = &quake_data.params;

    game_thread = std::thread([&, quakespasm_argc, quakespasm_argv] {
        // INIT Quake
        std::vector<const char*> quakespasm_args = {"quakespasm"};
        if (quakespasm_argc > 0) {
            quakespasm_args.resize(1 + quakespasm_argc);
            memcpy(&quakespasm_args[1], quakespasm_argv, quakespasm_argc * sizeof(quakespasm_argv));
        }

        quake_data.params.argc = quakespasm_args.size();
        quake_data.params.argv = (char**)quakespasm_args.data();
        quake_data.params.errstate = 0;
        quake_data.params.memsize = 256 * 1024 * 1024; // qs default in 0.94.3
        quake_data.params.membase = malloc(quake_data.params.memsize);

        srand(1337); // quake uses this
        COM_InitArgv(quake_data.params.argc, quake_data.params.argv);
        Sys_Init();

        Sys_Printf("Quake %1.2f (c) id Software\n", VERSION);
        Sys_Printf("GLQuake %1.2f (c) id Software\n", GLQUAKE_VERSION);
        Sys_Printf("FitzQuake %1.2f (c) John Fitzgibbons\n", FITZQUAKE_VERSION);
        Sys_Printf("FitzQuake SDL port (c) SleepwalkR, Baker\n");
        Sys_Printf("QuakeSpasm " QUAKESPASM_VER_STRING
                   " (c) Ozkan Sezer, Eric Wasylishen & others\n");

        Sys_Printf("Host_Init\n");
        Host_Init();

        // Set target
        key_dest = key_game;
        m_state = m_none;
        merian::Stopwatch sw;

        // RUN GAMELOOP
        while (game_running) {

            if (!pending_commands.empty()) {
                Cmd_ExecuteString(pending_commands.front().c_str(), src_command);
                pending_commands.pop();
            }

            double newtime = Sys_DoubleTime();
            double timediff;
            if (force_timediff > 0) {
                timediff = force_timediff / 1000.0;
            } else {
                timediff = old_time == 0 ? 0. : newtime - old_time;
            }

            try {
                render_info.render = false;
                Host_Frame(timediff);
                if (!render_info.render) {
                    // make sure we release the main thread
                    sync_gamestate.push(true, 1);
                    if (!game_running) {
                        std::runtime_error{"quit"};
                    }
                    sync_render.pop();
                }
            } catch (const std::runtime_error&) {
                // game quit, do nothing
            }
            old_time = newtime;

            server_fps = 1 / sw.seconds();
            sw.reset();
        }

        // CLEANUP
        free(quake_data.params.membase);
    });
    sync_gamestate.pop();
}

QuakeNode::~QuakeNode() {
    game_running.store(false);
    // make sure to unlock
    sync_render.push(true);
    sync_render.push(true);
    game_thread.join();

    quake_data.audio_device.reset();
}

void QuakeNode::QS_worldspawn() {
    SPDLOG_DEBUG("worldspawn");
    render_info.worldspawn = true;
}

void QuakeNode::IN_Move(usercmd_t* cmd) {
    SPDLOG_TRACE("move");

    // pretty much a copy from in_sdl.c:

    int dmx = (this->mouse_x - this->mouse_oldx) * sensitivity.value;
    int dmy = (this->mouse_y - this->mouse_oldy) * sensitivity.value;
    this->mouse_oldx = this->mouse_x;
    this->mouse_oldy = this->mouse_y;

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
        /* johnfitz -- variable pitch clamping */
        if (cl.viewangles[PITCH] > cl_maxpitch.value)
            cl.viewangles[PITCH] = cl_maxpitch.value;
        if (cl.viewangles[PITCH] < cl_minpitch.value)
            cl.viewangles[PITCH] = cl_minpitch.value;
    } else {
        if ((in_strafe.state & 1) && noclip_anglehack)
            cmd->upmove -= m_forward.value * dmy;
        else
            cmd->forwardmove -= m_forward.value * dmy;
    }
}

void QuakeNode::R_RenderScene() {
    if (!game_running) {
        std::runtime_error{"quit"};
    }
    render_info.render = true;
    sync_gamestate.push(true, 1);
    sync_render.pop();
}

void QuakeNode::QS_texture_load(gltexture_t* glt, uint32_t* data) {
    // LOG -----------------------------------------------

    std::string source = strcmp(glt->source_file, "") == 0 ? "memory" : glt->source_file;
    SPDLOG_DEBUG("texture_load {} {} {}x{} from {}, frame: {}", glt->texnum, glt->name, glt->width,
                 glt->height, source, glt->visframe);

    if (glt->width == 0 || glt->height == 0) {
        SPDLOG_WARN("image extent was 0. skipping");
        return;
    }

    // STORE SOME TEXTURE IDs ----------------------------

    // HACK: for blood patch
    if (!strcmp(glt->name, "progs/gib_1.mdl:frame0"))
        render_info.texnum_blood = glt->texnum;
    // HACK: for sparks and for emissive rocket particle trails
    if (!strcmp(glt->name, "progs/s_exp_big.spr:frame10"))
        render_info.texnum_explosion = glt->texnum;

    // ALLOCATE ----------------------------

    // We store the texture on system memory for now
    // and upload in cmd_process later
    pending_uploads.emplace(glt, data);
}

std::vector<merian_nodes::OutputConnectorHandle>
QuakeNode::describe_outputs([[maybe_unused]] const merian_nodes::ConnectorIOMap& output_for_input) {
    return {con_render_info, con_textures};
}

void QuakeNode::update_textures(const vk::CommandBuffer& cmd, const merian_nodes::NodeIO& io) {
    for (auto tex : pending_uploads) {
        merian::TextureHandle gpu_tex = allocator->createTextureFromRGBA8(
            cmd, tex.cpu_tex.data(), tex.width, tex.height,
            tex.flags & TEXPREF_LINEAR ? vk::Filter::eLinear : vk::Filter::eNearest, !tex.linear,
            tex.name);
        io[con_textures].set(tex.texnum, gpu_tex, cmd, vk::AccessFlagBits2::eTransferWrite,
                             vk::PipelineStageFlagBits2::eTransfer);
    }
    pending_uploads.clear();
}

void QuakeNode::process([[maybe_unused]] merian_nodes::GraphRun& run,
                        [[maybe_unused]] const vk::CommandBuffer& cmd,
                        [[maybe_unused]] const merian::DescriptorSetHandle& descriptor_set,
                        [[maybe_unused]] const merian_nodes::NodeIO& io) {
    if (update_gamestate) {
        MERIAN_PROFILE_SCOPE(run.get_profiler(), "update gamestate");
        sync_render.push(true, 1);
        sync_gamestate.pop();
    }

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update textures");
        update_textures(cmd, io);
    }

    if (cl.worldmodel && render_info.worldspawn) {
        key_dest = key_game;
        m_state = m_none;

        parse_worldspawn(render_info);
        render_info.last_worldspawn_frame = render_info.frame;
        sv_player = nullptr;
    }

    if (update_gamestate && key_dest == key_game) {
        controller->request_raw_mouse_input(true);
    } else {
        controller->request_raw_mouse_input(false);
    }

    if (stop_after_worldspawn >= 0 &&
        render_info.frame - render_info.last_worldspawn_frame == (uint64_t)stop_after_worldspawn) {
        update_gamestate = false;
        if (rebuild_after_stop)
            run.request_reconnect();
    }

    io[con_render_info] = render_info;

    if (cl.worldmodel && render_info.worldspawn) {
        render_info.worldspawn = false;
    }
    render_info.frame++;
}

QuakeNode::NodeStatusFlags QuakeNode::properties(merian::Properties& config) {
    config.st_separate("General");
    config.config_bool("gamestate update", update_gamestate);
    update_gamestate |= render_info.frame == 0;

    std::array<char, 128> cmd_buffer = {0};
    if (config.config_text("command", cmd_buffer.size(), cmd_buffer.data(), true)) {
        queue_command(cmd_buffer.data());
        if (!update_gamestate) {
            SPDLOG_WARN("command unpaused gamestate update");
            update_gamestate = true;
        }
    }
    bool changed = config.config_text_multiline(
        "startup commands", startup_commands_buffer.size(), startup_commands_buffer.data(), false,
        "multiple commands separated by newline, lines starting with # are ignored");
    if (changed && render_info.frame == 0) {
        merian::split(startup_commands_buffer.data(), "\n", [&](const std::string& cmd) {
            if (!cmd.starts_with("#"))
                queue_command(cmd);
        });
    }

    config.st_separate("Reproducibility");
    config.config_int("stop after worldspawn", stop_after_worldspawn,
                      "Can be used for reference renders.");
    config.config_bool("rebuild after stop", rebuild_after_stop);
    config.config_float("force timediff (ms)", force_timediff,
                        "For reference renders and video outputs.");

    config.st_separate("Debug / Info");
    config.output_text(fmt::format("sun direction: ({}, {}, {})\nsun color: ({}, {}, {})",
                                   render_info.sun_direction.x, render_info.sun_direction.y,
                                   render_info.sun_direction.z, render_info.sun_color.r,
                                   render_info.sun_color.g, render_info.sun_color.b));
    config.output_text(fmt::format("view angles {} {} {}", r_refdef.viewangles[0],
                                   r_refdef.viewangles[1], r_refdef.viewangles[2]));
    config.output_text(fmt::format("server fps: {}", server_fps));

    return {};
}
