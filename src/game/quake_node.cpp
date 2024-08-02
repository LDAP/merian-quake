#include "quake_node.hpp"

#include "game/quake_helpers.hpp"
#include "merian/utils/audio/sdl_audio_device.hpp"
#include "merian/utils/colors.hpp"
#include "merian/utils/concurrent/utils.hpp"
#include "merian/utils/glm.hpp"

#include <GLFW/glfw3.h>

extern "C" {
#include "bgmusic.h"
#include "quakedef.h"
#include "screen.h"

extern cvar_t cl_maxpitch; // johnfitz -- variable pitch clamping
extern cvar_t cl_minpitch; // johnfitz -- variable pitch clamping

extern qboolean scr_drawloading;
}

struct QuakeData {
    QuakeNode* quake_node{nullptr};
    quakeparms_t params;
    std::unique_ptr<merian::SDLAudioDevice> audio_device;

    // updated in parse_worldspawn
    glm::vec3 current_sun_color{};
    glm::vec3 current_sun_direction{};
};
// Quake uses lots of static global variables,
// so we need to do that to
// e.g. to make sure there is only one QuakeNode.
static QuakeData quake_data;

void init_quakespasm(const int quakespasm_argc, const char** quakespasm_argv) {
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
    Sys_Printf("QuakeSpasm " QUAKESPASM_VER_STRING " (c) Ozkan Sezer, Eric Wasylishen & others\n");

    Host_Init();

    // Set target
    key_dest = key_game;
    m_state = m_none;
}

void shutdown_quakespasm() {
    CL_Disconnect();
    Host_ShutdownServer(false);
    Host_Shutdown();

    free(quake_data.params.membase);
    quake_data.quake_node = nullptr;
}

// CALLBACKS from within Quake --------------------------------------------------------------------

extern "C" void VID_Changed_f(cvar_t* var) {
    quake_data.quake_node->VID_Changed_f(var);
}

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

void parse_worldspawn() {
    glm::vec3& quake_sun_col = quake_data.current_sun_color;
    glm::vec3& quake_sun_dir = quake_data.current_sun_direction;

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

// If the supplied buffer is not nullptr and is large enough, it is returned and an upload is
// recorded. A new larger buffer is used otherwise.
template <typename T>
merian::BufferHandle ensure_buffer(const merian::ResourceAllocatorHandle& allocator,
                                   const vk::BufferUsageFlags usage,
                                   const vk::CommandBuffer& cmd,
                                   const std::vector<T>& data,
                                   const merian::BufferHandle optional_buffer,
                                   const std::optional<vk::DeviceSize> min_alignment = std::nullopt,
                                   const std::string& debug_name = {}) {
    merian::BufferHandle buffer = optional_buffer;
    if (!allocator->ensureBufferSize(buffer, merian::size_of(data), usage, debug_name,
                                     merian::MemoryMappingType::NONE, min_alignment, 1.25)) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer, {}, {},
            buffer->buffer_barrier(vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                       vk::AccessFlagBits::eShaderRead |
                                       vk::AccessFlagBits::eTransferRead,
                                   vk::AccessFlagBits::eTransferWrite),
            {});
    }
    allocator->getStaging()->cmdToBuffer(cmd, *buffer, 0, merian::size_of(data), data.data());
    return buffer;
}

// Creates a vertex and index buffer for rt on the device and records the upload.
// If the supplied buffers are not nullptr and are large enough, they are returned and an upload
// is recorded. Returns (vertex_buffer, index_buffer).
// Appropriate barriers are inserted.
std::tuple<merian::BufferHandle, merian::BufferHandle, merian::BufferHandle, merian::BufferHandle>
ensure_vertex_index_ext_buffer(const merian::ResourceAllocatorHandle& allocator,
                               const vk::CommandBuffer& cmd,
                               const std::vector<float>& vtx,
                               const std::vector<float>& prev_vtx,
                               const std::vector<uint32_t>& idx,
                               const std::vector<VertexExtraData>& ext,
                               const merian::BufferHandle optional_vtx_buffer,
                               const merian::BufferHandle optional_prev_vtx_buffer,
                               const merian::BufferHandle optional_idx_buffer,
                               const merian::BufferHandle optional_ext_buffer) {
    auto usage_rt = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    auto usage_storage = vk::BufferUsageFlagBits::eStorageBuffer;

    merian::BufferHandle vertex_buffer = ensure_buffer(
        allocator, usage_rt, cmd, vtx, optional_vtx_buffer, {}, "Quake: vertex buffer");
    merian::BufferHandle prev_vertex_buffer =
        ensure_buffer(allocator, usage_rt, cmd, prev_vtx, optional_prev_vtx_buffer, {},
                      "Quake: previous vertex buffer");
    merian::BufferHandle index_buffer = ensure_buffer(
        allocator, usage_rt, cmd, idx, optional_idx_buffer, {}, "Quake: index buffer");
    merian::BufferHandle ext_buffer = ensure_buffer(allocator, usage_storage, cmd, ext,
                                                    optional_ext_buffer, {}, "Quake: ext buffer");

    const std::array<vk::BufferMemoryBarrier2, 4> barriers = {
        vertex_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
                vk::AccessFlagBits2::eTransferWrite),
        prev_vertex_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferWrite),
        index_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
                vk::AccessFlagBits2::eTransferWrite),
        ext_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferWrite),
    };

    vk::DependencyInfo dep_info{{}, {}, barriers, {}};
    cmd.pipelineBarrier2(dep_info);

    return std::make_tuple(vertex_buffer, prev_vertex_buffer, index_buffer, ext_buffer);
}

QuakeNode::RTGeometry get_rt_geometry(const merian::ResourceAllocatorHandle& allocator,
                                      const vk::CommandBuffer& cmd,
                                      const std::vector<float>& vtx,
                                      const std::vector<float>& prev_vtx,
                                      const std::vector<uint32_t>& idx,
                                      const std::vector<VertexExtraData>& ext,
                                      const QuakeNode::RTGeometry old_geo,
                                      const vk::BuildAccelerationStructureFlagsKHR flags) {
    assert(!vtx.empty());
    assert(!prev_vtx.empty());
    assert(!idx.empty());
    assert(!ext.empty());

    QuakeNode::RTGeometry geo;

    const uint32_t vertex_count = vtx.size() / 3;
    const uint32_t primitive_count = idx.size() / 3;

    std::tie(geo.vtx, geo.prev_vtx, geo.idx, geo.ext) =
        ensure_vertex_index_ext_buffer(allocator, cmd, vtx, prev_vtx, idx, ext, old_geo.prev_vtx,
                                       old_geo.vtx, old_geo.idx, old_geo.ext);

    if (old_geo.blas_info) {
        geo.blas_info = old_geo.blas_info;
        geo.geo_handle = old_geo.geo_handle;
        geo.blas_info->update_geometry_f32_u32(geo.geo_handle, vertex_count, primitive_count,
                                               geo.vtx, geo.idx);
    } else {
        geo.blas_info = std::make_shared<merian_nodes::DeviceASBuilder::BlasBuildInfo>(flags);
        geo.geo_handle =
            geo.blas_info->add_geometry_f32_u32(vertex_count, primitive_count, geo.vtx, geo.idx);
    }

    return geo;
}

QuakeNode::QuakeNode([[maybe_unused]] const merian::ContextHandle& context,
                     const merian::ResourceAllocatorHandle allocator,
                     const int quakespasm_argc,
                     const char** quakespasm_argv)
    : Node(), context(context), allocator(allocator) {

    // reserve roughly 1GB for all vectors
    vtx.reserve(256 * 1024 * 1024 / sizeof(float));
    prev_vtx.reserve(256 * 1024 * 1024 / sizeof(float));
    idx.reserve(256 * 1024 * 1024 / sizeof(uint32_t));
    ext.reserve(256 * 1024 * 1024 / sizeof(VertexExtraData));

    // INIT QUAKE
    if (quake_data.quake_node) {
        throw std::runtime_error{"Only one quake node can be created."};
    }
    quake_data.quake_node = this;
    host_parms = &quake_data.params;

    init_quakespasm(quakespasm_argc, quakespasm_argv);

    game_thread = std::thread([&] {
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
    });
    sync_gamestate.pop();
}

QuakeNode::~QuakeNode() {
    game_running.store(false);
    // make sure to unlock
    sync_render.push(true);
    sync_render.push(true);
    game_thread.join();

    shutdown_quakespasm();
}

void QuakeNode::QS_worldspawn() {
    SPDLOG_DEBUG("worldspawn");

    parse_worldspawn();

    last_worldspawn_frame = frame;
    render_info.constant_data_update = true;
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

void QuakeNode::VID_Changed_f([[maybe_unused]] cvar_t* var) {
    if (!con_resolution) {
        return;
    }
    con_resolution->set(
        vk::Extent3D{static_cast<uint32_t>(vid.width), static_cast<uint32_t>(vid.height), 1});
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
        texnum_blood = glt->texnum;
    // HACK: for sparks and for emissive rocket particle trails
    if (!strcmp(glt->name, "progs/s_exp_big.spr:frame10"))
        texnum_explosion = glt->texnum;

    // ALLOCATE ----------------------------

    // We store the texture on system memory for now
    // and upload in cmd_process later
    if (pending_uploads.contains(glt->texnum)) {
        pending_uploads.erase(glt->texnum);
    }
    pending_uploads.try_emplace(glt->texnum, glt, data);
}

void QuakeNode::set_controller(const merian::InputControllerHandle& controller) {
    this->controller = controller;

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
}

std::vector<merian_nodes::OutputConnectorHandle>
QuakeNode::describe_outputs([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout) {
    con_resolution = merian_nodes::SpecialStaticOut<vk::Extent3D>::create(
        "resolution",
        vk::Extent3D{static_cast<uint32_t>(vid.width), static_cast<uint32_t>(vid.height), 1});

    return {
        con_resolution, con_render_info, con_tlas_info, con_textures,
        con_vtx,        con_prev_vtx,    con_idx,       con_ext,
    };
}

void QuakeNode::update_textures(const vk::CommandBuffer& cmd, const merian_nodes::NodeIO& io) {

    for (const auto& [texnum, tex] : pending_uploads) {
        vk::Filter mag_filter;
        if (default_filtering == 0) {
            mag_filter = tex.flags & TEXPREF_LINEAR ? vk::Filter::eLinear : vk::Filter::eNearest;
        } else {
            mag_filter = tex.flags & TEXPREF_NEAREST ? vk::Filter::eNearest : vk::Filter::eLinear;
        }

        merian::TextureHandle gpu_tex = allocator->createTextureFromRGBA8(
            cmd, tex.cpu_tex.data(), tex.width, tex.height, mag_filter, vk::Filter::eLinear,
            !tex.linear, tex.name, tex.flags & TEXPREF_MIPMAP);
        io[con_textures].set(texnum, gpu_tex, cmd, vk::AccessFlagBits2::eTransferWrite,
                             vk::PipelineStageFlagBits2::eTransfer);
    }
    pending_uploads.clear();
}

QuakeNode::NodeStatusFlags QuakeNode::on_connected(
    [[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
    [[maybe_unused]] const merian::DescriptorSetLayoutHandle& descriptor_set_layout) {
    render_info.constant_data_update = true;
    return {};
}

QuakeNode::NodeStatusFlags QuakeNode::pre_process([[maybe_unused]] merian_nodes::GraphRun& run,
                                                  [[maybe_unused]] const merian_nodes::NodeIO& io) {
    return {};
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

    render_info.render &= !scr_drawloading;

    if (cl.worldmodel && frame == last_worldspawn_frame) {
        key_dest = key_game;
        m_state = m_none;
        sv_player = nullptr;

        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update static geo");
            update_static_geo(cmd);
        }
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update dynamic geo");
        update_dynamic_geo(cmd, run.get_profiler());
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "update as");
        update_as(cmd, io);
    }

    // Update constant data
    if (render_info.constant_data_update) {
        if (overwrite_sun) {
            render_info.constant.sun_color = overwrite_sun_col;
            render_info.constant.sun_direction = overwrite_sun_dir;
        } else {
            render_info.constant.sun_color = quake_data.current_sun_color;
            render_info.constant.sun_direction = quake_data.current_sun_direction;
        }
        if (glm::length(render_info.constant.sun_direction) > 0)
            render_info.constant.sun_direction = glm::normalize(render_info.constant.sun_direction);

        // Quake sets fov assuming a 4x3 screen :D
        // Set the stuff in quake and let quake compute the fov.
        render_info.constant.fov = r_refdef.fov_x;
        render_info.constant.fov_tan_alpha_half = glm::tan(glm::radians(r_refdef.fov_x) / 2);
    }

    // Update uniform data
    {
        if (render_info.render && sv_player) {
            // Demos do not have a player set
            render_info.uniform.player.flags = 0;
            render_info.uniform.player.flags |=
                sv_player->v.weapon == 1 ? PLAYER_FLAGS_TORCH : 0; // shotgun has torch
            render_info.uniform.player.flags |=
                sv_player->v.waterlevel >= 3 ? PLAYER_FLAGS_UNDERWATER : 0;
        } else {
            render_info.uniform.player = {0, 0, 0, 0};
        }
        render_info.uniform.frame = frame - last_worldspawn_frame;
        render_info.uniform.prev_cam_x_mu_sx = render_info.uniform.cam_x_mu_t;
        render_info.uniform.prev_cam_w_mu_sy = render_info.uniform.cam_w;
        render_info.uniform.prev_cam_u_mu_sz = render_info.uniform.cam_u;
        float rgt[3];
        AngleVectors(r_refdef.viewangles, &render_info.uniform.cam_w.x, rgt,
                     &render_info.uniform.cam_u.x);
        render_info.uniform.cam_x_mu_t = glm::vec4(*merian::as_vec3(r_refdef.vieworg), 1);
        render_info.uniform.sky.fill(notexture->texnum);
        if (!render_info.render) {
            render_info.uniform.sky.fill(notexture->texnum);
        } else if (skybox_name[0]) {
            for (int i = 0; i < 6; i++)
                render_info.uniform.sky[i] = skybox_textures[i]->texnum;
        } else if (solidskytexture) {
            render_info.uniform.sky[0] = solidskytexture->texnum;
            render_info.uniform.sky[1] = alphaskytexture->texnum;
            render_info.uniform.sky[2] = static_cast<uint16_t>(-1u);
        }

        if (mu_t_s_overwrite) {
            render_info.uniform.cam_x_mu_t.a = mu_t;
            render_info.uniform.prev_cam_x_mu_sx.a = mu_s_div_mu_t.r * mu_t;
            render_info.uniform.prev_cam_w_mu_sy.a = mu_s_div_mu_t.g * mu_t;
            render_info.uniform.prev_cam_u_mu_sz.a = mu_s_div_mu_t.b * mu_t;
        } else {
            render_info.uniform.cam_x_mu_t.a = Fog_GetDensity();
            render_info.uniform.cam_x_mu_t.a *= render_info.uniform.cam_x_mu_t.a;
            render_info.uniform.cam_x_mu_t.a *= 0.1;

            const float* fog_color = Fog_GetColor();
            render_info.uniform.prev_cam_x_mu_sx.a =
                fog_color[0] * render_info.uniform.cam_x_mu_t.a;
            render_info.uniform.prev_cam_w_mu_sy.a =
                fog_color[1] * render_info.uniform.cam_x_mu_t.a;
            render_info.uniform.prev_cam_u_mu_sz.a =
                fog_color[2] * render_info.uniform.cam_x_mu_t.a;
        }
        {
            const float time_diff = cl.time - render_info.uniform.cl_time;
            if (time_diff > 0)
                render_info.uniform.cam_w.a = time_diff;
            else
                render_info.uniform.cam_w.a = 1.0;
        }
        render_info.uniform.cl_time = cl.time;
    }

    if (update_gamestate && key_dest == key_game) {
        controller->request_raw_mouse_input(true);
    } else {
        controller->request_raw_mouse_input(false);
    }

    if (stop_after_worldspawn >= 0 &&
        render_info.uniform.frame == (uint64_t)stop_after_worldspawn) {
        update_gamestate = false;
        if (rebuild_after_stop)
            run.request_reconnect();
    }

    io[con_render_info] = std::make_shared<QuakeRenderInfo>(render_info);
    if (render_info.constant_data_update) {
        render_info.constant_data_update = false;
    }

    frame++;
}

void QuakeNode::update_static_geo(const vk::CommandBuffer& cmd) {
    std::vector<RTGeometry> old_static_geo = static_geo;
    static_geo.clear();

    vtx.clear();
    prev_vtx.clear();
    idx.clear();
    ext.clear();

    add_geo_brush(cl_entities, cl_entities->model, vtx, prev_vtx, idx, ext, 1);
    SPDLOG_DEBUG("static opaque geo: vtx size: {} idx size: {} ext size: {}", vtx.size(),
                 idx.size(), ext.size());

    if (!idx.empty()) {
        RTGeometry old_geo = old_static_geo.size() > 0 ? old_static_geo[0] : RTGeometry();
        static_geo.emplace_back(
            get_rt_geometry(allocator, cmd, vtx, vtx, idx, ext, old_geo,
                            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess));
        static_geo.back().instance_flags =
            vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise |
            vk::GeometryInstanceFlagBitsKHR::eForceOpaque;
    }

    vtx.clear();
    prev_vtx.clear();
    idx.clear();
    ext.clear();

    add_geo_brush(cl_entities, cl_entities->model, vtx, prev_vtx, idx, ext, 2);
    SPDLOG_DEBUG("static non-opaque geo: vtx size: {} idx size: {} ext size: {}", vtx.size(),
                 idx.size(), ext.size());
    if (!idx.empty()) {
        RTGeometry old_geo = old_static_geo.size() > 1 ? old_static_geo[1] : RTGeometry();
        static_geo.emplace_back(
            get_rt_geometry(allocator, cmd, vtx, vtx, idx, ext, old_geo,
                            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess));
        static_geo.back().instance_flags =
            vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
    }
}

void QuakeNode::update_dynamic_geo(const vk::CommandBuffer& cmd,
                                   const merian::ProfilerHandle& profiler) {
    vtx.clear();
    prev_vtx.clear();
    idx.clear();
    ext.clear();

    const uint32_t number_tasks = context->thread_pool.size();
    std::vector<std::vector<float>> thread_dynamic_vtx(number_tasks);
    std::vector<std::vector<float>> thread_dynamic_prev_vtx(number_tasks);
    std::vector<std::vector<uint32_t>> thread_dynamic_idx(number_tasks);
    std::vector<std::vector<VertexExtraData>> thread_dynamic_ext(number_tasks);

    {
        MERIAN_PROFILE_SCOPE(profiler, "parallel transform");
        std::future<void> future = context->thread_pool.submit<void>([&]() {
            if (playermodel == 1) {
                add_geo(&cl.viewent, vtx, prev_vtx, idx, ext);
            } else if (playermodel == 2) {
                add_geo(&cl.viewent, vtx, prev_vtx, idx, ext);
                add_geo(&cl_entities[cl.viewentity], vtx, prev_vtx, idx, ext);
            }
            add_particles(vtx, prev_vtx, idx, ext, texnum_blood, texnum_explosion,
                          reproducible_renders, render_info.uniform.cl_time);
        });

        merian::parallel_for(
            std::max(cl_numvisedicts, cl.num_statics),
            [&](uint32_t index, uint32_t thread_index) {
                if (index < (uint32_t)cl_numvisedicts)
                    add_geo(cl_visedicts[index], thread_dynamic_vtx[thread_index],
                            thread_dynamic_prev_vtx[thread_index], thread_dynamic_idx[thread_index],
                            thread_dynamic_ext[thread_index]);
                if (index < (uint32_t)cl.num_statics)
                    add_geo(cl_static_entities + index, thread_dynamic_vtx[thread_index],
                            thread_dynamic_prev_vtx[thread_index], thread_dynamic_idx[thread_index],
                            thread_dynamic_ext[thread_index]);
            },
            context->thread_pool, number_tasks);

        future.get();
    }

    {
        MERIAN_PROFILE_SCOPE(profiler, "merge");
        for (uint32_t i = 0; i < number_tasks; i++) {
            uint32_t old_vtx_count = vtx.size() / 3;

            merian::raw_copy_back(vtx, thread_dynamic_vtx[i]);
            merian::raw_copy_back(prev_vtx, thread_dynamic_prev_vtx[i]);
            merian::raw_copy_back(ext, thread_dynamic_ext[i]);

            uint32_t old_idx_size = idx.size();
            idx.resize(old_idx_size + thread_dynamic_idx[i].size());
            for (uint32_t j = 0; j < thread_dynamic_idx[i].size(); j++) {
                idx[old_idx_size + j] = old_vtx_count + thread_dynamic_idx[i][j];
            }
        }
    }

    // for (int i=1 ; i<MAX_MODELS ; i++)
    //     add_geo(cl.model_precache+i, p->vtx + 3*vtx_cnt, p->idx + idx_cnt, 0, &vtx_cnt,
    //     &idx_cnt);
    // for (int i=0; i<cl_max_edicts; i++)
    //     add_geo(cl_entities+i, p->vtx + 3*vtx_cnt, p->idx + idx_cnt, 0, &vtx_cnt, &idx_cnt);

    SPDLOG_TRACE("dynamic geo: vtx size: {} idx size: {} ext size: {}", dynamic_vtx.size(),
                 dynamic_idx.size(), dynamic_ext.size());

    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "upload / copy");
        std::vector<RTGeometry> old_dynamic_geo = dynamic_geo;
        dynamic_geo.clear();
        if (!idx.empty()) {
            RTGeometry old_geo = old_dynamic_geo.size() > 0 ? old_dynamic_geo[0] : RTGeometry();
            dynamic_geo.emplace_back(
                get_rt_geometry(allocator, cmd, vtx, prev_vtx, idx, ext, old_geo,
                                vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
                                    vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess));
            dynamic_geo.back().instance_flags =
                vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
        }
    }
}

void QuakeNode::update_as(const vk::CommandBuffer& cmd, const merian_nodes::NodeIO& io) {
    std::shared_ptr<merian_nodes::DeviceASBuilder::TlasBuildInfo> tlas_info =
        std::make_shared<merian_nodes::DeviceASBuilder::TlasBuildInfo>(
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess);

    assert(static_geo.size() + dynamic_geo.size() < MAX_GEOMETRIES);

    uint32_t instance_index = 0;
    for (auto& geo_vec : {static_geo, dynamic_geo}) {
        for (const RTGeometry& geo : geo_vec) {
            // clang-format off
            tlas_info->add_instance(geo.blas_info, geo.instance_flags, instance_index);
            io[con_vtx].set(instance_index, geo.vtx, cmd, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
            io[con_prev_vtx].set(instance_index, geo.prev_vtx, cmd, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
            io[con_idx].set(instance_index, geo.idx, cmd, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
            io[con_ext].set(instance_index, geo.ext, cmd, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
            instance_index++;
            // clang-format on
        }
    }

    io[con_tlas_info] = tlas_info;
}

QuakeNode::NodeStatusFlags QuakeNode::properties(merian::Properties& config) {
    const bool old_overwrite_sun = overwrite_sun;
    const glm::vec3 old_overwrite_sun_dir = overwrite_sun_dir;
    const glm::vec3 old_overwrite_sun_col = overwrite_sun_col;

    config.st_separate("General");
    config.config_bool("gamestate update", update_gamestate);
    update_gamestate |= frame == 0;

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
    config.config_float("force timediff (ms)", force_timediff,
                        "For reference renders and video outputs.");
    config.config_bool("reproducible renders", reproducible_renders,
                       "e.g. disables random behavior");

    config.st_separate("Debug / Info");
    config.config_bool("overwrite sun", overwrite_sun);
    if (overwrite_sun) {
        config.config_float3("sun dir", &overwrite_sun_dir.x);
        config.config_float3("sun col", &overwrite_sun_col.x);
    }
    if (config.config_float("volume max t", render_info.constant.volume_max_t)) {
        render_info.constant_data_update = true;
    }
    config.config_bool("overwrite mu_t/s", mu_t_s_overwrite);
    if (mu_t_s_overwrite) {
        config.config_float("mu_t", mu_t, "", 0.000001);
        config.config_float3("mu_s / mu_t", &mu_s_div_mu_t.x);
    } else {
        config.output_text(fmt::format(
            "mu_t: {}\nmu_s: ({}, {}, {})", render_info.uniform.cam_x_mu_t.a,
            render_info.uniform.prev_cam_x_mu_sx.a, render_info.uniform.prev_cam_w_mu_sy.a,
            render_info.uniform.prev_cam_u_mu_sz.a));
    }
    config.output_text(
        fmt::format("sun direction: ({}, {}, {})\nsun color: ({}, {}, {})",
                    render_info.constant.sun_direction.x, render_info.constant.sun_direction.y,
                    render_info.constant.sun_direction.z, render_info.constant.sun_color.r,
                    render_info.constant.sun_color.g, render_info.constant.sun_color.b));
    config.output_text(fmt::format("view angles {} {} {}", r_refdef.viewangles[0],
                                   r_refdef.viewangles[1], r_refdef.viewangles[2]));
    config.output_text(fmt::format("server fps: {}", server_fps));
    config.config_options("player model", playermodel, {"none", "gun only", "full"});

    if (old_overwrite_sun != overwrite_sun || old_overwrite_sun_dir != overwrite_sun_dir ||
        old_overwrite_sun_col != overwrite_sun_col) {
        render_info.constant_data_update = true;
    }

    return {};
}
