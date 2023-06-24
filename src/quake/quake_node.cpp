#include "quake/quake_node.hpp"
#include "merian/utils/string.hpp"

struct QuakeData {
    // The first quake node sets this
    // If this is not null we won't allow new
    // Quake nodes
    QuakeNode* node{nullptr};
    quakeparms_t params;
};

extern "C" {

#include "bgmusic.h"
#include "quakedef.h"

// Quake uses lots of static global variables,
// so we need to do that to
// e.g. to make sure there is only one QuakeNode.
static QuakeData quake_data;
// from r_alias.c
extern float r_avertexnormals[162][3];
extern particle_t* active_particles;
extern cvar_t cl_maxpitch; // johnfitz -- variable pitch clamping
extern cvar_t cl_minpitch; // johnfitz -- variable pitch clamping
}
// CALLBACKS from within Quake --------------------------------------------------------------------

// called each time a new map is (re)loaded
extern "C" void QS_worldspawn() {
    quake_data.node->QS_worldspawn();
}

// called when a texture should be loaded
extern "C" void QS_texture_load(gltexture_t* glt, uint32_t* data) {
    quake_data.node->QS_texture_load(glt, data);
}

// called from within qs, pretty much a copy from in_sdl.c:
extern "C" void IN_Move(usercmd_t* cmd) {
    quake_data.node->IN_Move(cmd);
}

// QuakeNode --------------------------------------------------------------------------------------

QuakeNode::QuakeNode(const merian::ResourceAllocatorHandle& allocator, const char* base_dir)
    : allocator(allocator) {

    if (quake_data.node) {
        throw std::runtime_error{"Only one quake node can be created."};
    }
    quake_data.node = this;
    host_parms = &quake_data.params;

    const char* argv[] = {
        "quakespasm",
        "-basedir",
        base_dir,
        "+skill",
        "2",
        "-game",
        "ad",
        "+map",
        "e1m1",
        "-game",
        "SlayerTest", // needs different particle rules! also overbright
                      // and lack of alpha?
        "+map",
        "e1m2b",
        "+map",
        "ep1m1",
        "+map",
        "e1m1b",
        "+map",
        "st1m1",
    };
    const int argc = 9;

    quake_data.params.basedir = base_dir; // does not work
    quake_data.params.argc = argc;
    quake_data.params.argv = (char**)argv;
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

    Sys_Printf("Host_Init\n");
    Host_Init(); // this one has a lot of meat! we'll need to cut it short i suppose

    // S_BlockSound(); // only start when grabbing
    // close menu because we don't have an esc key:
    key_dest = key_game;
    m_state = m_none;
    IN_Activate();
}

QuakeNode::~QuakeNode() {
    free(quake_data.params.membase);
}

// -------------------------------------------------------------------------------------------

void QuakeNode::QS_worldspawn() {
    // Called after ~5 frames when textures are loaded.
    SPDLOG_DEBUG("worldspawn");
    worldspawn = true;
}

void QuakeNode::QS_texture_load(gltexture_t* glt, uint32_t* data) {
    // LOG -----------------------------------------------

    std::string source = strcmp(glt->source_file, "") == 0 ? "memory" : glt->source_file;
    SPDLOG_DEBUG("texture_load {} {} {}x{} from {}, frame: {}", glt->texnum, glt->name, glt->width,
                 glt->height, source, glt->visframe);

    // STORE SOME TEXTURE IDs ----------------------------

    // HACK: for blood patch
    if (!strcmp(glt->name, "progs/gib_1.mdl:frame0"))
        texnum_blood = glt->texnum;
    // HACK: for sparks and for emissive rocket particle trails
    if (!strcmp(glt->name, "progs/s_exp_big.spr:frame10"))
        texnum_explosion = glt->texnum;

    // classic quake sky
    if (merian::ends_with(glt->name, "_front"))
        texnum_skybox[1] = glt->texnum;
    if (merian::ends_with(glt->name, "_back"))
        texnum_skybox[0] = glt->texnum;

    // full featured cube map/arcane dimensions
    if (merian::starts_with(glt->name, "gfx/env/")) {
        if (merian::ends_with(glt->name, "_rt"))
            texnum_skybox[0] = glt->texnum;
        if (merian::ends_with(glt->name, "_bk"))
            texnum_skybox[1] = glt->texnum;
        if (merian::ends_with(glt->name, "_lf"))
            texnum_skybox[2] = glt->texnum;
        if (merian::ends_with(glt->name, "_ft"))
            texnum_skybox[3] = glt->texnum;
        if (merian::ends_with(glt->name, "_up"))
            texnum_skybox[4] = glt->texnum;
        if (merian::ends_with(glt->name, "_dn"))
            texnum_skybox[5] = glt->texnum;
    }

    // ALLOCATE ----------------------------

    // We store the texture on system memory for now
    // and upload in cmd_process later
    QuakeTexture texture(glt, data);
    // If we replace an existing texture the old texture is automatically freed.
    // TODO: Multiple frames in flight? (maybe use glt->visframe, semaphore,...?)
    textures.insert(std::make_pair(glt->texnum, std::move(texture)));
    pending_uploads.push(glt->texnum);
}

// pretty much a copy from in_sdl.c:
void QuakeNode::IN_Move(usercmd_t* cmd) {
    SPDLOG_DEBUG("move");

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

// -------------------------------------------------------------------------------------------

void QuakeNode::cmd_build(const vk::CommandBuffer& cmd,
                          const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                          const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                          const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                          const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs) {}

void QuakeNode::cmd_process(const vk::CommandBuffer& cmd,
                            merian::GraphRun& run,
                            const uint32_t set_index,
                            const std::vector<merian::ImageHandle>& image_inputs,
                            const std::vector<merian::BufferHandle>& buffer_inputs,
                            const std::vector<merian::ImageHandle>& image_outputs,
                            const std::vector<merian::BufferHandle>& buffer_outputs) {
    if (!pause) {
        double newtime = Sys_DoubleTime();
        double time = old_time == 0 ? 0.0 : newtime - old_time;
        Host_Frame(time);
        // init some left/right vectors also used for sound
        R_SetupView();
        old_time = newtime;
    }

    // if (frame == 0) {
    //     // careful to only do this at == 0 so sv_player (among others) will not crash
    //     const char* p_exec = nullptr; // map exec string from examples/ad.cfg
    //     if (p_exec) {
    //         Cmd_ExecuteString(p_exec, src_command);
    //         sv_player_set = 0; // just in case we loaded a map (demo, savegame)
    //     }
    // }
    
    // commit_params does here stuff with sv_player

    frame++;
}
