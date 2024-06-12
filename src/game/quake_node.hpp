#pragma once

#include "merian-nodes/connectors/any_out.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/utils/input_controller.hpp"

#include "renderer/render_markovchain.hpp"

#include <queue>

class QuakeNode : public merian_nodes::Node {
  public:
    struct QuakeRenderInfo {
        glm::vec3 sun_color{};
        glm::vec3 sun_direction{};
        bool worldspawn{};
        uint64_t last_worldspawn_frame = 0;
        uint64_t frame = 0;
    };

  public:
    QuakeNode(const merian::SharedContext& context,
              const std::shared_ptr<merian::InputController>& controller,
              const int quakespasm_argc,
              const char** quakespasm_argv,
              RendererMarkovChain* renderer);

    ~QuakeNode();

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const merian_nodes::ConnectorIOMap& output_for_input);

    void process(merian_nodes::GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian_nodes::NodeIO& io);

    NodeStatusFlags configuration(merian::Configuration& config);

    // -----------------------------------------------------

    // called from within qs
    void IN_Move(usercmd_t* cmd);

    // called when Quake wants to render
    void R_RenderScene();

    // called each time a new is (re)loaded
    void QS_worldspawn();

    // -----------------------------------------------------

    void queue_command(std::string command) {
        pending_commands.push(command);
    }

  private:
    // Graph outputs
    merian_nodes::AnyOutHandle con_render_info = merian_nodes::AnyOut::create("render_info");

    // Game thread / synchronization
    std::thread game_thread;
    std::atomic_bool game_running = true;
    merian::ConcurrentQueue<bool> sync_render;
    merian::ConcurrentQueue<bool> sync_gamestate;

    // Game state
    double old_time = 0;
    float force_timediff = 0;
    bool update_gamestate = true;
    QuakeRenderInfo render_info;

    // Input processing
    const std::shared_ptr<merian::InputController> controller;
    double mouse_oldx = 0;
    double mouse_oldy = 0;
    double mouse_x = 0;
    double mouse_y = 0;
    bool raw_mouse_was_enabled = false;

    // Quake commands
    std::array<char, 512> startup_commands_buffer = {0};
    std::queue<std::string> pending_commands;

    // Helpers for image generation
    int stop_after_worldspawn = -1;
    bool rebuild_after_stop = true;
};
