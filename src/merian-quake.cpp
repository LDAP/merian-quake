#include "gbuffer/gbuffer.hpp"
#include "imgui.h"
#include "merian-nodes/nodes/glfw_window/glfw_window.hpp"

#include "merian-nodes/graph/graph.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/input_controller_dummy.hpp"
#include "merian/utils/input_controller_glfw.hpp"
#include "merian/utils/properties_imgui.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "merian/vk/extension/extension_vk_ray_query.hpp"
#include "merian/vk/extension/extension_vk_ray_tracing_position_fetch.hpp"
#include "merian/vk/window/glfw_imgui.hpp"

#include <csignal>
#include <merian/vk/window/imgui_context.hpp>

#include "configuration.hpp"

#include "game/quake_node.hpp"
#include "hud/hud.hpp"
#include "render_mc/render_markovchain.hpp"
#include "render_restir/renderer_restir.hpp"

std::atomic_bool stop(false);
ImFont* quake_font_sm;
ImFont* quake_font_lg;

extern "C" {

// centerstring
extern char scr_centerstring[1024];
extern float scr_centertime_off;
extern cvar_t scr_centertime;
extern qboolean scr_drawloading;

// console notify
extern int con_linewidth;
extern char* con_text;
extern int con_current;
extern cvar_t con_notifytime;
// from console.c
#define NUM_CON_TIMES 4
extern float con_times[NUM_CON_TIMES];
}

static void QuakeMessageOverlay() {
    const ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    const ImVec2 window_pos(center.x, (center.y + 0) / 2);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);

    ImGui::PushFont(quake_font_sm);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    if (ImGui::Begin("CenterString", NULL, window_flags)) {
        if (!cl.intermission && !((scr_centertime_off <= 0 || key_dest != key_game || cl.paused))) {
            std::string s;
            s = scr_centerstring;
            // undo colored text
            for (uint32_t i = 0; i < s.size(); i++)
                s[i] &= ~128;

            merian::split(s, "\n", [](const std::string& s) {
                // hack to display centered text
                const float font_size = ImGui::CalcTextSize(s.c_str()).x;

                ImGui::Text("%s", "");
                ImGui::SameLine(ImGui::GetWindowSize().x / 2 - font_size + (font_size / 2));
                ImGui::Text("%s", s.c_str());
            });
        }
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always, ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    if (ImGui::Begin("ConsoleNotify", NULL, window_flags)) {
        // mostly from console.c
        std::string s;
        for (int i = con_current - NUM_CON_TIMES + 1; i <= con_current; i++) {
            if (i < 0)
                continue;
            float time = con_times[i % NUM_CON_TIMES];
            if (time == 0)
                continue;
            time = realtime - time;
            if (time > con_notifytime.value)
                continue;
            const char* text = con_text + (i % con_totallines) * con_linewidth;
            for (int i = 0; i < con_linewidth; i++)
                s += (text[i] & ~128);
            s += "\n";
        }
        ImGui::Text("%s", s.c_str());
    }
    ImGui::End();
    ImGui::PopFont();

    ImGui::PushFont(quake_font_lg);
    if (scr_drawloading || (cl.intermission == 1 && key_dest == key_game)) {
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
        if (ImGui::Begin("Intermission", NULL, window_flags)) {
            if (scr_drawloading) {
                ImGui::Text("Loading...");
            } else {
                ImGui::Text("Time: %d:%02d", cl.completed_time / 60, cl.completed_time % 60);
                ImGui::Text("Secrets: %d/%2d", cl.stats[STAT_SECRETS], cl.stats[STAT_TOTALSECRETS]);
                ImGui::Text("Monsters: %d/%2d", cl.stats[STAT_MONSTERS],
                            cl.stats[STAT_TOTALMONSTERS]);
            }
        }
        ImGui::End();
    }
    ImGui::PopFont();

    ImGui::PopStyleVar(1);
}

static void signal_handler(int signal) {
    SPDLOG_INFO("SIGINT/TERM ({}) caught. Shutting down", signal);
    stop.store(true);
}

int main(const int argc, const char** argv) {
    spdlog::set_level(spdlog::level::trace);

    std::shared_ptr<merian::ExtensionGLFW> ext_glfw;
    auto resources = std::make_shared<merian::ExtensionResources>();
    auto ext_as = std::make_shared<merian::ExtensionVkAccelerationStructure>();
    auto ext_rq = std::make_shared<merian::ExtensionVkRayQuery>();
    auto ext_rt_pos = std::make_shared<merian::ExtensionVkRayTracingPositionFetch>();
    auto ext_push_desc = std::make_shared<merian::ExtensionVkPushDescriptor>();
    auto ext_core = std::make_shared<merian::ExtensionVkCore>();
    auto ext_float = std::make_shared<merian::ExtensionVkFloatAtomics>();
    std::vector<std::shared_ptr<merian::Extension>> extensions = {
        resources, ext_as, ext_rq, ext_rt_pos, ext_push_desc, ext_core, ext_float};

    std::shared_ptr<merian::ExtensionVkDebugUtils> debug_utils;
#ifndef NDEBUG
    debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(true);
    extensions.push_back(debug_utils);
#endif

    if (argc == 1 || strcmp(argv[1], "--headless") != 0) {
        ext_glfw = std::make_shared<merian::ExtensionGLFW>();
        extensions.push_back(ext_glfw);
    }

    merian::ContextHandle context = merian::Context::create(extensions, "merian-quake");
    auto alloc = resources->resource_allocator();
    auto queue = context->get_queue_GCT();

    std::optional<std::filesystem::path> dev_data_dir =
        merian::FileLoader::search_cwd_parents("res");
    if (dev_data_dir) {
        context->file_loader.add_search_path(*dev_data_dir);
    }
    context->file_loader.add_search_path(MERIAN_QUAKE_DATA_DIR);

    merian_nodes::Graph<> graph(context, alloc);

    graph.get_registry().register_node<QuakeNode>(merian_nodes::NodeRegistry::NodeInfo{
        "Quake", "Extract geometry info from Quake",
        [=]() { return std::make_shared<QuakeNode>(context, alloc, argc - 1, argv + 1); }});
    graph.get_registry().register_node<merian::QuakeHud>(merian_nodes::NodeRegistry::NodeInfo{
        "Hud", "Show gamestate and apply screen effects.",
        [=]() { return std::make_shared<merian::QuakeHud>(context); }});
    graph.get_registry().register_node<RendererMarkovChain>(merian_nodes::NodeRegistry::NodeInfo{
        "Renderer (Markov Chain Raytracer)", "Renders a scene using Markov Chain Path Guiding.",
        [=]() { return std::make_shared<RendererMarkovChain>(context, alloc); }});
    graph.get_registry().register_node<RendererRESTIR>(merian_nodes::NodeRegistry::NodeInfo{
        "Renderer (RESTIR)", "Renders a scene using RESTIR.",
        [=]() { return std::make_shared<RendererRESTIR>(context, alloc); }});
    graph.get_registry().register_node<GBuffer>(
        merian_nodes::NodeRegistry::NodeInfo{"GBuffer", "Generates the GBuffer for Quake.",
                                             [=]() { return std::make_shared<GBuffer>(context); }});

    // this also creates all nodes in the graph.
    ConfigurationManager config_manager(graph, context->file_loader);
    config_manager.load();

    std::shared_ptr<merian_nodes::GLFWWindow> output =
        graph.find_node_for_identifier_and_type<merian_nodes::GLFWWindow>("output");
    std::shared_ptr<QuakeNode> quake =
        graph.find_node_for_identifier_and_type<QuakeNode>("Quake 0");

    merian::InputControllerHandle controller = std::make_shared<merian::DummyInputController>();
    if (output && quake && output->get_window()) {
        controller = std::make_shared<merian::GLFWInputController>(output->get_window());
        quake->set_controller(controller);
    }

    // TODO!
    // image_writer->set_callback([accum]() { accum->request_clear(); });
    // image_writer_volume->set_callback([volume_accum]() { volume_accum->request_clear(); });

    merian::ImGuiProperties config;
    merian::ImGuiContextWrapperHandle debug_ctx = std::make_shared<merian::ImGuiContextWrapper>();
    merian::GLFWImGui imgui(context, debug_ctx, true);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    quake_font_sm = io.Fonts->AddFontFromFileTTF(
        context->file_loader.find_file("dpquake.ttf")->string().c_str(), 26);
    quake_font_lg = io.Fonts->AddFontFromFileTTF(
        context->file_loader.find_file("dpquake.ttf")->string().c_str(), 46);
    merian::Stopwatch frametime;
    if (output) {
        output->set_on_blit_completed(
            [&](const vk::CommandBuffer& cmd, merian::SwapchainAcquireResult& aquire_result) {
                imgui.new_frame(queue, cmd, *output->get_window(), aquire_result);

                const double frametime_ms = frametime.millis();
                frametime.reset();
                ImGui::Begin(fmt::format("Quake Debug ({:.02f}ms, {:.02f} fps)###DebugWindow",
                                         frametime_ms, 1000 / frametime_ms)
                                 .c_str(),
                             NULL, ImGuiWindowFlags_NoFocusOnAppearing);

                config_manager.get(config);
                ImGui::End();

                QuakeMessageOverlay();

                imgui.render(cmd);
                controller->set_active(
                    controller->get_raw_mouse_input() ||
                    !(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse));
            });
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    graph.set_on_run_starting([](merian_nodes::GraphRun&) { glfwPollEvents(); });
    while (!(stop || (output && output->get_window() && output->get_window()->should_close()))) {
        graph.run();
    }

    config_manager.store();
}
