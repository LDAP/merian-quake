#include "imgui.h"
#include "merian-nodes/nodes/glfw_window/glfw_window.hpp"

#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/configuration_imgui.hpp"
#include "merian/utils/input_controller_glfw.hpp"
#include "merian/vk/command/ring_command_pool.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_glfw.hpp"
#include "merian/vk/extension/extension_vk_ray_query.hpp"
#include "merian/vk/window/glfw_imgui.hpp"

#include <csignal>
#include <merian/vk/window/imgui_context.hpp>

#include "configuration.hpp"
#include "processing_graph.hpp"

std::weak_ptr<merian::GLFWWindow> weak_window;
ImFont* quake_font_sm;
ImFont* quake_font_lg;

extern "C" {

// centerstring
extern char scr_centerstring[1024];
extern float scr_centertime_start;
extern cvar_t scr_centertime;

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
        if (scr_centertime_start <= cl.time &&
            cl.time < scr_centertime_start + scr_centertime.value) {
            std::string s = scr_centerstring;
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
    if (cl.intermission == 1 && key_dest == key_game) {
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
        if (ImGui::Begin("Intermission", NULL, window_flags)) {
            ImGui::Text("Time: %d:%02d", cl.completed_time / 60, cl.completed_time % 60);
            ImGui::Text("Secrets: %d/%2d", cl.stats[STAT_SECRETS], cl.stats[STAT_TOTALSECRETS]);
            ImGui::Text("Monsters: %d/%2d", cl.stats[STAT_MONSTERS], cl.stats[STAT_TOTALMONSTERS]);
        }
        ImGui::End();
    }
    ImGui::PopFont();

    ImGui::PopStyleVar(1);
}

static void signal_handler(int signal) {
    SPDLOG_INFO("SIGINT/TERM ({}) caught. Shutting down", signal);
    if (weak_window.expired())
        return;
    glfwSetWindowShouldClose(*weak_window.lock(), GLFW_TRUE);
}

int main(const int argc, const char** argv) {
    spdlog::set_level(spdlog::level::debug);
    merian::FileLoader loader{{"./res", "../res", MERIAN_QUAKE_RESOURCES}};

    auto extGLFW = std::make_shared<merian::ExtensionVkGLFW>();
    auto resources = std::make_shared<merian::ExtensionResources>();
    auto extAS = std::make_shared<merian::ExtensionVkAccelerationStructure>();
    auto extRQ = std::make_shared<merian::ExtensionVkRayQuery>();
    std::vector<std::shared_ptr<merian::Extension>> extensions = {extGLFW, resources, extAS, extRQ};
    std::shared_ptr<merian::ExtensionVkDebugUtils> debug_utils;
#ifndef NDEBUG
    debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(true);
    extensions.push_back(debug_utils);
#endif

    merian::SharedContext context = merian::Context::make_context(extensions, "Quake");
    auto alloc = resources->resource_allocator();
    auto queue = context->get_queue_GCT();

    auto output = std::make_shared<merian_nodes::GLFWWindow>(context);
    std::shared_ptr<merian::InputController> controller =
        std::make_shared<merian::GLFWInputController>(output->get_window());
    weak_window = output->get_window();

    ProcessingGraph graph(argc, argv, context, alloc, loader, controller);
    graph.add_beauty_output("output", output, "src");

    merian::ImGuiConfiguration config;

    auto ring_cmd_pool = make_shared<merian::RingCommandPool<>>(context, queue);
    merian::ImGuiContextWrapperHandle debug_ctx = std::make_shared<merian::ImGuiContextWrapper>();
    merian::GLFWImGui imgui(context, debug_ctx, true);

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    quake_font_sm =
        io.Fonts->AddFontFromFileTTF(loader.find_file("dpquake.ttf")->string().c_str(), 26);
    quake_font_lg =
        io.Fonts->AddFontFromFileTTF(loader.find_file("dpquake.ttf")->string().c_str(), 46);

    merian::Stopwatch frametime;

    ConfigurationManager config_manager(graph, loader);
    config_manager.load();

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

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
                !(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse));
        });

    while (!output->get_window()->should_close()) {
        glfwPollEvents();
        graph.get().run();
    }

    config_manager.store();
}
