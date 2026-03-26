#include "gbuffer/gbuffer.hpp"
#include "imgui.h"
#include "merian-nodes/merian_nodes_extension.hpp"
#include "merian-nodes/nodes/window/window_node.hpp"

#include "merian-nodes/graph/graph.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/imgui_spdlog_sink.hpp"
#include "merian/utils/input_controller_dummy.hpp"
#include "merian/utils/properties_imgui.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/imgui/imgui_merian_window_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"

#include <csignal>

#include "configuration.hpp"

#include "game/quake_node.hpp"
#include "hud/hud.hpp"
#include "render_mcpg/render_mcpg.hpp"
#include "render_restir/renderer_restir.hpp"
#include "render_ssmm/render_ssmm.hpp"

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
    std::shared_ptr<merian::ImguiSpdlogSink> imgui_spdlog =
        std::make_shared<merian::ImguiSpdlogSink>();
    spdlog::default_logger()->sinks().push_back(imgui_spdlog);

    std::vector<std::string> context_extensions = {merian::MerianNodesExtension::name};

#ifndef NDEBUG
    context_extensions.push_back(merian::ExtensionVkValidationLayers::name);
#endif

    // Prepare additional search paths for shader includes and data files
    std::vector<std::filesystem::path> additional_search_paths;

    std::optional<std::filesystem::path> dev_data_dir =
        merian::FileLoader::search_cwd_parents("res");
    if (dev_data_dir) {
        additional_search_paths.push_back(*dev_data_dir);
    }
    additional_search_paths.push_back(MERIAN_QUAKE_DATA_DIR);
    if (const auto prefix = merian::FileLoader::portable_prefix(); prefix) {
        additional_search_paths.push_back(*prefix / merian::FileLoader::install_datadir_name() /
                                          std::filesystem::path(MERIAN_QUAKE_PROJECT_NAME));
    }
    if (const auto prefix = merian::FileLoader::install_prefix(); prefix) {
        additional_search_paths.push_back(*prefix / merian::FileLoader::install_datadir_name() /
                                          std::filesystem::path(MERIAN_QUAKE_PROJECT_NAME));
    }

    merian::NodeRegistry& registry = merian::NodeRegistry::get_instance();

    registry.register_node_type<QuakeNode>(
        merian::NodeRegistry::NodeTypeInfo{"Quake", "Extract geometry info from Quake", [=]() {
                                               auto quake_node = std::make_shared<QuakeNode>();
                                               quake_node->set_cmd_args(argc - 1, argv + 1);
                                               return quake_node;
                                           }});
    registry.register_node_type<merian::QuakeHud>("Hud",
                                                  "Show gamestate and apply screen effects.");
    registry.register_node_type<GBuffer>("GBuffer", "Generates the GBuffer for Quake.");
    registry.register_node_type<RendererMarkovChain>(
        "Renderer (MCPG)", "Renders a scene using Markov Chain Path Guiding.");
    registry.register_node_type<RendererRESTIR>("Renderer (RESTIR)",
                                                "Renders a scene using RESTIR.");
    registry.register_node_type<RendererSSMM>(
        "Renderer (SSMM)",
        "Renders s scene using screen-space mixture models by Dittebrandt et al. (2023)");

    merian::ContextCreateInfo create_info{
        .context_extensions = context_extensions,
        .additional_search_paths = additional_search_paths,
        .application_name = "merian-quake",
    };
    const merian::ContextHandle context = merian::Context::create(create_info);
    auto resources = context->get_context_extension<merian::ExtensionResources>();
    auto alloc = resources->resource_allocator();
    auto queue = context->get_queue_GCT();

    merian::GraphHandle graph =
        context->get_context_extension<merian::MerianNodesExtension>()->create({context, alloc});
    // this also creates all nodes in the graph.
    ConfigurationManager config_manager(*graph, *context->get_file_loader());
    config_manager.load();

    std::shared_ptr<merian::WindowNode> output =
        graph->find_node_for_identifier_and_type<merian::WindowNode>("output");
    std::shared_ptr<QuakeNode> quake =
        graph->find_node_for_identifier_and_type<QuakeNode>("Quake 0");

    merian::InputControllerHandle controller = std::make_shared<merian::DummyInputController>();
    if (quake) {
        quake->set_controller(controller);
    }

    auto debug_ctx = std::make_shared<merian::ImGuiContext>();
    std::shared_ptr<merian::ImGuiMerianBackend> imgui_backend =
        std::make_shared<merian::ImGuiMerianBackend>(debug_ctx);
    auto imgui_renderer = std::make_shared<merian::ImGuiRenderer>(context, alloc, debug_ctx);
    debug_ctx->with_context([&] {
        ImFontConfig quake_cfg;
        quake_cfg.PixelSnapH = true;
        ImGuiIO& io = ImGui::GetIO();
        quake_font_sm = io.Fonts->AddFontFromFileTTF(
            context->get_file_loader()->find_file("dpquake.ttf")->string().c_str(), 26, &quake_cfg);
        quake_font_lg = io.Fonts->AddFontFromFileTTF(
            context->get_file_loader()->find_file("dpquake.ttf")->string().c_str(), 46, &quake_cfg);
    });

    if (output) {
        output->set_on_window_created([&](const merian::WindowHandle& win) {
            if (quake) {
                controller = win->get_input_controller();
                quake->set_controller(controller);
            }
            imgui_backend.reset();
            imgui_backend = std::make_shared<merian::ImGuiMerianWindowBackend>(debug_ctx, win);
        });
    }

    merian::ImGuiProperties config;
    merian::Stopwatch frametime;
    if (output) {
        output->set_on_blit_completed([&](const merian::CommandBufferHandle& cmd,
                                          const merian::SwapchainAcquireResult& aquire_result,
                                          const merian::ProfilerHandle& profiler) {
            const double frametime_ms = frametime.millis();
            frametime.reset();
            imgui_backend->new_frame(static_cast<float>(frametime_ms / 1000.0));

            const float alpha = controller->is_mouse_grabbed() ? 0.2f : 1.0f;

            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
            ImGui::Begin(fmt::format("Quake Debug ({:.02f}ms, {:.02f} fps)###DebugWindow",
                                     frametime_ms, 1000 / frametime_ms)
                             .c_str(),
                         NULL, ImGuiWindowFlags_NoFocusOnAppearing);

            config_manager.get(config);
            if (ImGui::TreeNodeEx("Log", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed,
                                  "%s", "Log")) {
                imgui_spdlog->imgui_draw_log();
                ImGui::TreePop();
            }

            ImGui::End();
            ImGui::PopStyleVar();

            QuakeMessageOverlay();

            imgui_renderer->render(cmd, aquire_result.image_view, profiler);
        });
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    while (!stop) {
        graph->run();
    }

    config_manager.store();
}
