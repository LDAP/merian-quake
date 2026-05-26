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
#include "game/quake_ui_node.hpp"
#include "hud/hud.hpp"
#include "render_mcpg/render_mcpg.hpp"
#include "render_restir/renderer_restir.hpp"
#include "render_ssmm/render_ssmm.hpp"

std::atomic_bool stop(false);

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
    registry.register_node_type<merian_quake::QuakeUiNode>(
        "Quake UI", "Replay Quake's 2D HUD/menu/console as a transparent overlay.");
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
    std::shared_ptr<QuakeNode> quake = graph->find_node_for_identifier_and_type<QuakeNode>("quake");

    merian::InputControllerHandle controller = std::make_shared<merian::DummyInputController>();
    if (quake) {
        quake->set_controller(controller);
    }

    auto debug_ctx = std::make_shared<merian::ImGuiContext>();
    std::shared_ptr<merian::ImGuiMerianBackend> imgui_backend =
        std::make_shared<merian::ImGuiMerianBackend>(debug_ctx);
    auto imgui_renderer = std::make_shared<merian::ImGuiRenderer>(context, alloc, debug_ctx);

    if (output) {
        output->set_on_window_created([&](const merian::WindowHandle& win) {
            if (quake) {
                controller = win->get_input_controller();
                quake->set_controller(controller, win);
            }
            imgui_backend.reset();
            imgui_backend = std::make_shared<merian::ImGuiMerianWindowBackend>(debug_ctx, win);
        });
    }

    merian::ImGuiProperties config;
    merian::Stopwatch frametime;
    if (output) {
        output->set_on_blit_completed([&](const merian::CommandBufferHandle& cmd,
                                          const merian::SwapchainAcquireResult& aquire_result) {
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

            imgui_renderer->render(cmd, aquire_result.image_view);
        });
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    while (!stop) {
        graph->run();
    }

    config_manager.store();
}
