#include "hud/hud.hpp"
#include "imgui.h"
#include "merian-nodes/ab_compare/ab_compare.hpp"
#include "merian-nodes/accumulate/accumulate.hpp"
#include "merian-nodes/blit_external/blit_external.hpp"
#include "merian-nodes/blit_glfw_window/blit_glfw_window.hpp"
#include "merian-nodes/color_output/color_output.hpp"
#include "merian-nodes/exposure/exposure.hpp"
#include "merian-nodes/image/image.hpp"
#include "merian-nodes/image_write/image_write.hpp"
#include "merian-nodes/median_approx/median.hpp"
#include "merian-nodes/shadertoy_spheres/spheres.hpp"
#include "merian-nodes/svgf/svgf.hpp"
#include "merian-nodes/taa/taa.hpp"
#include "merian-nodes/tonemap/tonemap.hpp"
#include "merian-nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/configuration_imgui.hpp"
#include "merian/utils/configuration_json_dump.hpp"
#include "merian/utils/configuration_json_load.hpp"
#include "merian/utils/input_controller_glfw.hpp"
#include "merian/vk/command/ring_command_pool.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_glfw.hpp"
#include "merian/vk/extension/extension_vk_ray_query.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/sync/ring_fences.hpp"
#include "merian/vk/window/glfw_imgui.hpp"
#include "quake/quake_node.hpp"

struct FrameData {
    merian::StagingMemoryManager::SetID staging_set_id{};
    merian::ProfilerHandle profiler{};
};

int main(const int argc, const char** argv) {
    spdlog::set_level(spdlog::level::debug);
    merian::FileLoader loader{{"./res", "../res", MERIAN_QUAKE_RESOURCES}};

    auto debugUtils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    auto extGLFW = std::make_shared<merian::ExtensionVkGLFW>();
    auto resources = std::make_shared<merian::ExtensionResources>();
    auto extAS = std::make_shared<merian::ExtensionVkAccelerationStructure>();
    auto extRQ = std::make_shared<merian::ExtensionVkRayQuery>();
    std::vector<std::shared_ptr<merian::Extension>> extensions = {extGLFW, debugUtils, resources,
                                                                  extAS, extRQ};

    merian::SharedContext context = merian::Context::make_context(extensions, "Quake");
    auto alloc = resources->resource_allocator();
    auto queue = context->get_queue_GCT();
    auto [window, surface] = extGLFW->get();

    std::shared_ptr<merian::InputController> controller =
        std::make_shared<merian::GLFWInputController>(window);
    auto ring_fences = make_shared<merian::RingFences<2, FrameData>>(context);

    merian::Graph graph{context, alloc, queue, debugUtils};
    auto output =
        std::make_shared<merian::GLFWWindowNode<merian::FIT>>(context, window, surface, queue);
    // output->get_swapchain()->set_vsync(true);
    auto blue_noise = std::make_shared<merian::ImageNode>(
        alloc, "blue_noise/1024_1024/LDR_RGBA_0.png", loader, true);
    auto black = std::make_shared<merian::ColorOutputNode>(vk::Format::eR16G16B16A16Sfloat,
                                                           vk::Extent3D{1920, 1080, 1});
    auto quake = std::make_shared<QuakeNode>(context, alloc, controller, ring_fences->ring_size(),
                                             argc - 1, argv + 1);
    auto accum = std::make_shared<merian::AccumulateNode>(context, alloc);
    auto svgf = std::make_shared<merian::SVGFNode>(context, alloc);
    auto tonemap = std::make_shared<merian::TonemapNode>(context, alloc, vk::Format::eR8G8B8A8Snorm);
    auto image_writer = std::make_shared<merian::ImageWriteNode>(context, alloc, "image");
    auto exposure = std::make_shared<merian::ExposureNode>(context, alloc);
    auto median = std::make_shared<merian::MedianApproxNode>(context, alloc, 3);
    auto hud = std::make_shared<merian::QuakeHud>(context, alloc);

    graph.add_node("output", output);
    graph.add_node("black_color", black);
    graph.add_node("blue_noise", blue_noise);
    graph.add_node("quake", quake);
    graph.add_node("accum", accum);
    graph.add_node("denoiser", svgf);
    graph.add_node("tonemap", tonemap);
    graph.add_node("image writer", image_writer);
    graph.add_node("exposure", exposure);
    graph.add_node("median variance", median);
    graph.add_node("hud", hud);


    graph.connect_image(blue_noise, quake, 0, 0);

    graph.connect_image(accum, accum, 0, 0); // feedback
    graph.connect_image(accum, accum, 1, 1);
    graph.connect_image(quake, accum, 0, 2); // irr
    graph.connect_image(quake, accum, 2, 3); // mv
    graph.connect_image(quake, accum, 4, 4); // moments
    graph.connect_buffer(quake, accum, 3, 0); // gbuffer
    graph.connect_buffer(quake, accum, 3, 1);

    graph.connect_buffer(quake, quake, 3, 1); // gbuf

    graph.connect_image(svgf, quake, 0, 1); // prev final image (with variance)
    graph.connect_buffer(median, quake, 0, 0);

    graph.connect_image(svgf, svgf, 0, 0);  // feedback
    graph.connect_image(accum, svgf, 0, 1); // irr
    graph.connect_image(accum, svgf, 1, 2); // moments
    graph.connect_image(quake, svgf, 1, 3); // albedo
    graph.connect_image(quake, svgf, 2, 4); // mv
    graph.connect_image(svgf, exposure, 0, 0);
    graph.connect_image(svgf, median, 0, 0);
    graph.connect_buffer(quake, svgf, 3, 0); // gbuffer
    graph.connect_buffer(quake, svgf, 3, 1);

    graph.connect_image(exposure, tonemap, 0, 0);
    //graph.connect_image(tonemap, output, 0, 0);
    graph.connect_image(tonemap, hud, 0, 0);
    graph.connect_image(hud, output, 0, 0);

    //  debug output
    //graph.connect_image(quake, output, 5, 0);

    graph.connect_image(svgf, image_writer, 0, 0);

    merian::ImGuiConfiguration config;

    auto ring_cmd_pool =
        make_shared<merian::RingCommandPool<>>(context, context->queue_family_idx_GCT);
    merian::GLFWImGui imgui(context, true);
    merian::Profiler::Report report;
    bool clear_profiler = false;
    merian::Stopwatch report_intervall;
    merian::Stopwatch frametime;
    auto load = merian::JSONLoadConfiguration("config.json");
    graph.get_configuration(load);

    while (!glfwWindowShouldClose(*window)) {
        auto& frame_data = ring_fences->next_cycle_wait_and_get();

        if (!frame_data.user_data.profiler) {
            frame_data.user_data.profiler = std::make_shared<merian::Profiler>(context);
        } else {
            frame_data.user_data.profiler->collect();
            if (report_intervall.millis() > 100) {
                report = frame_data.user_data.profiler->get_report();
                clear_profiler = true;
                report_intervall.reset();
            } else {
                clear_profiler = false;
            }
        }

        alloc->getStaging()->releaseResourceSet(frame_data.user_data.staging_set_id);
        auto cmd_pool = ring_cmd_pool->set_cycle();
        auto cmd = cmd_pool->create_and_begin();
        glfwPollEvents();
        frame_data.user_data.profiler->cmd_reset(cmd, clear_profiler);

        auto& run = graph.cmd_run(cmd, frame_data.user_data.profiler);
        // MERIAN_PROFILE_SCOPE_GPU(frame_data.user_data.profiler, cmd, "frame");

        if (output->current_aquire_result().has_value()) {
            imgui.new_frame(cmd, *window, output->current_aquire_result().value());

            const double frametime_ms = frametime.millis();
            frametime.reset();
            ImGui::Begin(fmt::format("Quake Debug ({:.02f}ms, {:.02f} fps)###DebugWindow",
                                     frametime_ms, 1000 / frametime_ms)
                             .c_str());

            frame_data.user_data.profiler->get_report_imgui(report);
            graph.get_configuration(config);

            ImGui::End();

            imgui.render(cmd);
            controller->set_active(
                !(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse));
        }

        frame_data.user_data.staging_set_id = alloc->getStaging()->finalizeResourceSet();
        cmd_pool->end_all();
        queue->submit(cmd_pool, frame_data.fence, run.get_signal_semaphore(),
                      run.get_wait_semaphores(), run.get_wait_stages());
        run.execute_callbacks(queue);
    }

    auto dump = merian::JSONDumpConfiguration("config.json");
    graph.get_configuration(dump);
}
