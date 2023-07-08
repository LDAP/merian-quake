#include "imgui.h"
#include "merian-nodes/ab_compare/ab_compare.hpp"
#include "merian-nodes/blit_external/blit_external.hpp"
#include "merian-nodes/blit_glfw_window/blit_glfw_window.hpp"
#include "merian-nodes/color_output/color_output.hpp"
#include "merian-nodes/image/image.hpp"
#include "merian-nodes/shadertoy_spheres/spheres.hpp"
#include "merian-nodes/taa/taa.hpp"
#include "merian-nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"
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

int main() {
    spdlog::set_level(spdlog::level::debug);
    merian::FileLoader loader{{"/", "./", "./build", "./res", "./res/shaders"}};

    merian::ExtensionVkDebugUtils debugUtils(true);
    merian::ExtensionVkGLFW extGLFW;
    merian::ExtensionResources resources;
    merian::ExtensionVkAccelerationStructure extAS;
    merian::ExtensionVkRayQuery extRQ;
    std::vector<merian::Extension*> extensions = {&extGLFW, &debugUtils, &resources, &extAS,
                                                  &extRQ};

    merian::SharedContext context = merian::Context::make_context(extensions, "merian-cornell-box");
    auto alloc = resources.resource_allocator();
    auto queue = context->get_queue_GCT();
    auto [window, surface] = extGLFW.get();

    std::shared_ptr<merian::InputController> controller =
        std::make_shared<merian::GLFWInputController>(window);
    auto ring_fences = make_shared<merian::RingFences<2, FrameData>>(context);

    merian::Graph graph{context, alloc, queue};
    auto output =
        std::make_shared<merian::GLFWWindowNode<merian::FIT>>(context, window, surface, queue);
    // output->get_swapchain()->set_vsync(true);
    auto blue_noise = std::make_shared<merian::ImageNode>(
        alloc, "blue_noise/1024_1024/LDR_RGBA_0.png", loader, true);
    auto black_color = std::make_shared<merian::ColorOutputNode>(
        vk::Format::eR16G16B16A16Sfloat, vk::Extent3D{1920, 1080, 1});
    auto quake = std::make_shared<QuakeNode>(context, alloc, controller, ring_fences->ring_size());
    auto taa = std::make_shared<merian::TAANode>(context, alloc);
    graph.add_node("output", output);
    graph.add_node("black_color", black_color);
    graph.add_node("blue_noise", blue_noise);
    graph.add_node("quake", quake);
    graph.add_node("taa", taa);

    graph.connect_image(quake, quake, 2, 0); // gbuffer
    graph.connect_image(quake, quake, 3, 3); // nee
    graph.connect_image(black_color, quake, 0, 1);
    graph.connect_image(blue_noise, quake, 0, 2);

    graph.connect_image(black_color, taa, 0, 2);
    graph.connect_image(taa, taa, 0, 1);
    graph.connect_image(quake, taa, 1, 0);

    graph.connect_image(taa, output, 0, 0);
    merian::ImGuiConfiguration config;

    auto ring_cmd_pool =
        make_shared<merian::RingCommandPool<>>(context, context->queue_family_idx_GCT);
    // quake->queue_command("game SlayerTest");
    // quake->queue_command("map st1m1");
    quake->queue_command("game ad");
    quake->queue_command("map ad_tears");
    // quake->queue_command("map e1m6");
    // quake->queue_command("map e1m1");
    merian::GLFWImGui imgui(context, true);
    merian::Profiler::Report report;
    bool clear_profiler = false;
    merian::Stopwatch sw;
    while (!glfwWindowShouldClose(*window)) {
        auto& frame_data = ring_fences->next_cycle_wait_and_get();

        if (!frame_data.user_data.profiler) {
            frame_data.user_data.profiler = std::make_shared<merian::Profiler>(context);
        } else {
            if (sw.millis() > 100) {
                frame_data.user_data.profiler->collect();
                report = frame_data.user_data.profiler->get_report();
                clear_profiler = true;
                sw.reset();
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

            ImGui::Begin("Quake Debug");

            frame_data.user_data.profiler->get_report_imgui(report);
            graph.get_configuration(config);

            ImGui::End();

            imgui.render(cmd);
            controller->set_active(!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse));
        }

        frame_data.user_data.staging_set_id = alloc->getStaging()->finalizeResourceSet();
        cmd_pool->end_all();
        queue->submit(cmd_pool, frame_data.fence, run.get_signal_semaphore(),
                      run.get_wait_semaphores(), run.get_wait_stages());
        run.execute_callbacks(queue);
    }
}
