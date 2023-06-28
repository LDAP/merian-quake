#include "merian-nodes/ab_compare/ab_compare.hpp"
#include "merian-nodes/blit_external/blit_external.hpp"
#include "merian-nodes/blit_glfw_window/blit_glfw_window.hpp"
#include "merian-nodes/color_output/color_output.hpp"
#include "merian-nodes/image/image.hpp"
#include "merian-nodes/shadertoy_spheres/spheres.hpp"
#include "merian-nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"
#include "merian/io/file_loader.hpp"
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
#include "quake/quake_node.hpp"

struct FrameData {
    merian::StagingMemoryManager::SetID staging_set_id{};
};

int main() {
    spdlog::set_level(spdlog::level::debug);
    merian::FileLoader loader{{"/", "./", "./build", "./res", "./res/shaders"}};

    merian::ExtensionVkDebugUtils debugUtils;
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

    merian::Graph graph{context, alloc, queue};
    auto output =
        std::make_shared<merian::GLFWWindowNode<merian::FIT>>(context, window, surface, queue);
    output->get_swapchain()->set_vsync(true);
    auto blue_noise = std::make_shared<merian::ImageNode>(
        alloc, "blue_noise/1024_1024/LDR_RGBA_0.png", loader, true);
    auto black_color = std::make_shared<merian::ColorOutputNode>(
        vk::Format::eR16G16B16A16Sfloat, vk::ClearColorValue{}, vk::Extent3D{1920, 1080, 1});
    auto quake = std::make_shared<QuakeNode>(context, alloc);
    graph.add_node("output", output);
    graph.add_node("black_color", black_color);
    graph.add_node("blue_noise", blue_noise);
    graph.add_node("quake", quake);

    graph.connect_image(quake, quake, 2, 0); // gbuffer
    graph.connect_image(quake, quake, 3, 3); // nee
    graph.connect_image(black_color, quake, 0, 1);
    graph.connect_image(blue_noise, quake, 0, 2);

    graph.connect_image(quake, output, 1, 0);

    auto ring_cmd_pool =
        make_shared<merian::RingCommandPool<>>(context, context->queue_family_idx_GCT);
    auto ring_fences = make_shared<merian::RingFences<1, FrameData>>(context);
    uint64_t frames = 50;
    uint64_t frame = 0;
    quake->queue_command("game ad");
    quake->queue_command("map ad_tears");
    while (!glfwWindowShouldClose(*window)) {
        // if (frame++ >= frames)
        //     break;

        auto& frame_data = ring_fences->next_cycle_wait_and_get();
        alloc->getStaging()->releaseResourceSet(frame_data.user_data.staging_set_id);
        auto cmd_pool = ring_cmd_pool->set_cycle();
        auto cmd = cmd_pool->create_and_begin();
        glfwPollEvents();

        auto& run = graph.cmd_run(cmd);

        frame_data.user_data.staging_set_id = alloc->getStaging()->finalizeResourceSet();
        cmd_pool->end_all();
        queue->submit(cmd_pool, frame_data.fence, run.get_signal_semaphore(),
                      run.get_wait_semaphores(), run.get_wait_stages());
        run.execute_callbacks(queue);
    }
}
