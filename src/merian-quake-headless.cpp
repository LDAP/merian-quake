#include "merian/io/file_loader.hpp"
#include "merian/utils/input_controller.hpp"
#include "merian/utils/input_controller_dummy.hpp"
#include "merian/vk/command/ring_command_pool.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/sync/ring_fences.hpp"
#include "merian/vk/utils/profiler.hpp"

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_ray_query.hpp"

#include <spdlog/spdlog.h>

#include "configuration.hpp"
#include "processing_graph.hpp"

#include <csignal>

struct FrameData {
    merian::StagingMemoryManager::SetID staging_set_id{};
    merian::ProfilerHandle profiler{};
    merian::GraphFrameData graph_frame_data{};
};

std::atomic_bool stop;

static void signal_handler(int signal) {
    SPDLOG_INFO("SIGINT/TERM ({}) caught. Shutting down", signal);
    stop.store(true);
}

int main(const int argc, const char** argv) {
    spdlog::set_level(spdlog::level::debug);
    merian::FileLoader loader{{"./res", "../res", MERIAN_QUAKE_RESOURCES}};

    auto resources = std::make_shared<merian::ExtensionResources>();
    auto extAS = std::make_shared<merian::ExtensionVkAccelerationStructure>();
    auto extRQ = std::make_shared<merian::ExtensionVkRayQuery>();
    std::vector<std::shared_ptr<merian::Extension>> extensions = {resources, extAS, extRQ};
    std::shared_ptr<merian::ExtensionVkDebugUtils> debug_utils;
#ifndef NDEBUG
    debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    extensions.push_back(debug_utils);
#endif

    merian::SharedContext context = merian::Context::make_context(extensions, "Quake");
    auto alloc = resources->resource_allocator();
    auto queue = context->get_queue_GCT();

    std::shared_ptr<merian::InputController> controller =
        std::make_shared<merian::DummyInputController>();
    auto ring_fences = make_shared<merian::RingFences<2, FrameData>>(context);

    ProcessingGraph graph(argc, argv, context, alloc, queue, loader, ring_fences->ring_size(),
                          controller);

    auto ring_cmd_pool = make_shared<merian::RingCommandPool<>>(context, queue);

    merian::Profiler::Report report;
    bool clear_profiler = false;
    merian::Stopwatch report_intervall;

    ConfigurationManager config_manager(graph, loader);
    config_manager.load();

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    while (!stop.load()) {
        auto& frame_data = ring_fences->next_cycle_wait_and_get();

        if (!frame_data.user_data.profiler) {
            frame_data.user_data.profiler = std::make_shared<merian::Profiler>(context, queue);
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
        frame_data.user_data.profiler->cmd_reset(cmd, clear_profiler);

        auto& run = graph.get().cmd_run(cmd, frame_data.user_data.graph_frame_data,
                                        frame_data.user_data.profiler);

        frame_data.user_data.staging_set_id = alloc->getStaging()->finalizeResourceSet();
        cmd_pool->end_all();
        queue->submit(cmd_pool, frame_data.fence, run.get_signal_semaphore(),
                      run.get_wait_semaphores(), run.get_wait_stages());
        run.execute_callbacks(queue);
    }

    config_manager.store();
}
