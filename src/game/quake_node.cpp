#include "quake_node.hpp"

#include "../../res/shader/config.h"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object_allocator.hpp"

#include <spdlog/spdlog.h>

QuakeNode::QuakeNode() : Node() {}

QuakeNode::~QuakeNode() {
    // Tear down the scene first so the game thread stops and Quake shuts
    // down before the resource allocator etc. go away.
    scene.reset();
}

merian::DeviceSupportInfo
QuakeNode::query_device_support(const merian::DeviceSupportQueryInfo& query_info) {
    return merian::DeviceSupportInfo::check(query_info, {"rayQuery", "accelerationStructure"});
}

void QuakeNode::initialize(const merian::ContextHandle& context,
                           const merian::ResourceAllocatorHandle& allocator) {
    assert(this->context == nullptr && "QuakeNode was initialized multiple times.");
    this->context = context;
    this->allocator = allocator;
    compile_context = merian::ShaderCompileContext::create(context);
}

std::vector<merian::OutputConnectorDescriptor>
QuakeNode::describe_outputs([[maybe_unused]] const merian::NodeIOLayout& io_layout) {
    return {{"scene", con_scene}};
}

void QuakeNode::process(merian::GraphRun& run,
                        [[maybe_unused]] const merian::DescriptorSetHandle& descriptor_set,
                        const merian::NodeIO& io) {
    const merian::CommandBufferHandle& cmd = run.get_cmd();

    // Lazy init: deferred to the first process so the
    // FrameCachingShaderObjectAllocator can be sized to iterations_in_flight,
    // mirroring GLTFSceneNode.
    if (!obj_allocator) {
        obj_allocator = std::make_shared<merian::FrameCachingShaderObjectAllocator>(
            allocator, run.get_iterations_in_flight());
        texture_manager = std::make_shared<merian::TextureManager>(
            compile_context, context, allocator, obj_allocator, MAX_GLTEXTURES);
        material_system = std::make_shared<merian::MaterialSystem>(
            compile_context, context, allocator, obj_allocator, texture_manager);
        scene = std::make_shared<merian_quake::QuakeScene>(
            compile_context, context, allocator, obj_allocator, material_system, argc, argv);
        if (pending_controller) {
            scene->set_controller(pending_controller);
            pending_controller.reset();
        }
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    scene->update(cmd, static_cast<float>(run.get_elapsed()),
                  static_cast<float>(run.get_time_delta()), run.get_total_iteration());

    io[con_scene] = std::static_pointer_cast<merian::Scene>(scene);
}

QuakeNode::NodeStatusFlags QuakeNode::properties(merian::Properties& config) {
    if (scene)
        scene->properties(config);
    return {};
}

void QuakeNode::set_cmd_args(const uint32_t argc, const char** argv) {
    this->argc = argc;
    this->argv = argv;
}

void QuakeNode::set_controller(const merian::InputControllerHandle& controller) {
    if (scene) {
        scene->set_controller(controller);
    } else {
        pending_controller = controller;
    }
}

void QuakeNode::queue_command(const std::string& command) {
    if (scene) {
        scene->queue_command(command);
    } else {
        SPDLOG_WARN("QuakeNode: queue_command before scene init dropped: {}", command);
    }
}
