#include "quake_node.hpp"

#include "../../res/shader/config.h"
#include "merian/shader/shader_compile_context.hpp"

#include <spdlog/spdlog.h>

QuakeNode::QuakeNode() : Node() {}

QuakeNode::~QuakeNode() {
    // Tear down the scene first so Quake shuts down before the allocator goes away.
    scene.reset();
}

merian::DeviceSupportInfo
QuakeNode::query_device_support(const merian::DeviceSupportQueryInfo& query_info) {
    return merian::DeviceSupportInfo::check(
        query_info, {"rayQuery", "accelerationStructure", "shaderInt64", "bufferDeviceAddress"});
}

void QuakeNode::initialize(const merian::ContextHandle& context,
                           const merian::ResourceAllocatorHandle& allocator) {
    assert(this->context == nullptr && "QuakeNode was initialized multiple times.");
    this->context = context;
    this->allocator = allocator;
    compile_context = merian::ShaderCompileContext::create(context);
    texture_manager = std::make_shared<merian::TextureManager>(compile_context, context, allocator,
                                                               MAX_GLTEXTURES);
    material_system = std::make_shared<merian::MaterialSystem>(compile_context, context, allocator,
                                                               texture_manager);
    scene = std::make_shared<merian_quake::QuakeScene>(compile_context, context, allocator,
                                                       material_system, argc, argv);
    if (pending_controller) {
        scene->set_controller(pending_controller);
        pending_controller.reset();
    }
}

std::vector<merian::OutputConnectorDescriptor>
QuakeNode::describe_outputs([[maybe_unused]] const merian::NodeIOLayout& io_layout) {
    return {{"scene", con_scene}};
}

void QuakeNode::process(merian::GraphRun& run,
                        [[maybe_unused]] const merian::DescriptorSetHandle& descriptor_set,
                        const merian::NodeIO& io) {
    const merian::CommandBufferHandle& cmd = run.get_cmd();

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
