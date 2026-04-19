#pragma once

#include "game/quake_render_info.hpp"
#include "game/quake_scene.hpp"

#include "../../res/shader/config.h"
#include "merian-nodes/connectors/ptr_out.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-shaders/scene/scene.hpp"

namespace merian_quake {
class QuakeScene;
} // namespace merian_quake

class QuakeNode : public merian::Node {
  public:
    // Renderer-facing aliases. Kept on QuakeNode so existing renderer code
    // (`QuakeNode::QuakeRenderInfo`, `QuakeNode::UniformData`) still compiles
    // until the renderer migration lands. The actual data lives on the scene.
    using QuakeRenderInfo = merian_quake::QuakeRenderInfo;
    using UniformData = merian_quake::UniformData;
    using ConstantData = merian_quake::ConstantData;
    using PlayerData = merian_quake::PlayerData;
    using RTConfig = merian_quake::RTConfig;

    QuakeNode();
    ~QuakeNode() override;

    merian::DeviceSupportInfo
    query_device_support(const merian::DeviceSupportQueryInfo& query_info) override;

    void initialize(const merian::ContextHandle& context,
                    const merian::ResourceAllocatorHandle& allocator) override;

    std::vector<merian::OutputConnectorDescriptor>
    describe_outputs(const merian::NodeIOLayout& io_layout) override;

    void process(merian::GraphRun& run,
                 const merian::DescriptorSetHandle& descriptor_set,
                 const merian::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

    void set_cmd_args(uint32_t argc, const char** argv);

    void set_controller(const merian::InputControllerHandle& controller);

    void queue_command(const std::string& command);

  private:
    merian::ContextHandle context;
    merian::ResourceAllocatorHandle allocator;
    merian::ShaderCompileContextHandle compile_context;
    std::shared_ptr<merian::FrameCachingShaderObjectAllocator> obj_allocator;
    merian::TextureManagerHandle texture_manager;
    merian::MaterialSystemHandle material_system;

    merian_quake::QuakeSceneHandle scene;

    uint32_t argc = 0;
    const char** argv = nullptr;

    // Stashed controller; either applied during lazy scene init or when
    // set_controller is called after the scene exists.
    merian::InputControllerHandle pending_controller;

    merian::PtrOutHandle<merian::Scene> con_scene = merian::PtrOut<merian::Scene>::create(true);
};
