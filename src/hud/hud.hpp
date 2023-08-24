#pragma once

#include "merian-nodes/compute_node/compute_node.hpp"

namespace merian {

class QuakeHud : public ComputeNode {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float armor;
        float health;
    };

  public:
    QuakeHud(const SharedContext context, const ResourceAllocatorHandle alloc);

    ~QuakeHud();

    std::string name() override;

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    vk::Extent3D extent;
    PushConstant pc;
    ShaderModuleHandle shader;
};

} // namespace merian
