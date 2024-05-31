#pragma once

#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian {

class QuakeHud : public merian_nodes::AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        glm::vec4 blend = glm::vec4(0);
        float armor = 0;
        float health = 0;
        int32_t effect = 0;
    };

  public:
    QuakeHud(const SharedContext context);

    ~QuakeHud();

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs(const merian_nodes::ConnectorIOMap& output_for_input) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant(merian_nodes::GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    merian_nodes::VkImageInHandle con_src = merian_nodes::VkImageIn::compute_read("src");

    vk::Extent3D extent;
    PushConstant pc;
    ShaderModuleHandle shader;
};

} // namespace merian
