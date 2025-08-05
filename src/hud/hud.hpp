#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
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
    QuakeHud(const ContextHandle context);

    ~QuakeHud();

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs(const merian_nodes::NodeIOLayout& io_layout) override;

    SpecializationInfoHandle
    get_specialization_info([[maybe_unused]] const merian_nodes::NodeIO& io) noexcept override;

    const void* get_push_constant(merian_nodes::GraphRun& run,
                                  const merian_nodes::NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const merian_nodes::NodeIO& io) const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    merian_nodes::VkSampledImageInHandle con_src = merian_nodes::VkSampledImageIn::compute_read("src");

    SpecializationInfoHandle spec_info;

    vk::Extent3D extent;
    PushConstant pc;
    ShaderModuleHandle shader;
};

} // namespace merian
