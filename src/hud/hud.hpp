#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian {

class QuakeHud : public merian::AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        merian::float4 blend = merian::float4(0);
        float armor = 0;
        float health = 0;
        int32_t effect = 0;
    };

  public:
    QuakeHud();

    void initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) override;

    ~QuakeHud();

    std::vector<merian::InputConnectorHandle> describe_inputs() override;

    std::vector<merian::OutputConnectorHandle>
    describe_outputs(const merian::NodeIOLayout& io_layout) override;

    const void* get_push_constant(merian::GraphRun& run, const merian::NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const merian::NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    merian::VkSampledImageInHandle con_src = merian::VkSampledImageIn::compute_read("src");

    vk::Extent3D extent;
    PushConstant pc;
    VulkanEntryPointHandle shader;
};

} // namespace merian
