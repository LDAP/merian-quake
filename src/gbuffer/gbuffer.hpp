#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"
#include "merian-nodes/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-nodes/connectors/connector_utils.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "game/quake_node.hpp"
#include "merian/shader/entry_point.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

class GBuffer : public merian::Node {

  private:
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;

  public:
    GBuffer(const merian::ContextHandle& context);

    ~GBuffer();

    std::vector<merian::InputConnectorHandle> describe_inputs() override;

    std::vector<merian::OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const merian::NodeIOLayout& io_layout) override;

    virtual NodeStatusFlags
    on_connected(const merian::NodeIOLayout& io_layout,
                 const merian::DescriptorSetLayoutHandle& descriptor_set_layout) override;

    virtual void process(merian::GraphRun& run,
                         const merian::DescriptorSetHandle& descriptor_set,
                         const merian::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
    const merian::ContextHandle context;

    merian::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");
    merian::VkSampledImageInHandle con_textures =
        merian::VkSampledImageIn::compute_read("textures");
    merian::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian::VkBufferInHandle con_vtx = merian::VkBufferIn::compute_read("vtx");
    merian::VkBufferInHandle con_prev_vtx = merian::VkBufferIn::compute_read("prev_vtx");
    merian::VkBufferInHandle con_idx = merian::VkBufferIn::compute_read("idx");
    merian::VkBufferInHandle con_ext = merian::VkBufferIn::compute_read("ext");
    merian::VkTLASInHandle con_tlas = merian::VkTLASIn::compute_read("tlas");

    merian::ManagedVkImageOutHandle con_albedo;
    merian::ManagedVkImageOutHandle con_irradiance;
    merian::ManagedVkImageOutHandle con_mv;

    merian::GBufferOutHandle con_gbuffer;
    merian::ManagedVkBufferOutHandle con_hits;

    vk::Extent3D extent;
    merian::EntryPointHandle shader;

    merian::DescriptorSetLayoutHandle descriptor_set_layout;
    merian::PipelineHandle pipe;
    merian::PipelineHandle clear_pipe;

    bool hide_sun = true;
    bool enable_albedo_mipmap = true;
    bool enable_emission_mipmap = true;
};
