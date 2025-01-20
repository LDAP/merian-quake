#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"
#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_buffer_array_in.hpp"
#include "merian-nodes/connectors/vk_texture_array_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "game/quake_node.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

class GBuffer : public merian_nodes::Node {

  private:
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;

  public:
    GBuffer(const merian::ContextHandle& context);

    ~GBuffer();

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout) override;

    virtual NodeStatusFlags
    on_connected(const merian_nodes::NodeIOLayout& io_layout,
                 const merian::DescriptorSetLayoutHandle& descriptor_set_layout) override;

    virtual void process(merian_nodes::GraphRun& run,
                         const merian::CommandBufferHandle& cmd,
                         const merian::DescriptorSetHandle& descriptor_set,
                         const merian_nodes::NodeIO& io) override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
    const merian::ContextHandle context;

    merian_nodes::PtrInHandle<QuakeNode::QuakeRenderInfo> con_render_info =
        merian_nodes::PtrIn<QuakeNode::QuakeRenderInfo>::create("render_info");
    merian_nodes::VkTextureArrayInHandle con_textures =
        merian_nodes::VkTextureArrayIn::compute_read("textures");
    merian_nodes::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian_nodes::SpecialStaticIn<vk::Extent3D>::create("resolution");
    merian_nodes::VkBufferArrayInHandle con_vtx =
        merian_nodes::VkBufferArrayIn::compute_read("vtx");
    merian_nodes::VkBufferArrayInHandle con_prev_vtx =
        merian_nodes::VkBufferArrayIn::compute_read("prev_vtx");
    merian_nodes::VkBufferArrayInHandle con_idx =
        merian_nodes::VkBufferArrayIn::compute_read("idx");
    merian_nodes::VkBufferArrayInHandle con_ext =
        merian_nodes::VkBufferArrayIn::compute_read("ext");
    merian_nodes::VkTLASInHandle con_tlas = merian_nodes::VkTLASIn::compute_read("tlas");

    merian_nodes::ManagedVkImageOutHandle con_albedo;
    merian_nodes::ManagedVkImageOutHandle con_irradiance;
    merian_nodes::ManagedVkImageOutHandle con_mv;

    merian_nodes::ManagedVkBufferOutHandle con_gbuffer;
    merian_nodes::ManagedVkBufferOutHandle con_hits;

    vk::Extent3D extent;
    merian::ShaderModuleHandle shader;

    merian::DescriptorSetLayoutHandle descriptor_set_layout;
    merian::PipelineHandle pipe;
    merian::PipelineHandle clear_pipe;

    bool hide_sun = true;
    bool enable_mipmap = true;
};
