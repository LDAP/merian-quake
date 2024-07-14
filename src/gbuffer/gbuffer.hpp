#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"
#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/connectors/vk_buffer_array_in.hpp"
#include "merian-nodes/connectors/vk_texture_array_in.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "merian-nodes/nodes/compute_node/compute_node.hpp"

#include "game/quake_node.hpp"

class GBuffer : public merian_nodes::AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;

  public:
    GBuffer(const merian::SharedContext context);

    ~GBuffer();

    std::vector<merian_nodes::InputConnectorHandle> describe_inputs() override;

    std::vector<merian_nodes::OutputConnectorHandle>
    describe_outputs(const merian_nodes::ConnectorIOMap& output_for_input) override;

    merian::SpecializationInfoHandle get_specialization_info(
        [[maybe_unused]] const merian_nodes::NodeIO& io) noexcept override;

    const void* get_push_constant(merian_nodes::GraphRun& run,
                                  const merian_nodes::NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const merian_nodes::NodeIO& io) const noexcept override;

    merian::ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(merian::Properties& config) override;

  private:
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
    merian_nodes::ManagedVkImageOutHandle con_mv;

    merian_nodes::ManagedVkBufferOutHandle con_gbuffer;

    merian::SpecializationInfoHandle spec_info;

    vk::Extent3D extent;
    merian::ShaderModuleHandle shader;
};
