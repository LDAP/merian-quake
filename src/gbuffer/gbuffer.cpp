#include "gbuffer.hpp"

#include "gbuffer.comp.spv.h"

#include "game/quake_node.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "hit.glsl.h"
#include "merian-nodes/common/gbuffer.glsl.h"

GBuffer::GBuffer(const merian::SharedContext context)
    : AbstractCompute(context, sizeof(QuakeNode::UniformData)) {
    shader = std::make_shared<merian::ShaderModule>(context, merian_gbuffer_comp_spv_size(),
                                                    merian_gbuffer_comp_spv());
}

GBuffer::~GBuffer() {}

std::vector<merian_nodes::InputConnectorHandle> GBuffer::describe_inputs() {
    return {
        con_render_info, con_textures, con_resolution, con_vtx,
        con_prev_vtx,    con_idx,      con_ext,        con_tlas,
    };
}

std::vector<merian_nodes::OutputConnectorHandle>
GBuffer::describe_outputs(const merian_nodes::ConnectorIOMap& output_for_input) {
    extent.width = output_for_input[con_resolution]->value().width;
    extent.height = output_for_input[con_resolution]->value().height;

    con_albedo = merian_nodes::ManagedVkImageOut::compute_write(
        "albedo", vk::Format::eR16G16B16A16Sfloat, extent.width, extent.height);
    con_irradiance = merian_nodes::ManagedVkImageOut::compute_write(
        "irradiance", vk::Format::eR16G16B16A16Sfloat, extent.width, extent.height);
    con_mv = merian_nodes::ManagedVkImageOut::compute_write("mv", vk::Format::eR16G16Sfloat,
                                                            extent.width, extent.height);
    con_gbuffer = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "gbuffer", vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             extent.width * extent.height * sizeof(merian_nodes::GBuffer),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc});
    con_hits = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "hits", vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             extent.width * extent.height * sizeof(Hit),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc});

    return {con_albedo, con_irradiance, con_mv, con_gbuffer, con_hits};
}

merian::SpecializationInfoHandle
GBuffer::get_specialization_info([[maybe_unused]] const merian_nodes::NodeIO& io) noexcept {
    QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];
    if (!spec_info || render_info.constant_data_update) {
        auto spec_builder = merian::SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        spec_builder.add_entry(render_info.constant.fov_tan_alpha_half);

        spec_builder.add_entry(render_info.constant.sun_direction.x);
        spec_builder.add_entry(render_info.constant.sun_direction.y);
        spec_builder.add_entry(render_info.constant.sun_direction.z);

        const glm::vec3 sun_color = hide_sun ? glm::vec3(0) : render_info.constant.sun_color;
        spec_builder.add_entry(sun_color.r);
        spec_builder.add_entry(sun_color.g);
        spec_builder.add_entry(sun_color.b);

        spec_builder.add_entry(render_info.constant.volume_max_t);
        spec_info = spec_builder.build();
    }

    return spec_info;
}

const void* GBuffer::get_push_constant([[maybe_unused]] merian_nodes::GraphRun& run,
                                       [[maybe_unused]] const merian_nodes::NodeIO& io) {
    return &io[con_render_info]->uniform;
}

std::tuple<uint32_t, uint32_t, uint32_t>
GBuffer::get_group_count([[maybe_unused]] const merian_nodes::NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

merian::ShaderModuleHandle GBuffer::get_shader_module() {
    return shader;
}

GBuffer::NodeStatusFlags GBuffer::properties([[maybe_unused]] merian::Properties& props) {
    bool spec_changed = props.config_bool("hide sun", hide_sun);

    if (spec_changed) {
        spec_info.reset();
    }

    return {};
}
