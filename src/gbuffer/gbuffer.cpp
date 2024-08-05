#include "gbuffer.hpp"

#include "gbuffer.comp.spv.h"

#include "game/quake_node.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "hit.glsl.h"
#include "merian-shaders/gbuffer.glsl.h"

GBuffer::GBuffer(const merian::ContextHandle context) : context(context) {
    shader = std::make_shared<merian::ShaderModule>(context, merian_quake_gbuffer_comp_spv_size(),
                                                    merian_quake_gbuffer_comp_spv());
}

GBuffer::~GBuffer() {}

std::vector<merian_nodes::InputConnectorHandle> GBuffer::describe_inputs() {
    return {
        con_render_info, con_textures, con_resolution, con_vtx,
        con_prev_vtx,    con_idx,      con_ext,        con_tlas,
    };
}

std::vector<merian_nodes::OutputConnectorHandle>
GBuffer::describe_outputs(const merian_nodes::NodeIOLayout& io_layout) {
    extent.width = io_layout[con_resolution]->value().width;
    extent.height = io_layout[con_resolution]->value().height;

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
                             gbuffer_size_bytes(extent.width, extent.height),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc});
    con_hits = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "hits", vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             gbuffer_size(extent.width, extent.height) * sizeof(Hit),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc});

    return {con_albedo, con_irradiance, con_mv, con_gbuffer, con_hits};
}

std::tuple<uint32_t, uint32_t, uint32_t>
GBuffer::get_group_count([[maybe_unused]] const merian_nodes::NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

GBuffer::NodeStatusFlags GBuffer::properties([[maybe_unused]] merian::Properties& props) {
    bool spec_changed = props.config_bool("hide sun", hide_sun);
    spec_changed |= props.config_bool("enable mipmap", enable_mipmap);

    if (spec_changed) {
        pipe.reset();
    }

    return {};
}

GBuffer::NodeStatusFlags
GBuffer::on_connected([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
                      const merian::DescriptorSetLayoutHandle& descriptor_set_layout) {
    this->descriptor_set_layout = descriptor_set_layout;
    this->pipe.reset();

    return {};
}

void GBuffer::process([[maybe_unused]] merian_nodes::GraphRun& run,
                      const vk::CommandBuffer& cmd,
                      const merian::DescriptorSetHandle& descriptor_set,
                      const merian_nodes::NodeIO& io) {
    merian::PipelineHandle& old_pipeline = io.frame_data<merian::PipelineHandle>();
    old_pipeline.reset();

    QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];

    if (!pipe || render_info.constant_data_update) {
        auto pipe_builder = merian::PipelineLayoutBuilder(context);
        pipe_builder.add_push_constant<QuakeNode::UniformData>();
        merian::PipelineLayoutHandle pipe_layout =
            pipe_builder.add_descriptor_set_layout(descriptor_set_layout).build_pipeline_layout();

        // REGULAR PIPE
        auto spec_builder = merian::SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        spec_builder.add_entry<VkBool32>(false);
        spec_builder.add_entry(render_info.constant.fov_tan_alpha_half);

        spec_builder.add_entry(render_info.constant.sun_direction.x);
        spec_builder.add_entry(render_info.constant.sun_direction.y);
        spec_builder.add_entry(render_info.constant.sun_direction.z);

        const glm::vec3 sun_color = hide_sun ? glm::vec3(0) : render_info.constant.sun_color;
        spec_builder.add_entry(sun_color.r);
        spec_builder.add_entry(sun_color.g);
        spec_builder.add_entry(sun_color.b);

        spec_builder.add_entry(render_info.constant.volume_max_t);
        spec_builder.add_entry(enable_mipmap);

        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, shader, spec_builder.build());

        // CLEAR PIPE
        auto clear_spec_builder = merian::SpecializationInfoBuilder();
        clear_spec_builder.add_entry(local_size_x, local_size_y);
        clear_spec_builder.add_entry<VkBool32>(true);

        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, shader,
                                                               clear_spec_builder.build());
    }

    merian::PipelineHandle current_pipe;
    if (render_info.render) {
        current_pipe = pipe;
    } else {
        current_pipe = clear_pipe;
    }

    old_pipeline = current_pipe;
    current_pipe->bind(cmd);
    current_pipe->bind_descriptor_set(cmd, descriptor_set);
    current_pipe->push_constant(cmd, render_info.uniform);
    auto [x, y, z] = get_group_count(io);
    cmd.dispatch(x, y, z);
}
