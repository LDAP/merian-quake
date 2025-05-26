#include "gbuffer.hpp"

#include "game/quake_node.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "../../res/shader/hit.glsl.h"
#include "merian-shaders/image_buffer.glsl.h"

GBuffer::GBuffer(const merian::ContextHandle& context) : context(context) {}

GBuffer::~GBuffer() {}

std::vector<merian_nodes::InputConnectorHandle> GBuffer::describe_inputs() {
    return {
        con_render_info, con_textures, con_resolution, con_vtx,
        con_prev_vtx,    con_idx,      con_ext,        con_tlas,
    };
}

std::vector<merian_nodes::OutputConnectorHandle>
GBuffer::describe_outputs(const merian_nodes::NodeIOLayout& io_layout) {
    extent = io_layout[con_resolution]->value();

    con_albedo = merian_nodes::ManagedVkImageOut::compute_write(
        "albedo", vk::Format::eR16G16B16A16Sfloat, extent.width, extent.height);
    con_irradiance = merian_nodes::ManagedVkImageOut::compute_write(
        "irradiance", vk::Format::eR16G16B16A16Sfloat, extent.width, extent.height);
    con_mv = merian_nodes::ManagedVkImageOut::compute_write("mv", vk::Format::eR16G16Sfloat,
                                                            extent.width, extent.height);
    con_gbuffer = merian_nodes::GBufferOut::compute_write("gbuffer", extent.width, extent.height);
    con_hits = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "hits", vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{
            {},
            image_to_buffer_size(extent.width, extent.height) * sizeof(CompressedHit),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc});

    return {con_albedo, con_irradiance, con_mv, con_gbuffer, con_hits};
}

GBuffer::NodeStatusFlags GBuffer::properties([[maybe_unused]] merian::Properties& props) {
    bool spec_changed = false;
    spec_changed |= props.config_bool("hide sun", hide_sun);
    spec_changed |= props.config_bool("enable albedo mipmap", enable_albedo_mipmap);
    spec_changed |= props.config_bool("enable emission mipmap", enable_emission_mipmap);

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
                      const merian::DescriptorSetHandle& descriptor_set,
                      const merian_nodes::NodeIO& io) {
    const merian::CommandBufferHandle& cmd = run.get_cmd();

    QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];

    if (!pipe || render_info.constant_data_update) {
        shader = run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
            context, "shader/gbuffer/gbuffer.comp", std::nullopt, {},
            {
                {"ENABLE_ALBEDO_MIPMAP", std::to_string(static_cast<int>(enable_albedo_mipmap))},
                {"ENABLE_EMISSION_MIPMAP",
                 std::to_string(static_cast<int>(enable_emission_mipmap))},
            });

        auto pipe_builder = merian::PipelineLayoutBuilder(context);
        pipe_builder.add_push_constant<QuakeNode::UniformData>();
        merian::PipelineLayoutHandle pipe_layout =
            pipe_builder.add_descriptor_set_layout(descriptor_set_layout).build_pipeline_layout();

        // REGULAR PIPE
        auto spec_builder = merian::SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        spec_builder.add_entry(false);
        spec_builder.add_entry(render_info.constant.fov_tan_alpha_half);

        spec_builder.add_entry(render_info.constant.sun_direction.x);
        spec_builder.add_entry(render_info.constant.sun_direction.y);
        spec_builder.add_entry(render_info.constant.sun_direction.z);

        const glm::vec3 sun_color = hide_sun ? glm::vec3(0) : render_info.constant.sun_color;
        spec_builder.add_entry(sun_color.r);
        spec_builder.add_entry(sun_color.g);
        spec_builder.add_entry(sun_color.b);

        spec_builder.add_entry(render_info.constant.volume_max_t);

        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, shader, spec_builder.build());

        // CLEAR PIPE
        auto clear_spec_builder = merian::SpecializationInfoBuilder();
        clear_spec_builder.add_entry(local_size_x, local_size_y);
        clear_spec_builder.add_entry(true);

        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, shader,
                                                               clear_spec_builder.build());
    }

    merian::PipelineHandle current_pipe;
    if (render_info.render) {
        current_pipe = pipe;
    } else {
        current_pipe = clear_pipe;
    }

    cmd->bind(current_pipe);
    cmd->bind_descriptor_set(current_pipe, descriptor_set);
    cmd->push_constant(current_pipe, render_info.uniform);
    cmd->dispatch(extent, local_size_x, local_size_y);
}
