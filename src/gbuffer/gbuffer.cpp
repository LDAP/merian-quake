#include "gbuffer.hpp"

#include "gbuffer.comp.spv.h"
#include "gbuffer.frag.spv.h"

#include "game/quake_node.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_graphics_builder.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "hit.glsl.h"
#include "merian-nodes/common/gbuffer.glsl.h"
#include "merian/vk/renderpass/renderpass_builder.hpp"

GBuffer::GBuffer(const merian::ContextHandle context) : context(context) {
    shader = std::make_shared<merian::ShaderModule>(context, merian_gbuffer_comp_spv_size(),
                                                    merian_gbuffer_comp_spv());
    vertex_shader = merian::ShaderModule::fullscreen_triangle(context);
    fragment_shader = std::make_shared<merian::ShaderModule>(
        context, merian_gbuffer_frag_spv_size(), merian_gbuffer_frag_spv(),
        vk::ShaderStageFlagBits::eFragment);
    renderpass = merian::RenderpassBuilder().build(context);
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
    extent.depth = 1;

    framebuffer = std::make_shared<merian::Framebuffer>(context, renderpass, extent);

    con_albedo = merian_nodes::ManagedVkImageOut::compute_fragment_write(
        "albedo", vk::Format::eR16G16B16A16Sfloat, extent.width, extent.height);
    con_irradiance = merian_nodes::ManagedVkImageOut::compute_fragment_write(
        "irradiance", vk::Format::eR16G16B16A16Sfloat, extent.width, extent.height);
    con_mv = merian_nodes::ManagedVkImageOut::compute_fragment_write(
        "mv", vk::Format::eR16G16Sfloat, extent.width, extent.height);
    con_gbuffer = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "gbuffer", vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment,
        vk::BufferCreateInfo{{},
                             gbuffer_size_bytes(extent.width, extent.height),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc});
    con_hits = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "hits", vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment,
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

    if (spec_changed) {
        gfx_pipeline.reset();
    }

    return {};
}

GBuffer::NodeStatusFlags
GBuffer::on_connected([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
                      const merian::DescriptorSetLayoutHandle& descriptor_set_layout) {
    this->descriptor_set_layout = descriptor_set_layout;
    this->gfx_pipeline.reset();

    return {};
}

struct FrameData {
    merian::PipelineHandle pipeline;
    merian::FramebufferHandle framebuffer;
};

void GBuffer::process([[maybe_unused]] merian_nodes::GraphRun& run,
                      const vk::CommandBuffer& cmd,
                      const merian::DescriptorSetHandle& descriptor_set,
                      const merian_nodes::NodeIO& io) {
    FrameData& frame_data = io.frame_data<FrameData>();

    QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];

    if (!gfx_pipeline || render_info.constant_data_update) {
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

        // CLEAR PIPE
        auto clear_spec_builder = merian::SpecializationInfoBuilder();
        clear_spec_builder.add_entry(local_size_x, local_size_y);
        clear_spec_builder.add_entry<VkBool32>(true);

        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, shader,
                                                               clear_spec_builder.build());
        gfx_pipeline = merian::GraphicsPipelineBuilder()
                           .rasterizer_cull_mode(vk::CullModeFlagBits::eNone)
                           .set_vertex_shader(vertex_shader)
                           .set_fragment_shader(
                               fragment_shader->get_shader_stage_create_info(spec_builder.build()))
                           .viewport_add(io[con_irradiance]->get_extent())
                           .build(context, pipe_layout, renderpass);
    }

    if (render_info.render) {
        frame_data.pipeline = gfx_pipeline;
        frame_data.framebuffer = framebuffer;
        framebuffer->begin_renderpass(cmd);

        gfx_pipeline->bind(cmd);
        gfx_pipeline->bind_descriptor_set(cmd, descriptor_set);
        gfx_pipeline->push_constant(cmd, render_info.uniform);
        cmd.draw(3, 1, 0, 0);

        cmd.endRenderPass();
    } else {
        frame_data.pipeline = clear_pipe;
        clear_pipe->bind(cmd);
        clear_pipe->bind_descriptor_set(cmd, descriptor_set);
        clear_pipe->push_constant(cmd, render_info.uniform);
        auto [x, y, z] = get_group_count(io);
        cmd.dispatch(x, y, z);
    }

    // old_pipeline = current_pipe;
    // current_pipe->bind(cmd);
    // current_pipe->bind_descriptor_set(cmd, descriptor_set);
    // current_pipe->push_constant(cmd, render_info.uniform);
    // auto [x, y, z] = get_group_count(io);
    // cmd.dispatch(x, y, z);
}
