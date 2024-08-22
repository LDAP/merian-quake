#include "renderer_restir.hpp"

#include "restir_di_reservoir.glsl.h"

#include "src/render_restir/restir_di_clear.comp.spv.h"
#include "src/render_restir/restir_di_generate_samples_bsdf.comp.spv.h"
#include "src/render_restir/restir_di_shade.comp.spv.h"
#include "src/render_restir/restir_di_spatial_reuse.comp.spv.h"
#include "src/render_restir/restir_di_temporal_reuse.comp.spv.h"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "merian-shaders/gbuffer.glsl.h"

#include <random>

RendererRESTIR::RendererRESTIR(const merian::ContextHandle& context,
                               const merian::ResourceAllocatorHandle& allocator)
    : Node(), context(context), allocator(allocator) {

    // PIPELINE CREATION
    generate_samples_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_restir_di_generate_samples_bsdf_comp_spv_size(),
        merian_quake_restir_di_generate_samples_bsdf_comp_spv());
    temporal_reuse_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_restir_di_temporal_reuse_comp_spv_size(),
        merian_quake_restir_di_temporal_reuse_comp_spv());
    spatial_reuse_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_restir_di_spatial_reuse_comp_spv_size(),
        merian_quake_restir_di_spatial_reuse_comp_spv());
    shade_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_restir_di_shade_comp_spv_size(),
        merian_quake_restir_di_shade_comp_spv());
    clear_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_restir_di_clear_comp_spv_size(),
        merian_quake_restir_di_clear_comp_spv());
}

RendererRESTIR::~RendererRESTIR() {}

// -------------------------------------------------------------------------------------------

std::vector<merian_nodes::InputConnectorHandle> RendererRESTIR::describe_inputs() {
    return {
        con_vtx,          con_prev_vtx,      con_idx,      con_ext,  con_gbuffer,
        con_prev_gbuffer, con_hits,          con_textures, con_tlas, con_resolution,
        con_render_info,  con_reservoirs_in, con_mv,
    };
}

std::vector<merian_nodes::OutputConnectorHandle>
RendererRESTIR::describe_outputs(const merian_nodes::NodeIOLayout& io_layout) {

    const uint32_t render_width = io_layout[con_resolution]->value().width;
    const uint32_t render_height = io_layout[con_resolution]->value().height;

    con_irradiance = merian_nodes::ManagedVkImageOut::compute_write(
        "irradiance", vk::Format::eR32G32B32A32Sfloat, render_width, render_height);
    con_moments = merian_nodes::ManagedVkImageOut::compute_write(
        "moments", vk::Format::eR32G32Sfloat, render_width, render_height);
    con_debug = merian_nodes::ManagedVkImageOut::compute_write(
        "debug", vk::Format::eR16G16B16A16Sfloat, render_width, render_height);
    con_reservoirs_out = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "reservoirs", vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{
            {},
            gbuffer_size((unsigned long)render_width, render_height) * sizeof(ReSTIRDIReservoir),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc});

    return {
        con_irradiance,
        con_moments,
        con_debug,
        con_reservoirs_out,
    };
}

RendererRESTIR::NodeStatusFlags
RendererRESTIR::on_connected([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
                             const merian::DescriptorSetLayoutHandle& graph_desc_set_layout) {
    pipe_layout = merian::PipelineLayoutBuilder(context)
                      .add_descriptor_set_layout(graph_desc_set_layout)
                      .add_push_constant<QuakeNode::UniformData>()
                      .build_pipeline_layout();

    pipelines.recreate = true;
    return {};
}

void RendererRESTIR::process(merian_nodes::GraphRun& run,
                             const vk::CommandBuffer& cmd,
                             const merian::DescriptorSetHandle& graph_descriptor_set,
                             const merian_nodes::NodeIO& io) {
    const QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];

    // (RE-) CREATE PIPELINE
    if (render_info.constant_data_update || pipelines.recreate) {
        if (randomize_seed) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<uint32_t> dist;
            seed = dist(rng);
        }

        auto spec_builder = merian::SpecializationInfoBuilder();
        spec_builder.add_entry(
            LOCAL_SIZE_X, LOCAL_SIZE_Y, spp, max_path_length,
            render_info.constant.fov_tan_alpha_half, render_info.constant.sun_direction.x,
            render_info.constant.sun_direction.y, render_info.constant.sun_direction.z,
            render_info.constant.sun_color.r, render_info.constant.sun_color.g,
            render_info.constant.sun_color.b, render_info.constant.volume_max_t, seed,
            io.is_connected(con_debug), debug_output_selector, visibility_shade,
            temporal_normal_reject_cos, temporal_depth_reject_percent);

        auto spec = spec_builder.build();

        pipelines.generate_samples =
            std::make_shared<merian::ComputePipeline>(pipe_layout, generate_samples_shader, spec);
        pipelines.temporal_reuse =
            std::make_shared<merian::ComputePipeline>(pipe_layout, temporal_reuse_shader, spec);
        pipelines.spatial_reuse =
            std::make_shared<merian::ComputePipeline>(pipe_layout, spatial_reuse_shader, spec);
        pipelines.shade =
            std::make_shared<merian::ComputePipeline>(pipe_layout, shade_shader, spec);
        pipelines.clear =
            std::make_shared<merian::ComputePipeline>(pipe_layout, clear_shader, spec);

        pipelines.recreate = false;
    }

    auto& cur_frame_ptr = io.frame_data<std::shared_ptr<Pipelines>>();
    if (!cur_frame_ptr)
        cur_frame_ptr = std::make_shared<Pipelines>();
    Pipelines& cur_frame = *cur_frame_ptr;

    cur_frame = pipelines;

    const uint32_t render_width = io[con_resolution].width;
    const uint32_t render_height = io[con_resolution].height;

    if (!render_info.render) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "clear");
        pipelines.clear->bind(cmd);
        pipelines.clear->bind_descriptor_set(cmd, graph_descriptor_set);
        pipelines.clear->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                     (render_height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);
        return;
    }

    auto sync_reservoirs = [&]() {
        auto bar = io[con_reservoirs_out]->buffer_barrier(
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});
    };

    // BIND PIPELINE
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "generate samples");
        pipelines.generate_samples->bind(cmd);
        pipelines.generate_samples->bind_descriptor_set(cmd, graph_descriptor_set);
        pipelines.generate_samples->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                     (render_height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);
    }

    if (temporal_reuse_enable && run.get_iteration() > 0ul) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "temporal reuse");
        sync_reservoirs();
        pipelines.temporal_reuse->bind(cmd);
        pipelines.temporal_reuse->bind_descriptor_set(cmd, graph_descriptor_set);
        pipelines.temporal_reuse->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                     (render_height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);
    }

    if (spatial_reuse_iterations > 0) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "spatial reuse");
        for (int i = 0; i < spatial_reuse_iterations; i++) {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd,
                                     fmt::format("spatial reuse iteration {}", i));
            sync_reservoirs();
            pipelines.spatial_reuse->bind(cmd);
            pipelines.spatial_reuse->bind_descriptor_set(cmd, graph_descriptor_set);
            pipelines.spatial_reuse->push_constant(cmd, render_info.uniform);
            cmd.dispatch((render_width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                         (render_height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);
        }
    }

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "shade");
        sync_reservoirs();
        pipelines.shade->bind(cmd);
        pipelines.shade->bind_descriptor_set(cmd, graph_descriptor_set);
        pipelines.shade->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                     (render_height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);
    }
}

RendererRESTIR::NodeStatusFlags RendererRESTIR::properties(merian::Properties& config) {
    bool recreate_pipeline = false;
    bool rebuild_graph = false;

    config.st_separate("General");
    recreate_pipeline |=
        config.config_bool("randomize seed", randomize_seed, "randomize seed at every graph build");
    if (!randomize_seed) {
        recreate_pipeline |= config.config_uint("seed", seed, "");
    } else {
        config.output_text(fmt::format("seed: {}", seed));
    }

    config.st_separate("Generate samples");
    recreate_pipeline |= config.config_int("spp", spp, 0, 15, "samples per pixel");

    config.st_separate("Temporal Reuse");
    config.config_bool("enable temporal reuse", temporal_reuse_enable);
    float angle = glm::acos(temporal_normal_reject_cos);
    recreate_pipeline |= config.config_angle("temporal normal threshold", angle,
                                             "Reject points with normals farther apart", 0, 180);
    temporal_normal_reject_cos = glm::cos(angle);
    recreate_pipeline |=
        config.config_percent("temporal depth threshold", temporal_depth_reject_percent,
                              "Reject points with depths farther apart (relative to the max)");

    config.st_separate("Spatial Reuse");
    config.config_int("spatial reuse iterations", spatial_reuse_iterations, 0, 7);

    config.st_separate("Shade");
    recreate_pipeline |=
        config.config_bool("shade visibility", visibility_shade,
                           "Check visibility before shading (and write that back)");

    config.st_separate("Debug");
    recreate_pipeline |= config.config_options("debug output", debug_output_selector, {});

    // recreate_pipeline |=
    //     config.config_int("max path length", max_path_length, 0, 15, "maximum path length");

    if (recreate_pipeline) {
        pipelines.recreate = true;
    }
    if (rebuild_graph) {
        return NEEDS_RECONNECT;
    }

    return {};
}
