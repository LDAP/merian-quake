#include "renderer_restir.hpp"

#include "reservoir.glsl.h"
#include "src/render_restir/clear.comp.spv.h"
#include "src/render_restir/generate_samples.comp.spv.h"

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
        context, merian_quake_restir_generate_samples_comp_spv_size(),
        merian_quake_restir_generate_samples_comp_spv());
    clear_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_restir_clear_comp_spv_size(), merian_quake_restir_clear_comp_spv());
}

RendererRESTIR::~RendererRESTIR() {}

// -------------------------------------------------------------------------------------------

std::vector<merian_nodes::InputConnectorHandle> RendererRESTIR::describe_inputs() {
    return {
        con_vtx,      con_prev_vtx, con_idx,        con_ext,         con_gbuffer,       con_hits,
        con_textures, con_tlas,     con_resolution, con_render_info, con_reservoirs_in,
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
            io.is_connected(con_debug), debug_output_selector);

        auto spec = spec_builder.build();

        pipelines.generate_samples =
            std::make_shared<merian::ComputePipeline>(pipe_layout, generate_samples_shader, spec);
        pipelines.clear =
            std::make_shared<merian::ComputePipeline>(pipe_layout, clear_shader, spec);

        pipelines.recreate = false;
    }

    auto& cur_frame_ptr = io.frame_data<std::shared_ptr<Pipelines>>();
    if (!cur_frame_ptr)
        cur_frame_ptr = std::make_shared<Pipelines>();
    Pipelines& cur_frame = *cur_frame_ptr;

    cur_frame = pipelines;

    // RESET MARKOV CHAINS AT ITERATION 0
    if (run.get_iteration() == 0ul) {
        // Reset buffers
        // io[con_reservoirs_in]->fill(cmd);

        // std::vector<vk::BufferMemoryBarrier> barriers = {
        //     io[con_markovchain]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
        //                                         vk::AccessFlagBits::eShaderRead),
        //     io[con_lightcache]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
        //                                        vk::AccessFlagBits::eShaderRead),
        //     io[con_volume_distancemc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
        //                                               vk::AccessFlagBits::eShaderRead),
        // };

        // cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        //                     vk::PipelineStageFlagBits::eComputeShader, {}, {}, barriers, {});
    }

    const uint32_t render_width = io[con_resolution].width;
    const uint32_t render_height = io[con_resolution].height;

    if (!render_info.render || run.get_iteration() == 0ul) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "clear");
        pipelines.clear->bind(cmd);
        pipelines.clear->bind_descriptor_set(cmd, graph_descriptor_set);
        pipelines.clear->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                     (render_height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);
        return;
    }

    // BIND PIPELINE
    {
        // Surfaces
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "quake.comp");
        pipelines.generate_samples->bind(cmd);
        pipelines.generate_samples->bind_descriptor_set(cmd, graph_descriptor_set);
        pipelines.generate_samples->push_constant(cmd, render_info.uniform);
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

    recreate_pipeline |= config.config_int("spp", spp, 0, 15, "samples per pixel");
    // config.config_bool("adaptive sampling", adaptive_sampling, "Lowers spp adaptively");
    recreate_pipeline |=
        config.config_int("max path length", max_path_length, 0, 15, "maximum path length");

    config.st_separate("Debug");
    recreate_pipeline |= config.config_options("debug output", debug_output_selector, {});

    if (recreate_pipeline) {
        pipelines.recreate = true;
    }
    if (rebuild_graph) {
        return NEEDS_RECONNECT;
    }

    return {};
}
