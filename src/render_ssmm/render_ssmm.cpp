#include "render_ssmm.hpp"

#include "game/quake_node.hpp"
#include "merian-shaders/gbuffer.glsl.h"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

#include "../../res/shader/render_ssmm/ssmc_state.h"

#include <random>

// QuakeNode
// --------------------------------------------------------------------------------------

RendererSSMM::RendererSSMM(const merian::ContextHandle& context,
                           const merian::ResourceAllocatorHandle& allocator)
    : Node(), context(context), allocator(allocator) {}

RendererSSMM::~RendererSSMM() {}

// -------------------------------------------------------------------------------------------

std::vector<merian_nodes::InputConnectorHandle> RendererSSMM::describe_inputs() {
    return {con_vtx,      con_prev_vtx, con_idx, con_ext,        con_gbuffer,     con_hits,
            con_textures, con_tlas,     con_mv,  con_resolution, con_render_info, con_prev_ssmc};
}

std::vector<merian_nodes::OutputConnectorHandle>
RendererSSMM::describe_outputs(const merian_nodes::NodeIOLayout& io_layout) {

    const uint32_t render_width = io_layout[con_resolution]->value().width;
    const uint32_t render_height = io_layout[con_resolution]->value().height;

    con_irradiance = merian_nodes::ManagedVkImageOut::compute_write(
        "irradiance", vk::Format::eR32G32B32A32Sfloat, render_width, render_height);
    con_moments = merian_nodes::ManagedVkImageOut::compute_write(
        "moments", vk::Format::eR32G32Sfloat, render_width, render_height);
    con_ssmc = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "ssmc", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{
            {},
            gbuffer_size((long long)render_width, render_height) * sizeof(SSMCState),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc},
        false);

    return {con_irradiance, con_moments, con_ssmc};
}

RendererSSMM::NodeStatusFlags
RendererSSMM::on_connected([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
                           const merian::DescriptorSetLayoutHandle& graph_desc_set_layout) {
    pipe_layout = merian::PipelineLayoutBuilder(context)
                      .add_descriptor_set_layout(graph_desc_set_layout)
                      .add_push_constant<QuakeNode::UniformData>()
                      .build_pipeline_layout();

    pipe.reset();
    return {};
}

void RendererSSMM::process(merian_nodes::GraphRun& run,
                           const merian::DescriptorSetHandle& graph_descriptor_set,
                           const merian_nodes::NodeIO& io) {
    const merian::CommandBufferHandle& cmd = run.get_cmd();

    const QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];

    // (RE-) CREATE PIPELINE
    if (render_info.constant_data_update || !pipe || !clear_pipe) {
        if (randomize_seed) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<uint32_t> dist;
            seed = dist(rng);
        }

        const std::map<std::string, std::string> additional_macro_definitions = {
            {"SURFACE_SPP", std::to_string(spp)},
            {"FOV_TAN_ALPHA_HALF", std::to_string(render_info.constant.fov_tan_alpha_half)},
            {"SUN_W_X", std::to_string(render_info.constant.sun_direction.x)},
            {"SUN_W_Y", std::to_string(render_info.constant.sun_direction.y)},
            {"SUN_W_Z", std::to_string(render_info.constant.sun_direction.z)},
            {"SUN_COLOR_R", std::to_string(render_info.constant.sun_color.r)},
            {"SUN_COLOR_G", std::to_string(render_info.constant.sun_color.g)},
            {"SUN_COLOR_B", std::to_string(render_info.constant.sun_color.b)},
            {"VOLUME_MAX_T", std::to_string(render_info.constant.volume_max_t)},
            {"SURF_BSDF_P", std::to_string(surf_bsdf_p)},
            {"ML_PRIOR_N", std::to_string(ml_prior_n)},
            {"ML_MAX_N", std::to_string(ml_max_n)},
            {"ML_MIN_ALPHA", std::to_string(ml_min_alpha)},
            {"SMIS_GROUP_SIZE", std::to_string(smis_group_size)},
            {"SEED", std::to_string(seed)},
        };

        rt_shader = run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
            context, "shader/render_ssmm/ssmm.comp", std::nullopt, {},
            additional_macro_definitions);
        clear_shader = run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
            context, "shader/render_ssmm/clear.comp", std::nullopt, {},
            additional_macro_definitions);

        auto spec_builder = merian::SpecializationInfoBuilder();

        spec_builder.add_entry(local_size_x, local_size_y);

        auto spec = spec_builder.build();

        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, rt_shader, spec);
        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, clear_shader, spec);
    }

    // RESET MARKOV CHAINS AT ITERATION 0 and 1 (also clear previous frame)
    if (run.get_iteration() <= 1UL) {
        // zero markov chain
        cmd->fill(io[con_ssmc]);

        const std::array<vk::BufferMemoryBarrier, 1> barriers = {
            io[con_ssmc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                         vk::AccessFlagBits::eShaderRead),
        };

        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader, barriers);
    }

    if (!render_info.render || run.get_iteration() == 0UL /* prev not valid yet */) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "clear");
        cmd->bind(clear_pipe);
        cmd->bind_descriptor_set(clear_pipe, graph_descriptor_set);
        cmd->push_constant(clear_pipe, render_info.uniform);
        cmd->dispatch(io[con_resolution], local_size_x, local_size_y);
        return;
    }

    // BIND PIPELINE
    if (io.is_connected(con_irradiance) || io.is_connected(con_moments)) {
        // Surfaces
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "surface");
        cmd->bind(pipe);
        cmd->bind_descriptor_set(pipe, graph_descriptor_set);
        cmd->push_constant(pipe, render_info.uniform);
        cmd->dispatch(io[con_resolution], local_size_x, local_size_y);
    }
}

RendererSSMM::NodeStatusFlags RendererSSMM::properties(merian::Properties& config) {
    bool needs_pipeline_rebuild = false;

    config.st_separate("General");
    needs_pipeline_rebuild |=
        config.config_bool("randomize seed", randomize_seed, "randomize seed at every graph build");
    if (!randomize_seed) {
        needs_pipeline_rebuild |= config.config_uint("seed", seed, "");
    } else {
        config.output_text(fmt::format("seed: {}", seed));
    }

    config.st_separate();
    needs_pipeline_rebuild |= config.config_int("spp", spp, 0, 15, "samples per pixel");
    needs_pipeline_rebuild |=
        config.config_percent("BSDF Prob", surf_bsdf_p, "the probability to use BSDF sampling");

    config.st_separate("MLE estimation");
    needs_pipeline_rebuild |= config.config_float("prior N", ml_prior_n, "", 0.01);
    needs_pipeline_rebuild |= config.config_uint("max N", ml_max_n, "");
    needs_pipeline_rebuild |= config.config_float("min alpha", ml_min_alpha, "", 0.01);

    config.st_separate("SMIS");
    needs_pipeline_rebuild |= config.config_uint("group size", smis_group_size, "");

    // Only require a pipeline recreation
    if (needs_pipeline_rebuild) {
        pipe.reset();
    }

    return {};
}
