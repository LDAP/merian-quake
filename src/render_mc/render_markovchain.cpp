#include "render_markovchain.hpp"

#include "game/quake_node.hpp"
#include "merian-shaders/gbuffer.glsl.h"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "nlohmann/json.hpp"

#include "../../res/shader/render_mc/grid.h"

#include <fstream>
#include <random>

// QuakeNode
// --------------------------------------------------------------------------------------

RendererMarkovChain::RendererMarkovChain(const merian::ContextHandle& context,
                                         const merian::ResourceAllocatorHandle& allocator)
    : Node(), context(context), allocator(allocator) {}

RendererMarkovChain::~RendererMarkovChain() {}

// -------------------------------------------------------------------------------------------

std::vector<merian_nodes::InputConnectorHandle> RendererMarkovChain::describe_inputs() {
    return {con_vtx,      con_prev_vtx,   con_idx,
            con_ext,      con_gbuffer,    con_hits,
            con_textures, con_tlas,       con_prev_volume_depth,
            con_mv,       con_resolution, con_render_info,
            con_prev_ssmc};
}

std::vector<merian_nodes::OutputConnectorHandle>
RendererMarkovChain::describe_outputs(const merian_nodes::NodeIOLayout& io_layout) {

    const uint32_t render_width = io_layout[con_resolution]->value().width;
    const uint32_t render_height = io_layout[con_resolution]->value().height;

    con_irradiance = merian_nodes::ManagedVkImageOut::compute_write(
        "irradiance", vk::Format::eR32G32B32A32Sfloat, render_width, render_height);
    con_moments = merian_nodes::ManagedVkImageOut::compute_write(
        "moments", vk::Format::eR32G32Sfloat, render_width, render_height);
    con_volume = merian_nodes::ManagedVkImageOut::compute_write(
        "volume", vk::Format::eR16G16B16A16Sfloat, render_width, render_height);
    con_volume_moments = merian_nodes::ManagedVkImageOut::compute_write(
        "volume_moments", vk::Format::eR32G32Sfloat, render_width, render_height);
    con_volume_depth = merian_nodes::ManagedVkImageOut::compute_write(
        "volume_depth", vk::Format::eR32Sfloat, render_width, render_height);
    con_volume_mv = merian_nodes::ManagedVkImageOut::compute_read_write_transfer_dst(
        "volume_mv", vk::Format::eR16G16Sfloat, render_width, render_height, 1,
        vk::ImageLayout::eTransferDstOptimal);
    con_debug = merian_nodes::ManagedVkImageOut::compute_write(
        "debug", vk::Format::eR16G16B16A16Sfloat, render_width, render_height);

    con_markovchain = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "markovchain", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             (mc_adaptive_buffer_size + mc_static_buffer_size) * sizeof(MCState),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc},
        true);
    con_lightcache = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "lightcache", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             lc_buffer_size * sizeof(LightCacheVertex),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc},
        true);
    con_volume_distancemc = std::make_shared<merian_nodes::ManagedVkBufferOut>(
        "volume_distancemc", vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo{{},
                             (render_width / distance_mc_grid_width + 2) *
                                 (render_height / distance_mc_grid_width + 2) *
                                 MAX_DISTANCE_MC_VERTEX_STATE_COUNT * sizeof(DistanceMCState),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc},
        true);

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

    return {con_irradiance,
            con_moments,
            con_volume,
            con_volume_moments,
            con_volume_depth,
            con_volume_mv,
            con_debug,

            con_markovchain,
            con_lightcache,
            con_volume_distancemc,
            con_ssmc};
}

RendererMarkovChain::NodeStatusFlags
RendererMarkovChain::on_connected([[maybe_unused]] const merian_nodes::NodeIOLayout& io_layout,
                                  const merian::DescriptorSetLayoutHandle& graph_desc_set_layout) {
    pipe_layout = merian::PipelineLayoutBuilder(context)
                      .add_descriptor_set_layout(graph_desc_set_layout)
                      .add_push_constant<QuakeNode::UniformData>()
                      .build_pipeline_layout();

    pipe.reset();
    return {};
}

void RendererMarkovChain::process(merian_nodes::GraphRun& run,
                                  const merian::DescriptorSetHandle& graph_descriptor_set,
                                  const merian_nodes::NodeIO& io) {
    const merian::CommandBufferHandle& cmd = run.get_cmd();

    const QuakeNode::QuakeRenderInfo& render_info = *io[con_render_info];

    // (RE-) CREATE PIPELINE
    if (render_info.constant_data_update || !pipe || !clear_pipe || !volume_pipe ||
        !volume_forward_project_pipe) {
        if (randomize_seed) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<uint32_t> dist;
            seed = dist(rng);
        }

        const float draine_g = std::exp(-2.20679 / (volume_particle_size_um + 3.91029) - 0.428934);
        const float draine_a = std::exp(3.62489 - 8.29288 / (volume_particle_size_um + 5.52825));

        const std::map<std::string, std::string> additional_macro_definitions = {
            {"MERIAN_QUAKE_REFERENCE_MODE",
             std::to_string(static_cast<int>(reference_mode || surf_bsdf_p == 1.0))},
            {"MERIAN_QUAKE_ADAPTIVE_GRID_TYPE", std::to_string(mc_adaptive_grid_type)},
            {"SURFACE_SPP", std::to_string(spp)},
            {"MAX_PATH_LENGTH", std::to_string(max_path_length)},
            {"USE_LIGHT_CACHE_TAIL", std::to_string(use_light_cache_tail)},
            {"FOV_TAN_ALPHA_HALF", std::to_string(render_info.constant.fov_tan_alpha_half)},
            {"SUN_W_X", std::to_string(render_info.constant.sun_direction.x)},
            {"SUN_W_Y", std::to_string(render_info.constant.sun_direction.y)},
            {"SUN_W_Z", std::to_string(render_info.constant.sun_direction.z)},
            {"SUN_COLOR_R", std::to_string(render_info.constant.sun_color.r)},
            {"SUN_COLOR_G", std::to_string(render_info.constant.sun_color.g)},
            {"SUN_COLOR_B", std::to_string(render_info.constant.sun_color.b)},
            {"VOLUME_SPP", std::to_string(volume_spp)},
            {"VOLUME_USE_LIGHT_CACHE", std::to_string(volume_use_light_cache)},
            {"DRAINE_G", std::to_string(draine_g)},
            {"DRAINE_A", std::to_string(draine_a)},
            {"MC_SAMPLES", std::to_string(mc_samples)},
            {"MC_SAMPLES_ADAPTIVE_PROB", std::to_string(mc_samples_adaptive_prob)},
            {"DISTANCE_MC_SAMPLES", std::to_string(distance_mc_samples)},
            {"MC_FAST_RECOVERY", std::to_string(mc_fast_recovery)},
            {"MERIAN_QUAKE_LC_GRID_TYPE", std::to_string(lc_grid_type)},
            {"LIGHT_CACHE_BUFFER_SIZE", std::to_string(lc_buffer_size)},
            {"LC_GRID_STEPS_PER_UNIT_SIZE", std::to_string(lc_grid_steps_per_unit_size)},
            {"LC_GRID_TAN_ALPHA_HALF", std::to_string(lc_grid_tan_alpha_half)},
            {"LC_GRID_MIN_WIDTH", std::to_string(lc_grid_min_width)},
            {"LC_GRID_POWER", std::to_string(lc_grid_power)},
            {"MC_ADAPTIVE_BUFFER_SIZE", std::to_string(mc_adaptive_buffer_size)},
            {"MC_ADAPTIVE_GRID_TAN_ALPHA_HALF", std::to_string(mc_adaptive_grid_tan_alpha_half)},
            {"MC_ADAPTIVE_GRID_MIN_WIDTH", std::to_string(mc_adaptive_grid_min_width)},
            {"MC_ADAPTIVE_GRID_POWER", std::to_string(mc_adaptive_grid_power)},
            {"MC_ADAPTIVE_GRID_STEPS_PER_UNIT_SIZE",
             std::to_string(mc_adaptive_grid_steps_per_unit_size)},
            {"MC_STATIC_BUFFER_SIZE", std::to_string(mc_static_buffer_size)},
            {"MC_STATIC_GRID_WIDTH", std::to_string(mc_static_grid_width)},
            {"DISTANCE_MC_GRID_WIDTH", std::to_string(distance_mc_grid_width)},
            {"VOLUME_MAX_T", std::to_string(render_info.constant.volume_max_t)},
            {"SURF_BSDF_P", std::to_string(surf_bsdf_p)},
            {"VOLUME_PHASE_P", std::to_string(volume_phase_p)},
            {"DIR_GUIDE_PRIOR", std::to_string(dir_guide_prior)},
            {"DIST_GUIDE_P", std::to_string(dist_guide_p)},
            {"DISTANCE_MC_VERTEX_STATE_COUNT", std::to_string(distance_mc_vertex_state_count)},
            {"SEED", std::to_string(seed)},
            {"DEBUG_OUTPUT_CONNECTED",
             std::to_string(static_cast<int>(io.is_connected(con_debug)))},
            {"DEBUG_OUTPUT_SELECTOR", std::to_string(debug_output_selector)},
        };

        rt_shader = run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
            context, "shader/render_mc/quake.comp", std::nullopt, {}, additional_macro_definitions);
        clear_shader = run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
            context, "shader/render_mc/clear.comp", std::nullopt, {}, additional_macro_definitions);
        volume_shader = run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
            context, "shader/render_mc/volume.comp", std::nullopt, {},
            additional_macro_definitions);
        volume_forward_project_shader =
            run.get_shader_compiler()->find_compile_glsl_to_shadermodule(
                context, "shader/render_mc/volume_forward_project.comp", std::nullopt, {},
                additional_macro_definitions);

        auto spec_builder = merian::SpecializationInfoBuilder();

        spec_builder.add_entry(local_size_x, local_size_y);

        auto spec = spec_builder.build();

        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, rt_shader, spec);
        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, clear_shader, spec);
        volume_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, volume_shader, spec);
        volume_forward_project_pipe = std::make_shared<merian::ComputePipeline>(
            pipe_layout, volume_forward_project_shader, spec);
    }

    // RESET MARKOV CHAINS AT ITERATION 0
    if (run.get_iteration() == 0UL) {
        // ZERO markov chains and light cache
        cmd->fill(io[con_markovchain]);
        cmd->fill(io[con_lightcache]);
        cmd->fill(io[con_volume_distancemc]);
        cmd->fill(io[con_ssmc]);

        const std::array<vk::BufferMemoryBarrier, 4> barriers = {
            io[con_markovchain]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eShaderRead),
            io[con_lightcache]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                               vk::AccessFlagBits::eShaderRead),
            io[con_volume_distancemc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead),
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

    const bool enable_volume = io.is_connected(con_volume) || io.is_connected(con_volume_moments);

    if (enable_volume) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "copy mv for volume");
        cmd->copy(
            io[con_mv], vk::ImageLayout::eTransferSrcOptimal, io[con_volume_mv],
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageCopy{
                merian::first_layer(), {}, merian::first_layer(), {}, io[con_mv]->get_extent()});
        const auto volume_mv_bar = io[con_volume_mv]->barrier(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader, volume_mv_bar);
    }

    if (enable_volume && volume_spp > 0 && volume_forward_project) {
        // Forward project motion vectors for volumes

        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "volume forward project");
        cmd->bind(volume_forward_project_pipe);
        cmd->bind_descriptor_set(volume_forward_project_pipe, graph_descriptor_set);
        cmd->push_constant(volume_forward_project_pipe, render_info.uniform);
        cmd->dispatch(io[con_resolution], local_size_x, local_size_y);

        const auto volume_mv_bar =
            io[con_volume_mv]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                       vk::AccessFlagBits::eShaderRead);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, volume_mv_bar);
    }

    if (enable_volume) {
        // Volumes
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "volume");
        cmd->bind(volume_pipe);
        cmd->bind_descriptor_set(volume_pipe, graph_descriptor_set);
        cmd->push_constant(volume_pipe, render_info.uniform);
        cmd->dispatch(io[con_resolution], local_size_x, local_size_y);
    }

    if (dump_mc) {
        const std::size_t count =
            std::min(128L * 1024 * 1024 / sizeof(MCState), (std::size_t)mc_adaptive_buffer_size);
        const merian::MemoryAllocationHandle memory = allocator->getStaging()->cmd_from_device(
            cmd, io[con_markovchain], 0, sizeof(MCState) * count);
        run.add_submit_callback([count, memory](const merian::QueueHandle& queue,
                                                [[maybe_unused]] merian_nodes::GraphRun& run) {
            queue->wait_idle();
            nlohmann::json j;

            MCState* buf = memory->map_as<MCState>();
            for (const MCState* v = buf; v < buf + count; v++) {
                nlohmann::json o;
                o["N"] = v->N;
                o["hash"] = v->hash;
                o["w_cos"] = v->w_cos;
                o["sum_w"] = v->sum_w;
                o["w_tgt"] = fmt::format("{} {} {}", v->w_tgt.x, v->w_tgt.y, v->w_tgt.z);

                j.emplace_back(o);
            }
            memory->unmap();
            std::ofstream file("mc_dump.json");
            file << std::setw(4) << j << '\n';
        });

        dump_mc = false;
    }
}

RendererMarkovChain::NodeStatusFlags RendererMarkovChain::properties(merian::Properties& config) {
    const int32_t old_spp = spp;
    const int32_t old_max_path_lenght = max_path_length;
    const VkBool32 old_use_light_cache_tail = use_light_cache_tail;
    const int32_t old_volume_spp = volume_spp;
    const VkBool32 old_volume_use_light_cache = volume_use_light_cache;
    const float old_volume_particle_size_um = volume_particle_size_um;
    const int32_t old_mc_samples = mc_samples;
    const int32_t old_distance_mc_samples = distance_mc_samples;
    const float old_mc_samples_adaptive_prob = mc_samples_adaptive_prob;
    const VkBool32 old_mc_fast_recovery = mc_fast_recovery;
    const uint32_t old_mc_adaptive_buffer_size = mc_adaptive_buffer_size;
    const uint32_t old_mc_static_buffer_size = mc_static_buffer_size;
    const float old_mc_adaptive_grid_tan_alpha_half = mc_adaptive_grid_tan_alpha_half;
    const float old_mc_static_grid_width = mc_static_grid_width;
    const int32_t old_distance_mc_grid_width = distance_mc_grid_width;
    const uint32_t old_light_cache_buffer_size = lc_buffer_size;
    const float old_surf_bsdf_p = surf_bsdf_p;
    const float old_volume_phase_p = volume_phase_p;
    const float old_dir_guide_prior = dir_guide_prior;
    const float old_dist_guide_p = dist_guide_p;
    const uint32_t old_distance_mc_vertex_state_count = distance_mc_vertex_state_count;
    const uint32_t old_seed = seed;
    const bool old_randomize_seed = randomize_seed;
    const int old_debug_output_selector = debug_output_selector;
    const bool old_reference_mode = reference_mode;

    bool needs_pipeline_rebuild = false;
    bool needs_reconnect = false;

    config.st_separate("General");
    config.config_bool("randomize seed", randomize_seed, "randomize seed at every graph build");
    if (!randomize_seed) {
        config.config_uint("seed", seed, "");
    } else {
        config.output_text(fmt::format("seed: {}", seed));
    }
    config.config_bool("reference mode", reference_mode);

    config.st_separate("Guiding Markov chain");
    config.config_percent("ML Prior", dir_guide_prior);
    config.config_int("mc samples", mc_samples, 0, 30);

    config.config_percent("adaptive grid prob", mc_samples_adaptive_prob);
    needs_reconnect |= config.config_options("adaptive grid type", mc_adaptive_grid_type,
                                             {"exponential", "quadratic"});
    config.config_uint("adaptive grid buf size", mc_adaptive_buffer_size,
                       "buffer size backing the hash grid");
    config.config_float("adaptive grid tan(alpha/2)", mc_adaptive_grid_tan_alpha_half,
                        "the adaptive grid resolution, lower means higher resolution.", 0.0001);
    needs_pipeline_rebuild |= config.config_float("adaptive grid steps per unit",
                                                  mc_adaptive_grid_steps_per_unit_size, "", 0.1);
    needs_pipeline_rebuild |=
        config.config_float("adaptive grid min width", mc_adaptive_grid_min_width, "", 0.001);
    needs_pipeline_rebuild |=
        config.config_float("adaptive grid power", mc_adaptive_grid_power, "", 0.001);

    config.config_uint("static grid buf size", mc_static_buffer_size,
                       "buffer size backing the hash grid");
    config.config_float("mc static width", mc_static_grid_width,
                        "the static grid width in worldspace units, lower means higher resolution",
                        0.1);

    config.config_bool("mc fast recovery", mc_fast_recovery,
                       "When enabled, markov chains are flooded with invalidated states when no "
                       "light is detected.");

    config.st_separate("RT Surface");
    config.config_int("spp", spp, 0, 15, "samples per pixel");
    // config.config_bool("adaptive sampling", adaptive_sampling, "Lowers spp adaptively");
    config.config_int("max path length", max_path_length, 0, 15, "maximum path length");
    config.config_percent("BSDF Prob", surf_bsdf_p, "the probability to use BSDF sampling");

    config.st_separate("RT Volume");
    config.config_int("volume spp", volume_spp, 0, 15, "samples per pixel for volume events");
    config.config_int("dist mc samples", distance_mc_samples, 0, 30);
    config.config_int("dist mc grid width", distance_mc_grid_width,
                      "the markov chain hash grid width in pixels");
    config.config_uint("dist mc states per vertex", distance_mc_vertex_state_count, 1,
                       MAX_DISTANCE_MC_VERTEX_STATE_COUNT,
                       "number of markov chain states per vertex");
    config.config_float("particle size", volume_particle_size_um, "in mircometer (5-50)", 0.1);
    config.config_percent("dist guide p", dist_guide_p, "higher means more distance guiding");
    config.config_percent("Phase Prob", volume_phase_p,
                          "the probability to use phase function sampling");
    config.config_bool("volume forward project", volume_forward_project);

    config.st_separate("Light cache");
    config.config_bool("surf: use LC", use_light_cache_tail,
                       "use the light cache for the path tail");
    config.config_bool("volume: use LC", volume_use_light_cache,
                       "query light cache for non-emitting surfaces");

    needs_reconnect |=
        config.config_options("LC grid type", lc_grid_type, {"exponential", "quadratic"});
    config.config_uint("LC buf size", lc_buffer_size, "Size of buffer backing the hash grid");
    needs_pipeline_rebuild |=
        config.config_float("LC grid tan(alpha/2)", lc_grid_tan_alpha_half,
                            "the light cache resolution, lower means higher resolution.", 0.0001);
    needs_pipeline_rebuild |=
        config.config_float("LC grid steps per unit", lc_grid_steps_per_unit_size, "", 0.1);
    needs_pipeline_rebuild |=
        config.config_float("LC grid min width", lc_grid_min_width, "", 0.001);
    needs_pipeline_rebuild |= config.config_float("LC grid power", lc_grid_power, "", 0.1);

    config.st_separate("Debug");
    config.config_options("debug output", debug_output_selector,
                          {"light cache", "mc weight", "mc mean direction", "mc grid", "irradiance",
                           "moments", "mc cos", "mc N", "mc motion vectors"});
    needs_pipeline_rebuild |= config.config_bool("recreate pipeline");
    dump_mc = config.config_bool("Download 128MB MC states",
                                 "Dumps the states as json into mc_dump.json");

    // Only require a pipeline recreation
    if (needs_pipeline_rebuild || old_spp != spp || old_max_path_lenght != max_path_length ||
        old_use_light_cache_tail != use_light_cache_tail || old_volume_spp != volume_spp ||
        old_volume_use_light_cache != volume_use_light_cache ||
        old_volume_particle_size_um != volume_particle_size_um || old_mc_samples != mc_samples ||
        old_mc_samples_adaptive_prob != mc_samples_adaptive_prob ||
        old_distance_mc_samples != distance_mc_samples ||
        old_mc_fast_recovery != mc_fast_recovery ||
        old_mc_adaptive_grid_tan_alpha_half != mc_adaptive_grid_tan_alpha_half ||
        old_surf_bsdf_p != surf_bsdf_p || old_volume_phase_p != volume_phase_p ||
        old_dir_guide_prior != dir_guide_prior || old_dist_guide_p != dist_guide_p ||
        old_seed != seed || old_randomize_seed != randomize_seed ||
        old_debug_output_selector != debug_output_selector) {
        pipe.reset();
    }

    // Change outputs and require a graph rebuild
    if (needs_reconnect || old_mc_adaptive_buffer_size != mc_adaptive_buffer_size ||
        old_mc_static_buffer_size != mc_static_buffer_size ||
        old_mc_static_grid_width != mc_static_grid_width ||
        old_distance_mc_grid_width != distance_mc_grid_width ||
        old_light_cache_buffer_size != lc_buffer_size ||
        old_distance_mc_vertex_state_count != distance_mc_vertex_state_count ||
        old_reference_mode != reference_mode) {
        return NEEDS_RECONNECT;
    }

    return {};
}
