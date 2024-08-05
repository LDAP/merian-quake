#include "render_markovchain.hpp"

#include "game/quake_node.hpp"
#include "merian-shaders/gbuffer.glsl.h"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "nlohmann/json.hpp"

#include "grid.h"

#include "clear.comp.spv.h"
#include "quake.comp.spv.h"
#include "volume.comp.spv.h"
#include "volume_forward_project.comp.spv.h"

#include <fstream>
#include <random>

// QuakeNode
// --------------------------------------------------------------------------------------

RendererMarkovChain::RendererMarkovChain(const merian::ContextHandle& context,
                                         const merian::ResourceAllocatorHandle& allocator)
    : Node(), context(context), allocator(allocator) {

    // PIPELINE CREATION
    rt_shader = std::make_shared<merian::ShaderModule>(context, merian_quake_quake_comp_spv_size(),
                                                       merian_quake_quake_comp_spv());
    clear_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_clear_comp_spv_size(), merian_quake_clear_comp_spv());
    volume_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_volume_comp_spv_size(), merian_quake_volume_comp_spv());
    volume_forward_project_shader = std::make_shared<merian::ShaderModule>(
        context, merian_quake_volume_forward_project_comp_spv_size(),
        merian_quake_volume_forward_project_comp_spv());
}

RendererMarkovChain::~RendererMarkovChain() {}

// -------------------------------------------------------------------------------------------

std::vector<merian_nodes::InputConnectorHandle> RendererMarkovChain::describe_inputs() {
    return {
        con_vtx,      con_prev_vtx,   con_idx,
        con_ext,      con_gbuffer,    con_hits,
        con_textures, con_tlas,       con_prev_volume_depth,
        con_mv,       con_resolution, con_render_info,
    };
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
                             light_cache_buffer_size * sizeof(LightCacheVertex),
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

    return {
        con_irradiance,     con_moments,      con_volume,
        con_volume_moments, con_volume_depth, con_volume_mv,
        con_debug,

        con_markovchain,    con_lightcache,   con_volume_distancemc,
    };
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
                                  const vk::CommandBuffer& cmd,
                                  const merian::DescriptorSetHandle& graph_descriptor_set,
                                  const merian_nodes::NodeIO& io) {
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

        auto spec_builder = merian::SpecializationInfoBuilder();
        const float draine_g = std::exp(-2.20679 / (volume_particle_size_um + 3.91029) - 0.428934);
        const float draine_a = std::exp(3.62489 - 8.29288 / (volume_particle_size_um + 5.52825));
        spec_builder.add_entry(
            local_size_x, local_size_y, spp, max_path_length, use_light_cache_tail,
            render_info.constant.fov_tan_alpha_half, render_info.constant.sun_direction.x,
            render_info.constant.sun_direction.y, render_info.constant.sun_direction.z,
            render_info.constant.sun_color.r, render_info.constant.sun_color.g,
            render_info.constant.sun_color.b, adaptive_sampling, volume_spp, volume_use_light_cache,
            draine_g, draine_a, mc_samples, mc_samples_adaptive_prob, distance_mc_samples,
            mc_fast_recovery, light_cache_levels, light_cache_tan_alpha_half,
            light_cache_buffer_size, mc_adaptive_buffer_size, mc_static_buffer_size,
            mc_adaptive_grid_tan_alpha_half, mc_static_grid_width, mc_adaptive_grid_levels,
            distance_mc_grid_width, render_info.constant.volume_max_t, surf_bsdf_p, volume_phase_p,
            dir_guide_prior, dist_guide_p, distance_mc_vertex_state_count, seed,
            io.is_connected(con_debug), debug_output_selector);

        auto spec = spec_builder.build();

        pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, rt_shader, spec);
        clear_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, clear_shader, spec);
        volume_pipe = std::make_shared<merian::ComputePipeline>(pipe_layout, volume_shader, spec);
        volume_forward_project_pipe = std::make_shared<merian::ComputePipeline>(
            pipe_layout, volume_forward_project_shader, spec);
    }

    auto& cur_frame_ptr = io.frame_data<std::shared_ptr<FrameData>>();
    if (!cur_frame_ptr)
        cur_frame_ptr = std::make_shared<FrameData>();
    FrameData& cur_frame = *cur_frame_ptr;

    cur_frame.pipe = pipe;
    cur_frame.clear_pipe = clear_pipe;
    cur_frame.volume_pipe = volume_pipe;
    cur_frame.volume_forward_project_pipe = volume_forward_project_pipe;

    // RESET MARKOV CHAINS AT ITERATION 0
    if (run.get_iteration() == 0UL) {
        // ZERO markov chains and light cache
        io[con_markovchain]->fill(cmd);
        io[con_lightcache]->fill(cmd);
        io[con_volume_distancemc]->fill(cmd);

        std::vector<vk::BufferMemoryBarrier> barriers = {
            io[con_markovchain]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eShaderRead),
            io[con_lightcache]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                               vk::AccessFlagBits::eShaderRead),
            io[con_volume_distancemc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead),
        };

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, barriers, {});
    }

    uint32_t render_width = io[con_resolution].width;
    uint32_t render_height = io[con_resolution].height;

    if (!render_info.render) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "clear");
        clear_pipe->bind(cmd);
        clear_pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        clear_pipe->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
        return;
    }

    // BIND PIPELINE
    {
        // Surfaces
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "quake.comp");
        pipe->bind(cmd);
        pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        pipe->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
    }

    if (volume_forward_project && volume_spp > 0) {
        // Forward project motion vectors for volumes
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "copy mv for volume");
            cmd.copyImage(*io[con_mv], vk::ImageLayout::eTransferSrcOptimal, *io[con_volume_mv],
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageCopy{merian::first_layer(),
                                        {},
                                        merian::first_layer(),
                                        {},
                                        io[con_mv]->get_extent()});
            auto volume_mv_bar = io[con_volume_mv]->barrier(
                vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
                                volume_mv_bar);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "volume forward project");
            volume_forward_project_pipe->bind(cmd);
            volume_forward_project_pipe->bind_descriptor_set(cmd, graph_descriptor_set);
            volume_forward_project_pipe->push_constant(cmd, render_info.uniform);
            cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                         (render_height + local_size_y - 1) / local_size_y, 1);
        }
    }

    auto volume_mv_bar =
        io[con_volume_mv]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                   vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, volume_mv_bar);
    {
        // Volumes
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "volume");
        volume_pipe->bind(cmd);
        volume_pipe->bind_descriptor_set(cmd, graph_descriptor_set);
        volume_pipe->push_constant(cmd, render_info.uniform);
        cmd.dispatch((render_width + local_size_x - 1) / local_size_x,
                     (render_height + local_size_y - 1) / local_size_y, 1);
    }

    if (dump_mc) {
        const std::size_t count =
            std::min(128 * 1024 * 1024 / sizeof(MCState), (std::size_t)mc_adaptive_buffer_size);
        const MCState* buf = static_cast<const MCState*>(allocator->getStaging()->cmdFromBuffer(
            cmd, *io[con_markovchain], 0, sizeof(MCState) * count));
        run.add_submit_callback([count, buf](const merian::QueueHandle& queue) {
            queue->wait_idle();
            nlohmann::json j;

            for (const MCState* v = buf; v < buf + count; v++) {
                nlohmann::json o;
                o["N"] = v->N;
                o["hash"] = v->hash;
                o["w_cos"] = v->w_cos;
                o["sum_w"] = v->sum_w;
                o["w_tgt"] = fmt::format("{} {} {}", v->w_tgt.x, v->w_tgt.y, v->w_tgt.z);

                j.emplace_back(o);
            }

            std::ofstream file("mc_dump.json");
            file << std::setw(4) << j << std::endl;
        });

        dump_mc = false;
    }
}

RendererMarkovChain::NodeStatusFlags RendererMarkovChain::properties(merian::Properties& config) {
    const int32_t old_spp = spp;
    const int32_t old_max_path_lenght = max_path_length;
    const VkBool32 old_use_light_cache_tail = use_light_cache_tail;
    const int32_t old_adaptive_sampling = adaptive_sampling;
    const int32_t old_volume_spp = volume_spp;
    const VkBool32 old_volume_use_light_cache = volume_use_light_cache;
    const float old_volume_particle_size_um = volume_particle_size_um;
    const int32_t old_mc_samples = mc_samples;
    const int32_t old_distance_mc_samples = distance_mc_samples;
    const float old_mc_samples_adaptive_prob = mc_samples_adaptive_prob;
    const VkBool32 old_mc_fast_recovery = mc_fast_recovery;
    const float old_light_cache_levels = light_cache_levels;
    const float old_light_cache_tan_alpha_half = light_cache_tan_alpha_half;
    const uint32_t old_mc_adaptive_buffer_size = mc_adaptive_buffer_size;
    const uint32_t old_mc_static_buffer_size = mc_static_buffer_size;
    const float old_mc_adaptive_grid_tan_alpha_half = mc_adaptive_grid_tan_alpha_half;
    const int32_t old_mc_adaptive_grid_levels = mc_adaptive_grid_levels;
    const float old_mc_static_grid_width = mc_static_grid_width;
    const int32_t old_distance_mc_grid_width = distance_mc_grid_width;
    const uint32_t old_light_cache_buffer_size = light_cache_buffer_size;
    const float old_surf_bsdf_p = surf_bsdf_p;
    const float old_volume_phase_p = volume_phase_p;
    const float old_dir_guide_prior = dir_guide_prior;
    const float old_dist_guide_p = dist_guide_p;
    const uint32_t old_distance_mc_vertex_state_count = distance_mc_vertex_state_count;
    const uint32_t old_seed = seed;
    const bool old_randomize_seed = randomize_seed;
    const int old_debug_output_selector = debug_output_selector;

    config.st_separate("General");
    config.config_bool("randomize seed", randomize_seed, "randomize seed at every graph build");
    if (!randomize_seed) {
        config.config_uint("seed", seed, "");
    } else {
        config.output_text(fmt::format("seed: {}", seed));
    }

    config.st_separate("Guiding Markov chain");
    config.config_percent("ML Prior", dir_guide_prior);
    config.config_int("mc samples", mc_samples, 0, 30);
    config.config_percent("adaptive grid prob", mc_samples_adaptive_prob);
    config.config_uint("adaptive grid buf size", mc_adaptive_buffer_size,
                       "buffer size backing the hash grid");
    config.config_uint("static grid buf size", mc_static_buffer_size,
                       "buffer size backing the hash grid");
    config.config_float("mc adaptive tan(alpha/2)", mc_adaptive_grid_tan_alpha_half,
                        "the adaptive grid resolution, lower means higher resolution.", 0.0001);
    config.config_int("mc adaptive levels", mc_adaptive_grid_levels,
                      "number of quantization steps of the hash grid resolution");
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
    config.config_float("LC levels", light_cache_levels);
    config.config_float("LC tan(alpha/2)", light_cache_tan_alpha_half,
                        "the light cache resolution, lower means higher resolution.", 0.0001);
    config.config_uint("LC buf size", light_cache_buffer_size,
                       "Size of buffer backing the hash grid");

    config.st_separate("Debug");
    config.config_options(
        "debug output", debug_output_selector,
        {"light cache", "mc weight", "mc mean direction", "mc grid", "irradiance", "moments"});

    dump_mc = config.config_bool("Download 128MB MC states",
                                 "Dumps the states as json into mc_dump.json");

    // Only require a pipeline recreation
    if (old_spp != spp || old_max_path_lenght != max_path_length ||
        old_use_light_cache_tail != use_light_cache_tail ||
        old_adaptive_sampling != adaptive_sampling || old_volume_spp != volume_spp ||
        old_volume_use_light_cache != volume_use_light_cache ||
        old_volume_particle_size_um != volume_particle_size_um || old_mc_samples != mc_samples ||
        old_mc_samples_adaptive_prob != mc_samples_adaptive_prob ||
        old_distance_mc_samples != distance_mc_samples ||
        old_mc_fast_recovery != mc_fast_recovery || old_light_cache_levels != light_cache_levels ||
        old_light_cache_tan_alpha_half != light_cache_tan_alpha_half ||
        old_mc_adaptive_grid_tan_alpha_half != mc_adaptive_grid_tan_alpha_half ||
        old_mc_adaptive_grid_levels != mc_adaptive_grid_levels || old_surf_bsdf_p != surf_bsdf_p ||
        old_volume_phase_p != volume_phase_p || old_dir_guide_prior != dir_guide_prior ||
        old_dist_guide_p != dist_guide_p || old_seed != seed ||
        old_randomize_seed != randomize_seed ||
        old_debug_output_selector != debug_output_selector) {
        pipe.reset();
    }

    // Change outputs and require a graph rebuild
    if (old_mc_adaptive_buffer_size != mc_adaptive_buffer_size ||
        old_mc_static_buffer_size != mc_static_buffer_size ||
        old_mc_static_grid_width != mc_static_grid_width ||
        old_distance_mc_grid_width != distance_mc_grid_width ||
        old_light_cache_buffer_size != light_cache_buffer_size ||
        old_distance_mc_vertex_state_count != distance_mc_vertex_state_count) {
        return NEEDS_RECONNECT;
    }

    return {};
}
