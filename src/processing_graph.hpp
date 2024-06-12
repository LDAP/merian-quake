#pragma once

#include "game/quake_node.hpp"
#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/nodes/fxaa/fxaa.hpp"
#include "merian-nodes/nodes/image_read/ldr_image.hpp"
#include "merian/utils/input_controller.hpp"

#include "merian-nodes/nodes/accumulate/accumulate.hpp"
#include "merian-nodes/nodes/add/add.hpp"
#include "merian-nodes/nodes/color_image/color_output.hpp"
#include "merian-nodes/nodes/exposure/exposure.hpp"
#include "merian-nodes/nodes/image_write/image_write.hpp"
#include "merian-nodes/nodes/svgf/svgf.hpp"
#include "merian-nodes/nodes/tonemap/tonemap.hpp"

#include "hud/hud.hpp"
#include "renderer/render_markovchain.hpp"

class ProcessingGraph {
  public:
    ProcessingGraph(const int argc,
                    const char** argv,
                    const merian::SharedContext& context,
                    const merian::ResourceAllocatorHandle& alloc,
                    const merian::FileLoader& loader,
                    const merian::InputControllerHandle& controller)
        : graph(context, alloc) {

        auto blue_noise = std::make_shared<merian_nodes::LDRImageRead>(
            alloc->getStaging(), loader.find_file("blue_noise/1024_1024/LDR_RGBA_0.png").value(),
            true, true);
        const std::array<float, 4> white_color = {1., 1., 1., 1.};
        auto one = std::make_shared<merian_nodes::ColorImage>(vk::Format::eR16G16B16A16Sfloat,
                                                              vk::Extent3D{1920, 1080, 1},
                                                              vk::ClearColorValue(white_color));
        auto mc_render = std::make_shared<RendererMarkovChain>(context, alloc);
        auto quake =
            std::make_shared<QuakeNode>(context, controller, argc - 1, argv + 1, mc_render.get());
        auto accum = std::make_shared<merian_nodes::Accumulate>(context, alloc);
        auto volume_accum = std::make_shared<merian_nodes::Accumulate>(context, alloc);
        svgf = std::make_shared<merian_nodes::SVGF>(context, alloc);
        volume_svgf = std::make_shared<merian_nodes::SVGF>(context, alloc);
        auto tonemap = std::make_shared<merian_nodes::Tonemap>(context);
        auto image_writer = std::make_shared<merian_nodes::ImageWrite>(context, alloc, "image");
        auto image_writer_volume =
            std::make_shared<merian_nodes::ImageWrite>(context, alloc, "image");
        auto exposure = std::make_shared<merian_nodes::AutoExposure>(context);
        hud = std::make_shared<merian_nodes::QuakeHud>(context);
        auto add = std::make_shared<merian_nodes::Add>(context);
        auto beauty_image_write =
            std::make_shared<merian_nodes::ImageWrite>(context, alloc, "image");
        auto fxaa = std::make_shared<merian_nodes::FXAA>(context);

        image_writer->set_callback([accum]() { accum->request_clear(); });
        image_writer_volume->set_callback([volume_accum]() { volume_accum->request_clear(); });

        graph.add_node(quake);
        graph.add_node(one, "one");
        graph.add_node(blue_noise, "blue_noise");
        graph.add_node(mc_render, "render_markovchain");
        graph.add_node(accum, "accum");
        graph.add_node(svgf, "denoiser");
        graph.add_node(volume_svgf, "volume denoiser");
        graph.add_node(tonemap, "tonemap");
        graph.add_node(image_writer, "image writer");
        graph.add_node(image_writer_volume, "volume image writer");
        graph.add_node(exposure, "exposure");
        graph.add_node(hud, "hud");
        graph.add_node(volume_accum, "volume accum");
        graph.add_node(add, "add");
        graph.add_node(beauty_image_write, "beauty image write");
        graph.add_node(fxaa, "fxaa");

        graph.add_connection(quake, mc_render, "render_info", "render_info");
        graph.add_connection(one, volume_svgf, "out", "albedo");
        graph.add_connection(blue_noise, mc_render, "out", "blue_noise");
        graph.add_connection(mc_render, accum, "irradiance", "irr");
        graph.add_connection(mc_render, svgf, "albedo", "albedo");
        graph.add_connection(mc_render, accum, "mv", "mv");
        graph.add_connection(mc_render, svgf, "mv", "mv");
        graph.add_connection(mc_render, accum, "moments", "moments_in");
        graph.add_connection(mc_render, volume_accum, "volume", "irr");
        graph.add_connection(mc_render, volume_accum, "volume_moments", "moments_in");
        graph.add_connection(mc_render, mc_render, "volume_depth", "prev_volume_depth");
        graph.add_connection(mc_render, volume_accum, "volume_mv", "mv");
        graph.add_connection(mc_render, volume_svgf, "volume_mv", "mv");
        graph.add_connection(mc_render, accum, "gbuffer", "gbuf");
        graph.add_connection(mc_render, accum, "gbuffer", "prev_gbuf");
        graph.add_connection(mc_render, mc_render, "gbuffer", "prev_gbuf");
        graph.add_connection(mc_render, svgf, "gbuffer", "gbuffer");
        graph.add_connection(mc_render, svgf, "gbuffer", "prev_gbuffer");
        graph.add_connection(mc_render, volume_accum, "gbuffer", "gbuf");
        graph.add_connection(mc_render, volume_accum, "gbuffer", "prev_gbuf");
        graph.add_connection(mc_render, volume_svgf, "gbuffer", "gbuffer");
        graph.add_connection(mc_render, volume_svgf, "gbuffer", "prev_gbuffer");
        graph.add_connection(mc_render, hud, "gbuffer", "gbuf");
        graph.add_connection(volume_accum, volume_accum, "out_irr", "prev_accum");
        graph.add_connection(volume_accum, volume_svgf, "out_irr", "irr");
        graph.add_connection(volume_accum, volume_accum, "out_moments", "prev_moments");
        graph.add_connection(volume_accum, volume_svgf, "out_moments", "moments");
        graph.add_connection(accum, accum, "out_irr", "prev_accum");
        graph.add_connection(accum, svgf, "out_irr", "irr");
        graph.add_connection(accum, accum, "out_moments", "prev_moments");
        graph.add_connection(accum, svgf, "out_moments", "moments");
        graph.add_connection(volume_svgf, volume_svgf, "out", "prev_out");
        graph.add_connection(volume_svgf, add, "out", "a");
        graph.add_connection(volume_svgf, image_writer_volume, "out", "src");
        graph.add_connection(svgf, mc_render, "out", "prev_filtered");
        graph.add_connection(svgf, svgf, "out", "prev_out");
        graph.add_connection(svgf, image_writer, "out", "src");
        graph.add_connection(svgf, add, "out", "b");
        graph.add_connection(add, exposure, "out", "src");
        graph.add_connection(exposure, tonemap, "out", "src");
        graph.add_connection(tonemap, fxaa, "out", "src");
        graph.add_connection(fxaa, hud, "out", "src");
        graph.add_connection(fxaa, beauty_image_write, "out", "src");
    }

    // Outputs the final image
    void add_beauty_output(const std::string& name,
                           const merian_nodes::NodeHandle& node,
                           const std::string& node_input) {
        graph.add_node(node, name);
        graph.add_connection(hud, node, "output", node_input);
    }

    // Outputs the surfaces only
    void add_surface_output(const std::string& name,
                            const merian_nodes::NodeHandle& node,
                            const std::string& node_input) {
        graph.add_node(node, name);
        graph.add_connection(svgf, node, "out", node_input);
    }

    // Outputs the volumes only
    void add_volume_output(const std::string& name,
                           const merian_nodes::NodeHandle& node,
                           const std::string& node_input) {
        graph.add_node(node, name);
        graph.add_connection(volume_svgf, node, "out", node_input);
    }

    merian_nodes::Graph<>& get() {
        return graph;
    }

  private:
    merian_nodes::Graph<> graph;

    merian_nodes::NodeHandle hud;
    merian_nodes::NodeHandle svgf;
    merian_nodes::NodeHandle volume_svgf;
};
