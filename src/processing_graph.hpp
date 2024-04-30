#pragma once

#include "merian-nodes/fxaa/fxaa.hpp"
#include "merian/utils/input_controller.hpp"
#include "merian/vk/graph/graph.hpp"

#include "merian-nodes/accumulate/accumulate.hpp"
#include "merian-nodes/add/add.hpp"
#include "merian-nodes/color_output/color_output.hpp"
#include "merian-nodes/exposure/exposure.hpp"
#include "merian-nodes/image/image.hpp"
#include "merian-nodes/image_write/image_write.hpp"
#include "merian-nodes/median_approx/median.hpp"
#include "merian-nodes/svgf/svgf.hpp"
#include "merian-nodes/tonemap/tonemap.hpp"

#include "hud/hud.hpp"
#include "quake/quake_node.hpp"

class ProcessingGraph {
  public:
    ProcessingGraph(const int argc,
                    const char** argv,
                    const merian::SharedContext& context,
                    const merian::ResourceAllocatorHandle& alloc,
                    const merian::QueueHandle& wait_queue,
                    const merian::FileLoader& loader,
                    const uint32_t ring_size,
                    const merian::InputControllerHandle& controller)
        : graph(context, alloc, wait_queue) {

        auto blue_noise = std::make_shared<merian::ImageNode>(
            alloc, "blue_noise/1024_1024/LDR_RGBA_0.png", loader, true);
        const std::array<float, 4> white_color = {1., 1., 1., 1.};
        auto one = std::make_shared<merian::ColorOutputNode>(vk::Format::eR16G16B16A16Sfloat,
                                                             vk::Extent3D{1920, 1080, 1},
                                                             vk::ClearColorValue(white_color));
        auto quake =
            std::make_shared<QuakeNode>(context, alloc, controller, ring_size, argc - 1, argv + 1);
        auto accum = std::make_shared<merian::AccumulateNode>(context, alloc);
        auto volume_accum = std::make_shared<merian::AccumulateNode>(context, alloc);
        svgf = std::make_shared<merian::SVGFNode>(context, alloc);
        volume_svgf = std::make_shared<merian::SVGFNode>(context, alloc);
        auto tonemap = std::make_shared<merian::TonemapNode>(context, alloc);
        auto image_writer = std::make_shared<merian::ImageWriteNode>(context, alloc, "image");
        auto image_writer_volume =
            std::make_shared<merian::ImageWriteNode>(context, alloc, "image");
        auto exposure = std::make_shared<merian::ExposureNode>(context, alloc);
        hud = std::make_shared<merian::QuakeHud>(context, alloc);
        auto add = std::make_shared<merian::AddNode>(context, alloc);
        auto beauty_image_write = std::make_shared<merian::ImageWriteNode>(context, alloc, "image");
        auto fxaa = std::make_shared<merian::FXAA>(context, alloc);

        image_writer->set_callback([accum]() { accum->request_clear(); });
        image_writer_volume->set_callback(
            [volume_accum]() { volume_accum->request_clear(); });

        graph.add_node("one", one);
        graph.add_node("blue_noise", blue_noise);
        graph.add_node("quake", quake);
        graph.add_node("accum", accum);
        graph.add_node("denoiser", svgf);
        graph.add_node("volume denoiser", volume_svgf);
        graph.add_node("tonemap", tonemap);
        graph.add_node("image writer", image_writer);
        graph.add_node("volume image writer", image_writer_volume);
        graph.add_node("exposure", exposure);
        graph.add_node("hud", hud);
        graph.add_node("volume accum", volume_accum);
        graph.add_node("add", add);
        graph.add_node("beauty image write", beauty_image_write);
        graph.add_node("fxaa", fxaa);

        graph.connect_image(one, volume_svgf, "output", "albedo");
        graph.connect_image(blue_noise, quake, "output", "blue_noise");
        graph.connect_image(quake, accum, "irradiance", "irr");
        graph.connect_image(quake, svgf, "albedo", "albedo");
        graph.connect_image(quake, accum, "mv", "mv");
        graph.connect_image(quake, svgf, "mv", "mv");
        graph.connect_image(quake, accum, "moments", "moments_in");
        graph.connect_image(quake, volume_accum, "volume", "irr");
        graph.connect_image(quake, volume_accum, "volume_moments", "moments_in");
        graph.connect_image(quake, quake, "volume_depth", "prev_volume_depth");
        graph.connect_image(quake, volume_accum, "volume_mv", "mv");
        graph.connect_image(quake, volume_svgf, "volume_mv", "mv");
        graph.connect_buffer(quake, accum, "gbuffer", "gbuf");
        graph.connect_buffer(quake, accum, "gbuffer", "prev_gbuf");
        graph.connect_buffer(quake, quake, "gbuffer", "prev_gbuf");
        graph.connect_buffer(quake, svgf, "gbuffer", "gbuffer");
        graph.connect_buffer(quake, svgf, "gbuffer", "prev_gbuffer");
        graph.connect_buffer(quake, volume_accum, "gbuffer", "gbuf");
        graph.connect_buffer(quake, volume_accum, "gbuffer", "prev_gbuf");
        graph.connect_buffer(quake, volume_svgf, "gbuffer", "gbuffer");
        graph.connect_buffer(quake, volume_svgf, "gbuffer", "prev_gbuffer");
        graph.connect_buffer(quake, hud, "gbuffer", "gbuf");
        graph.connect_image(volume_accum, volume_accum, "out_irr", "prev_accum");
        graph.connect_image(volume_accum, volume_svgf, "out_irr", "irr");
        graph.connect_image(volume_accum, volume_accum, "out_moments", "prev_moments");
        graph.connect_image(volume_accum, volume_svgf, "out_moments", "moments");
        graph.connect_image(accum, accum, "out_irr", "prev_accum");
        graph.connect_image(accum, svgf, "out_irr", "irr");
        graph.connect_image(accum, accum, "out_moments", "prev_moments");
        graph.connect_image(accum, svgf, "out_moments", "moments");
        graph.connect_image(volume_svgf, volume_svgf, "out", "prev_out");
        graph.connect_image(volume_svgf, add, "out", "a");
        graph.connect_image(volume_svgf, image_writer_volume, "out", "src");
        graph.connect_image(svgf, quake, "out", "prev_filtered");
        graph.connect_image(svgf, svgf, "out", "prev_out");
        graph.connect_image(svgf, image_writer, "out", "src");
        graph.connect_image(svgf, add, "out", "b");
        graph.connect_image(add, exposure, "output", "src");
        graph.connect_image(exposure, tonemap, "output", "src");
        graph.connect_image(tonemap, fxaa, "output", "in");
        graph.connect_image(fxaa, hud, "out", "src");
        graph.connect_image(fxaa, beauty_image_write, "out", "src");
    }

    // Outputs the final image
    void add_beauty_output(const std::string& name,
                           const merian::NodeHandle& node,
                           const std::string& node_input) {
        graph.add_node(name, node);
        graph.connect_image(hud, node, "output", node_input);
    }

    // Outputs the surfaces only
    void add_surface_output(const std::string& name,
                            const merian::NodeHandle& node,
                            const std::string& node_input) {
        graph.add_node(name, node);
        graph.connect_image(svgf, node, "out", node_input);
    }

    // Outputs the volumes only
    void add_volume_output(const std::string& name,
                           const merian::NodeHandle& node,
                           const std::string& node_input) {
        graph.add_node(name, node);
        graph.connect_image(volume_svgf, node, "out", node_input);
    }

    merian::Graph& get() {
        return graph;
    }

  private:
    merian::Graph graph;

    merian::NodeHandle hud;
    merian::NodeHandle svgf;
    merian::NodeHandle volume_svgf;
};
