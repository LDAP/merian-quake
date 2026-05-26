#include "game/quake_ui_node.hpp"

#include "quake_draw.frag.spv.h"
#include "quake_draw.vert.spv.h"

#include "merian/shader/entry_point.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline_graphics_builder.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <array>
#include <cstdint>

namespace merian_quake {

namespace {

struct PushConstants {
    std::array<float, 2> scale;
    std::array<float, 2> translate;
};

// Quake's GL canvas is y-up with ortho_b > ortho_t; VK framebuffer is y-down.
std::pair<float, float> resolve_canvas(const UICanvas& c, const float lx, const float ly,
                                       const float fb_height) {
    const float fx = (lx - c.ortho_l) / (c.ortho_r - c.ortho_l);
    const float fy = (ly - c.ortho_t) / (c.ortho_b - c.ortho_t);
    const float px = static_cast<float>(c.vx) + fx * static_cast<float>(c.vw);
    const float py_gl = static_cast<float>(c.vy) + (1.0f - fy) * static_cast<float>(c.vh);
    return {px, fb_height - py_gl};
}

bool scissor_eq(const UIScissor& a, const UIScissor& b) {
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
}

} // namespace

QuakeUiNode::QuakeUiNode() : Node() {}

QuakeUiNode::~QuakeUiNode() = default;

std::vector<merian::InputConnectorDescriptor> QuakeUiNode::describe_inputs() {
    return {
        {"ui_draw_commands", con_ui_draw_commands},
        {"scene", con_scene},
        {"extent_ref", con_extent_ref},
    };
}

std::vector<merian::OutputConnectorDescriptor>
QuakeUiNode::describe_outputs(const merian::NodeIOLayout& io_layout) {
    extent = io_layout[con_extent_ref]->get_create_info_or_throw().extent;
    con_out = merian::ManagedVkImageOut::color_attachment(vk::Format::eR8G8B8A8Unorm, extent);
    return {{"output", con_out}};
}

void QuakeUiNode::initialize(const merian::ContextHandle& context,
                             const merian::ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
}

void QuakeUiNode::ensure_pipeline(const vk::Format color_format) {
    if (pipeline && current_format == color_format)
        return;

    descriptor_layout = merian::DescriptorSetLayoutBuilder()
                            .add_binding_combined_sampler()
                            .build_push_descriptor_layout(context);

    const vk::PushConstantRange pc_range{vk::ShaderStageFlagBits::eVertex, 0,
                                         sizeof(PushConstants)};
    pipeline_layout = std::make_shared<merian::PipelineLayout>(
        context, std::vector<merian::DescriptorSetLayoutHandle>{descriptor_layout},
        std::vector<vk::PushConstantRange>{pc_range});

    const auto vert = merian::EntryPoint::create(context, merian_quake_draw_vert_spv(),
                                                 merian_quake_draw_vert_spv_size(), "main",
                                                 vk::ShaderStageFlagBits::eVertex);
    const auto frag = merian::EntryPoint::create(context, merian_quake_draw_frag_spv(),
                                                 merian_quake_draw_frag_spv_size(), "main",
                                                 vk::ShaderStageFlagBits::eFragment);

    pipeline =
        merian::GraphicsPipelineBuilder()
            .set_vertex_shader(vert)
            .set_fragment_shader(frag)
            .vertex_input_add_binding({0, sizeof(Vertex), vk::VertexInputRate::eVertex})
            .vertex_input_add_attribute(
                {0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)})
            .vertex_input_add_attribute(
                {1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv)})
            .vertex_input_add_attribute(
                {2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(Vertex, rgba)})
            .rasterizer_cull_mode(vk::CullModeFlagBits::eNone)
            // premultiplied src-over: out = src + dst*(1-src.a)
            .blend_add_attachment(VK_TRUE, vk::BlendFactor::eOne,
                                  vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
                                  vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha,
                                  vk::BlendOp::eAdd)
            .dyanmic_state_add(vk::DynamicState::eViewport)
            .dyanmic_state_add(vk::DynamicState::eScissor)
            .viewport_add(1.0f, 1.0f) // overridden by dynamic state
            .build_dynamic_rendering(pipeline_layout, color_format);

    current_format = color_format;
}

void QuakeUiNode::ensure_white_texture(const merian::CommandBufferHandle& cmd) {
    if (white_texture)
        return;
    const uint32_t white_pixel = 0xFFFFFFFFu;
    white_texture = allocator->create_texture_from_rgba8(
        cmd, &white_pixel, 1, 1, vk::SamplerAddressMode::eClampToEdge, vk::Filter::eNearest,
        vk::Filter::eNearest, false, "quake_ui_white");
    cmd->barrier(white_texture->get_image()->barrier2(
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderSampledRead, vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eFragmentShader));
}

void QuakeUiNode::process(merian::GraphRun& run,
                          [[maybe_unused]] const merian::DescriptorSetHandle& descriptor_set,
                          const merian::NodeIO& io) {
    MERIAN_PROFILE_SCOPE_GPU(run.get_cmd(), "QuakeUiNode::process");
    const merian::CommandBufferHandle& cmd = run.get_cmd();
    const merian::ImageHandle& out_image = io[con_out].get_image(0);

    ensure_white_texture(cmd);
    ensure_pipeline(out_image->get_format());

    const std::shared_ptr<UIDrawCommands>& draw_commands = io[con_ui_draw_commands];
    const std::shared_ptr<merian::Scene>& scene = io[con_scene];

    // Scene::update bails before material_system::update while !is_ready
    // (splash/loading), but the UI textures still need uploading.
    if (scene && !scene->is_ready()) {
        const auto& tm = scene->get_texture_manager();
        if (tm)
            tm->update(cmd);
    }

    const merian::ImageViewHandle out_view = merian::ImageView::create(out_image);
    const vk::RenderingAttachmentInfo color_attachment{*out_view,
                                                       vk::ImageLayout::eColorAttachmentOptimal,
                                                       vk::ResolveModeFlagBits::eNone,
                                                       {},
                                                       {},
                                                       vk::AttachmentLoadOp::eClear,
                                                       vk::AttachmentStoreOp::eStore,
                                                       vk::ClearValue{vk::ClearColorValue{
                                                           std::array<float, 4>{0, 0, 0, 0}}}};

    const vk::Extent2D fb_extent{extent.width, extent.height};
    cmd->barrier(out_image->barrier2(vk::ImageLayout::eColorAttachmentOptimal));

    if (draw_commands->cmds.empty()) {
        cmd->begin_rendering(
            vk::RenderingInfo{{}, vk::Rect2D{{0, 0}, fb_extent}, 1, 0, color_attachment});
        cmd->end_rendering();
        return;
    }

    const float fb_h = static_cast<float>(extent.height);

    vertices.clear();
    indices.clear();
    batches.clear();
    vertices.reserve(draw_commands->cmds.size() * 4);
    indices.reserve(draw_commands->cmds.size() * 6);
    batches.reserve(16);

    UICanvas cur_canvas{0, static_cast<float>(extent.width), fb_h, 0,
                        0, 0, static_cast<int32_t>(extent.width),
                        static_cast<int32_t>(extent.height)};
    UIScissor cur_scissor{0, 0, 0, -1};
    uint32_t cur_color = 0xFFFFFFFFu;
    std::size_t evt_canvas = 0;
    std::size_t evt_color = 0;
    std::size_t evt_scissor = 0;

    int32_t batch_texnum = INT32_MIN;
    UIScissor batch_scissor{INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};

    for (std::size_t i = 0; i < draw_commands->cmds.size(); ++i) {
        while (evt_canvas < draw_commands->canvas_events.size() &&
               draw_commands->canvas_events[evt_canvas].first_cmd <= i) {
            cur_canvas = draw_commands->canvas_events[evt_canvas].value;
            ++evt_canvas;
        }
        while (evt_color < draw_commands->color_events.size() &&
               draw_commands->color_events[evt_color].first_cmd <= i) {
            cur_color = draw_commands->color_events[evt_color].value;
            ++evt_color;
        }
        while (evt_scissor < draw_commands->scissor_events.size() &&
               draw_commands->scissor_events[evt_scissor].first_cmd <= i) {
            cur_scissor = draw_commands->scissor_events[evt_scissor].value;
            ++evt_scissor;
        }

        const UIDrawCmd& q = draw_commands->cmds[i];

        if (batches.empty() || q.texnum != batch_texnum ||
            !scissor_eq(cur_scissor, batch_scissor)) {
            batches.push_back({.texnum = q.texnum,
                               .first_index = static_cast<uint32_t>(indices.size()),
                               .index_count = 0,
                               .scissor = cur_scissor});
            batch_texnum = q.texnum;
            batch_scissor = cur_scissor;
        }

        const auto [x0, y0] = resolve_canvas(cur_canvas, q.pos[0], q.pos[1], fb_h);
        const auto [x1, y1] = resolve_canvas(cur_canvas, q.pos[2], q.pos[3], fb_h);

        const auto base = static_cast<uint32_t>(vertices.size());
        vertices.push_back({.pos = {x0, y0}, .uv = {q.uv[0], q.uv[1]}, .rgba = cur_color});
        vertices.push_back({.pos = {x1, y0}, .uv = {q.uv[2], q.uv[1]}, .rgba = cur_color});
        vertices.push_back({.pos = {x1, y1}, .uv = {q.uv[2], q.uv[3]}, .rgba = cur_color});
        vertices.push_back({.pos = {x0, y1}, .uv = {q.uv[0], q.uv[3]}, .rgba = cur_color});

        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
        batches.back().index_count += 6;
    }

    const vk::DeviceSize vtx_bytes = vertices.size() * sizeof(Vertex);
    const vk::DeviceSize idx_bytes = indices.size() * sizeof(uint32_t);
    allocator->ensure_buffer_size(vertex_buffer, vtx_bytes,
                                  vk::BufferUsageFlagBits::eVertexBuffer |
                                      vk::BufferUsageFlagBits::eTransferDst,
                                  "quake_ui_vertices", merian::MemoryMappingType::NONE,
                                  std::nullopt, 1.5f);
    allocator->ensure_buffer_size(index_buffer, idx_bytes,
                                  vk::BufferUsageFlagBits::eIndexBuffer |
                                      vk::BufferUsageFlagBits::eTransferDst,
                                  "quake_ui_indices", merian::MemoryMappingType::NONE,
                                  std::nullopt, 1.5f);
    allocator->get_staging()->cmd_to_device(cmd, vertex_buffer, vertices.data(), 0, vtx_bytes);
    allocator->get_staging()->cmd_to_device(cmd, index_buffer, indices.data(), 0, idx_bytes);

    cmd->barrier(vertex_buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                                vk::PipelineStageFlagBits2::eVertexAttributeInput,
                                                vk::AccessFlagBits2::eTransferWrite,
                                                vk::AccessFlagBits2::eVertexAttributeRead));
    cmd->barrier(index_buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                               vk::PipelineStageFlagBits2::eIndexInput,
                                               vk::AccessFlagBits2::eTransferWrite,
                                               vk::AccessFlagBits2::eIndexRead));

    cmd->begin_rendering(
        vk::RenderingInfo{{}, vk::Rect2D{{0, 0}, fb_extent}, 1, 0, color_attachment});

    cmd->bind(pipeline);
    cmd->set_viewport(vk::Viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                                   static_cast<float>(extent.height), 0.0f, 1.0f});

    const PushConstants pc{.scale{2.0f / static_cast<float>(extent.width),
                                  2.0f / static_cast<float>(extent.height)},
                           .translate{-1.0f, -1.0f}};
    cmd->push_constant(pipeline, pc);

    cmd->bind_vertex_buffer(vertex_buffer);
    cmd->bind_index_buffer(index_buffer, vk::IndexType::eUint32);

    merian::ImageHandle last_bound_image;
    UIScissor last_scissor{INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
    uint32_t unique_textures = 0;
    for (const Batch& b : batches) {
        if (!scissor_eq(b.scissor, last_scissor)) {
            if (b.scissor.w <= 0 || b.scissor.h <= 0) {
                cmd->set_scissor(vk::Rect2D{{0, 0}, fb_extent});
            } else {
                // GL scissor is y-up; flip into VK y-down.
                const int32_t y_top =
                    static_cast<int32_t>(extent.height) - b.scissor.y - b.scissor.h;
                cmd->set_scissor(vk::Rect2D{{b.scissor.x, y_top},
                                            {static_cast<uint32_t>(b.scissor.w),
                                             static_cast<uint32_t>(b.scissor.h)}});
            }
            last_scissor = b.scissor;
        }

        merian::TextureHandle texture;
        if (b.texnum > 0 && scene) {
            const auto& tm = scene->get_texture_manager();
            if (tm && static_cast<uint32_t>(b.texnum) < tm->get_capacity())
                texture = tm->get_texture(static_cast<merian::TextureID>(b.texnum));
        }
        if (!texture)
            texture = white_texture;
        if (texture->get_image() != last_bound_image) {
            cmd->push_descriptor_set(
                pipeline, texture->get_descriptor_info(vk::ImageLayout::eShaderReadOnlyOptimal));
            last_bound_image = texture->get_image();
            ++unique_textures;
        }

        cmd->draw_indexed(b.index_count, 1, b.first_index, 0);
    }

    cmd->end_rendering();

    last_stats = Stats{
        .cmds = static_cast<uint32_t>(draw_commands->cmds.size()),
        .vertices = static_cast<uint32_t>(vertices.size()),
        .indices = static_cast<uint32_t>(indices.size()),
        .draw_calls = static_cast<uint32_t>(batches.size()),
        .canvas_events = static_cast<uint32_t>(draw_commands->canvas_events.size()),
        .scissor_events = static_cast<uint32_t>(draw_commands->scissor_events.size()),
        .color_events = static_cast<uint32_t>(draw_commands->color_events.size()),
        .unique_textures = unique_textures,
    };
}

QuakeUiNode::NodeStatusFlags QuakeUiNode::properties(merian::Properties& props) {
    props.output_text("Last frame:");
    props.output_text("  quads:           {}", last_stats.cmds);
    props.output_text("  vertices:        {}", last_stats.vertices);
    props.output_text("  indices:         {}", last_stats.indices);
    props.output_text("  draw calls:      {}", last_stats.draw_calls);
    props.output_text("  unique textures: {}", last_stats.unique_textures);
    props.output_text("  canvas events:   {}", last_stats.canvas_events);
    props.output_text("  scissor events:  {}", last_stats.scissor_events);
    props.output_text("  color events:    {}", last_stats.color_events);
    return {};
}

} // namespace merian_quake
