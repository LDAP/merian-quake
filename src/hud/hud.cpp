#include "hud.hpp"
#include "hud.comp.spv.h"
#include "merian/utils/glm.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "quake/quake_node.hpp"

namespace merian {

QuakeHud::QuakeHud(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ComputeNode(context, allocator, sizeof(PushConstant)) {
    shader =
        std::make_shared<ShaderModule>(context, merian_hud_comp_spv_size(), merian_hud_comp_spv());
}

QuakeHud::~QuakeHud() {}

std::string QuakeHud::name() {
    return "Hud";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
QuakeHud::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
QuakeHud::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                           const std::vector<NodeOutputDescriptorBuffer>&) {
    extent = connected_image_outputs[0].create_info.extent;
    return {
        {
            NodeOutputDescriptorImage::compute_write("output", vk::Format::eR16G16B16A16Sfloat, extent),
        },
        {},
    };
}

SpecializationInfoHandle QuakeHud::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

const void* QuakeHud::get_push_constant([[maybe_unused]] GraphRun& run) {
    if (sv_player) {
        // Demos do not have a player set
        pc.health = sv_player->v.health;
        pc.armor = sv_player->v.armorvalue;
        pc.blend = *merian::as_vec4(v_blend);
    } else {
        pc.health = 0;
        pc.armor = 0;
    }

    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> QuakeHud::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle QuakeHud::get_shader_module() {
    return shader;
}

void QuakeHud::get_configuration(Configuration&, bool&) {}

} // namespace merian
