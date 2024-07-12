#include "hud.hpp"
#include "hud.comp.spv.h"
#include "merian/utils/glm.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"

extern "C" {

#include "quakedef.h"

// from gl_rmain.c
extern mleaf_t* r_viewleaf;
}

namespace merian {

QuakeHud::QuakeHud(const SharedContext context) : AbstractCompute(context, sizeof(PushConstant)) {
    shader =
        std::make_shared<ShaderModule>(context, merian_hud_comp_spv_size(), merian_hud_comp_spv());
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
}

QuakeHud::~QuakeHud() {}

std::vector<merian_nodes::InputConnectorHandle> QuakeHud::describe_inputs() {
    return {
        con_src,
        merian_nodes::ManagedVkBufferIn::compute_read("gbuf"),
    };
}

std::vector<merian_nodes::OutputConnectorHandle>
QuakeHud::describe_outputs(const merian_nodes::ConnectorIOMap& output_for_input) {
    extent = output_for_input[con_src]->create_info.extent;
    return {
        merian_nodes::ManagedVkImageOut::compute_write("output", vk::Format::eR16G16B16A16Sfloat,
                                                       extent),
    };
}

SpecializationInfoHandle
QuakeHud::get_specialization_info([[maybe_unused]] const merian_nodes::NodeIO& io) const noexcept {
    return spec_info;
}

const void* QuakeHud::get_push_constant([[maybe_unused]] merian_nodes::GraphRun& run,
                                        [[maybe_unused]] const merian_nodes::NodeIO& io) {
    if (cl.worldmodel && sv_player && cl.intermission == 0) {
        // Demos do not have a player set
        pc.health = sv_player->v.health;
        pc.armor = sv_player->v.armorvalue;
        pc.blend = *merian::as_vec4(v_blend);

        pc.effect = 0;
        if (r_viewleaf) {
            if (r_viewleaf->contents == CONTENTS_WATER) {
                pc.effect = 1;
            } else if (r_viewleaf->contents == CONTENTS_LAVA) {
                pc.effect = 2;
            } else if (r_viewleaf->contents == CONTENTS_SLIME) {
                pc.effect = 3;
            }
        }
    } else {
        pc.health = 0;
        pc.armor = 0;
        pc.effect = 0;
        pc.blend = glm::vec4(0);
    }

    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t>
QuakeHud::get_group_count([[maybe_unused]] const merian_nodes::NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle QuakeHud::get_shader_module() {
    return shader;
}

QuakeHud::NodeStatusFlags QuakeHud::properties(Properties& config) {
    config.output_text(
        fmt::format("blend: ({}, {}, {}, {})", pc.blend.r, pc.blend.g, pc.blend.b, pc.blend.a));

    return {};
}

} // namespace merian
