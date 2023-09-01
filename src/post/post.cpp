#include "post.hpp"
#include "merian/utils/glm.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "post.comp.spv.h"
#include "quake/quake_node.hpp"

namespace merian {

QuakePost::QuakePost(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ComputeNode(context, allocator, sizeof(PushConstant)) {
    shader = std::make_shared<ShaderModule>(context, merian_post_comp_spv_size(),
                                            merian_post_comp_spv());
}

QuakePost::~QuakePost() {}

std::string QuakePost::name() {
    return "Post Processing";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
QuakePost::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {
            NodeInputDescriptorBuffer::compute_read("gbuffer"),
        },
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
QuakePost::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                            const std::vector<NodeOutputDescriptorBuffer>&) {
    extent = connected_image_outputs[0].create_info.extent;
    return {
        {
            NodeOutputDescriptorImage::compute_write("output", vk::Format::eR16G16B16A16Sfloat,
                                                     extent),
        },
        {},
    };
}

SpecializationInfoHandle QuakePost::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

const void* QuakePost::get_push_constant([[maybe_unused]] GraphRun& run) {
    float fog_density = Fog_GetDensity();
    fog_density *= fog_density;
    pc.fog = glm::vec4(*merian::as_vec3(Fog_GetColor()), fog_density);

    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> QuakePost::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle QuakePost::get_shader_module() {
    return shader;
}

void QuakePost::get_configuration(Configuration&, bool&) {}

} // namespace merian
