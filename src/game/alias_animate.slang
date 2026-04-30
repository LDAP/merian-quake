[vk::constant_id(0)] const int workgroup_size_x = 64;

import merian_shaders.utils.encoding;
import merian_shaders.scene.scene_data;

using merian;

struct AliasMeshDesc {
    float s;
    float t;
    uint vertindex;
};

[vk::binding(0)] StructuredBuffer<uint, ScalarDataLayout> pose_data;
[vk::binding(1)] StructuredBuffer<AliasMeshDesc, ScalarDataLayout> mesh_desc;
[vk::binding(2)] StructuredBuffer<float3, ScalarDataLayout> normal_table;
[vk::binding(3)] RWStructuredBuffer<PackedVertexData, ScalarDataLayout> dst_vertices;
[vk::binding(4)] RWStructuredBuffer<PackedPrevVertexData, ScalarDataLayout> dst_prev_vertices;

struct PushConstant {
    float3 scale;
    uint pose1_offset;
    float3 scale_origin;
    uint pose2_offset;
    float blend;
    float prev_blend;
    float inv_skin_w;
    float inv_skin_h;
    uint vertex_count;
};

[vk::push_constant] ConstantBuffer<PushConstant> pc;

void unpack_trivertx(uint packed, out float3 pos, out uint normal_idx) {
    pos = float3(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF);
    normal_idx = (packed >> 24) & 0xFF;
}

[numthreads(workgroup_size_x, 1, 1)][shader("compute")] void main(uint id : SV_DispatchThreadID) {
    if (id >= pc.vertex_count)
        return;

    const AliasMeshDesc desc = mesh_desc[id];

    float3 p1, p2;
    uint ni1, ni2;
    unpack_trivertx(pose_data[pc.pose1_offset + desc.vertindex], p1, ni1);
    unpack_trivertx(pose_data[pc.pose2_offset + desc.vertindex], p2, ni2);

    const float3 model_pos = lerp(p1, p2, pc.blend) * pc.scale + pc.scale_origin;
    const float3 prev_model_pos = lerp(p1, p2, pc.prev_blend) * pc.scale + pc.scale_origin;

    const float3 n1 = normal_table[ni1];
    const float3 n2 = normal_table[ni2];
    const float3 model_n = normalize(lerp(n1, n2, pc.blend) / pc.scale);

    PackedVertexData out_v;
    out_v.position = model_pos;
    out_v.encoded_normal = encode_normal(model_n);
    out_v.uv = half2(half((desc.s + 0.5f) * pc.inv_skin_w), half((desc.t + 0.5f) * pc.inv_skin_h));
    out_v.encoded_tangent = 0;
    dst_vertices[id] = out_v;

    PackedPrevVertexData out_prev;
    out_prev.position = prev_model_pos;
    dst_prev_vertices[id] = out_prev;
}
