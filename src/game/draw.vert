#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (push_constant) uniform PushConsts {
    mat4  mvp;
} pc;

layout (location = 0) in vec3 in_position;
layout (location = 1) in uint in_texnum;
layout (location = 2) in vec2 in_texcoord;

layout (location = 0) out uint out_texnum;
layout (location = 1) out vec2 out_texcoord;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main ()
{
    gl_Position = pc.mvp * vec4 (in_position, 1.0f);
    out_texcoord = in_texcoord;
}
