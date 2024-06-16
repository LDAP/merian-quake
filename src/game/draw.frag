#version 460
#extension GL_ARB_separate_shader_objects   : enable
#extension GL_ARB_shading_language_420pack  : enable
#extension GL_EXT_nonuniform_qualifier      : enable

layout (push_constant) uniform PushConsts {
    mat4  mvp;
} pc;

// same as in gl_texmgr.c
#define MAX_GLTEXTURES 4096
#define MAX_GEOMETRIES 16

layout (set = 0, binding = 0) uniform sampler2D textures[MAX_GLTEXTURES];

layout (location = 0) flat in uint in_texnum;
layout (location = 1) in vec2 in_texcoord;

layout (location = 0) out vec4 out_frag_color;

void main ()
{
    out_frag_color = texture(textures[nonuniformEXT(in_texnum)], in_texcoord.xy);
}
