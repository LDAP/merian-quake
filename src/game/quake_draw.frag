#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) out vec4 out_color;

// in_color is premultiplied; tex is unpremultiplied (standard RGBA8 source).
// Premultiply the sample before tinting so the result is premult for src-over.
void main() {
    vec4 t = texture(tex, in_uv);
    t.rgb *= t.a;
    out_color = in_color * t;
}
