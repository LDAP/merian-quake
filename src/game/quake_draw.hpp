#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace merian_quake {

struct UIDrawCmd {
    std::array<float, 4> pos;
    std::array<float, 4> uv;
    int32_t texnum; // 0 = untextured
};

// GL-convention y-up; w<0 disables the scissor.
struct UIScissor {
    int32_t x, y, w, h;
};

// GL-convention viewport (y up from bottom).
struct UICanvas {
    float ortho_l, ortho_r, ortho_b, ortho_t;
    int32_t vx, vy, vw, vh;
};

template <typename T> struct StickyEvent {
    std::size_t first_cmd; // event applies to cmds[i] for i >= first_cmd
    T value;
};

struct UIDrawCommands {
    std::vector<UIDrawCmd> cmds;
    std::vector<StickyEvent<UICanvas>> canvas_events;
    std::vector<StickyEvent<UIScissor>> scissor_events;
    std::vector<StickyEvent<uint32_t>> color_events; // RGBA8 premultiplied

    uint32_t width = 0;
    uint32_t height = 0;

    void clear() {
        cmds.clear();
        canvas_events.clear();
        scissor_events.clear();
        color_events.clear();
    }
};

} // namespace merian_quake
