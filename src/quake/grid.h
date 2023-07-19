#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;
using ivec3 = glm::ivec3;

#endif

#define STATES_PER_CELL 5
#define BUFFER_SIZE 200000
#define GRID_WIDTH 5.

struct MCState {
    vec3 sum_tgt;
    float sum_w;
    uint N;
    float sum_len;
    float f;
    bool sky;
};

struct GridCell {
    MCState states[STATES_PER_CELL];
    ivec3 grid_idx;
};
