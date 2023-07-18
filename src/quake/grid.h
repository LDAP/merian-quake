#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec3 = glm::vec3;

#endif

#define STATES_PER_CELL 1000
#define BUFFER_SIZE 2000
#define GRID_WIDTH 10.

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
};
