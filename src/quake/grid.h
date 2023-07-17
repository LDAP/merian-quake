
#define STATES_PER_CELL 100
#define BUFFER_SIZE 10000

struct MCState {
    int test;
};

struct GridCell {
    MCState states[STATES_PER_CELL];
};
