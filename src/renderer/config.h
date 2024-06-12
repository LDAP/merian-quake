#ifndef _QUAKE_CONFIG_H_
#define _QUAKE_CONFIG_H_

// same as in gl_texmgr.c
#define MAX_GLTEXTURES 4096
#define MAX_GEOMETRIES 16

// Configure ray tracing

// max ray tracing distance.
#define T_MAX 10000.0
// continue tracing if alpha of texture is smaller
#define ALPHA_THRESHOLD .666
// A ray may travel through multiple intersections
// for example transparent surfaces / water
#define MAX_INTERSECTIONS 5

// Prevent overflows in float_16
#define MAX_SUN_COLOR 20.f

// should match DISTANCE_MC_VERTEX_STATE_COUNT (only increase for testing purposes)
#define MAX_DISTANCE_MC_VERTEX_STATE_COUNT 10

// BINDINGS

#define BINDING_VTX_BUF 0
#define BINDING_IDX_BUF 1
#define BINDING_EXT_BUF 2
#define BINDING_IMG_TEX 3
#define BINDING_TLAS 4
#define BINDING_PREV_VTX_BUF 5

// Material flags

#define MAT_FLAGS_NONE 0
#define MAT_FLAGS_LAVA 1
#define MAT_FLAGS_SLIME 2
#define MAT_FLAGS_TELE 3
#define MAT_FLAGS_WATER 4
#define MAT_FLAGS_SKY 5
#define MAT_FLAGS_WATERFALL 6
#define MAT_FLAGS_SPRITE 7
// material has a solid color. n0 is albedo, n1 is emission (tex can still be used for alpha)
#define MAT_FLAGS_SOLID 8

// Player flags

#define PLAYER_FLAGS_TORCH 1
#define PLAYER_FLAGS_UNDERWATER 2

#endif
