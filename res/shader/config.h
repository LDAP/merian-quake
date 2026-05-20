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

// Material types — values for brush variants alias to Quake's SURF_DRAW* bits
// (see gl_model.h) so msurface_t::flags can be assigned directly.

#define MAT_TYPE_NONE 0
#define MAT_TYPE_SKY 0x4      // SURF_DRAWSKY
#define MAT_TYPE_LAVA 0x400   // SURF_DRAWLAVA
#define MAT_TYPE_SLIME 0x800  // SURF_DRAWSLIME
#define MAT_TYPE_TELE 0x1000  // SURF_DRAWTELE
#define MAT_TYPE_WATER 0x2000 // SURF_DRAWWATER
#define MAT_TYPE_WATERFALL 0x4000

// Surfaces that get quake_warp applied to their UVs.
#define MAT_TYPE_WARP (MAT_TYPE_LAVA | MAT_TYPE_SLIME | MAT_TYPE_TELE | MAT_TYPE_WATER)

// Player flags

#define PLAYER_FLAGS_TORCH 1
#define PLAYER_FLAGS_UNDERWATER 2

#endif
