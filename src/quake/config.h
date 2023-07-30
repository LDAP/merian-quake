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

#define MAX_PATH_LENGHT 3


// BINDINGS

#define BINDING_VTX_BUF 0
#define BINDING_IDX_BUF 1
#define BINDING_EXT_BUF 2
#define BINDING_IMG_TEX 3
#define BINDING_TLAS 4

// Material flags

#define MAT_FLAGS_NONE 0
#define MAT_FLAGS_LAVA 1
#define MAT_FLAGS_SLIME 2
#define MAT_FLAGS_TELE 3
#define MAT_FLAGS_WATER 4
#define MAT_FLAGS_WATER_LOWER 5
#define MAT_FLAGS_SKY 6
#define MAT_FLAGS_WATERFALL 7

// Player flags

#define PLAYER_FLAGS_TORCH 1
#define PLAYER_FLAGS_UNDERWATER 2

// RT FLAGS

#define RT_FLAG_LIGHT_CACHE_TAIL 1

#endif
