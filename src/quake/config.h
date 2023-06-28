#ifndef _QUAKE_CONFIG_H_
#define _QUAKE_CONFIG_H_

#define WATER_DEPTH 16.0 // 8.0
// water does not really work with stock quake maps (because r_vis 1 etc)
#define WATER_MODE_OFF 0     // opaque water
#define WATER_MODE_FLAT 1    // flat but transparent with "caustics"
#define WATER_MODE_NORMALS 2 // flat but with normals and transparent
#define WATER_MODE_FULL 3    // full blown animated procedural displacement
#define WATER_MODE WATER_MODE_OFF

#define T_MAX 10000.0 // max ray tracing distance. sky is that -1

#define MCMC_KAPPA 1
#define MCMC_ML 2
#define MCMC_ADAPTATION MCMC_ML

#define BINDING_VTX_BUF 0
#define BINDING_IDX_BUF 1
#define BINDING_EXT_BUF 2
#define BINDING_IMG_TEX 3
#define BINDING_TLAS 4

#define ARRAY_IDX_STATIC 0
#define ARRAY_IDX_DYNAMIC 1

#endif
