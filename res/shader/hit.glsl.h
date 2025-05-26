#ifndef _HIT_H_
#define _HIT_H_

#include "merian-shaders/types.glsl.h"

struct Hit {
    vec3 pos;
    vec3 prev_pos;
    vec3 wi;
    vec3 normal;
    uint enc_geonormal;

    // Material
    f16vec3 albedo;
    float16_t roughness;
};

struct CompressedHit {
    vec3 pos;

    f16vec3 mv;
    uint wi;
    uint normal;
    uint enc_geonormal;

    // Material
    f16vec3 albedo;
    float16_t roughness;
};

#ifndef __cplusplus

#include "merian-shaders/normal_encode.glsl"
    
void compress_hit(const Hit hit, out CompressedHit compressed_hit) {
    compressed_hit.pos = hit.pos;
    compressed_hit.mv = f16vec3(hit.pos - hit.prev_pos);
    compressed_hit.wi = geo_encode_normal(hit.wi);
    compressed_hit.normal = geo_encode_normal(hit.normal);
    compressed_hit.enc_geonormal = hit.enc_geonormal;
    compressed_hit.albedo = hit.albedo;
    compressed_hit.roughness = hit.roughness;
}

void decompress_hit(const CompressedHit compressed_hit, out Hit hit) {
    hit.pos = compressed_hit.pos;
    hit.prev_pos = compressed_hit.pos - compressed_hit.mv;
    hit.wi = geo_decode_normal(compressed_hit.wi);
    hit.normal = geo_decode_normal(compressed_hit.normal);
    hit.enc_geonormal = compressed_hit.enc_geonormal;
    hit.albedo = compressed_hit.albedo;
    hit.roughness = compressed_hit.roughness;
}

#endif

#endif // _HIT_H_
