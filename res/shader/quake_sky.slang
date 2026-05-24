__exported import merian_shaders.scene.environment_map;

import merian_shaders.utils.envmap;
import merian_shaders.distributions.von_mises_fisher;

namespace merian {

float3 quake_sun(const float3 dir, const float3 sun_dir, const float3 sun_color) {
    float3 sun = float3(0);
    sun += 0.5hf * pow(0.5hf * (1.0hf + float16_t(dot(sun_dir, dir))), 4.0hf);
    sun += 5.0hf * float16_t(vmf_pdf_tokuyoshi(dir, sun_dir, 3000.0));
    return sun * sun_color;
}

struct QuakeClassicSkyEnvMap : IEnvMap {
    TextureID solid;
    TextureID alpha;
    float3 sun_dir;
    float3 sun_color;

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        const float3 dir = -sd.wi;
        const float2 st = 0.5 + float2(dir.x, dir.y) / abs(dir.z);
        const float2 t = sd.time * float2(0.12, 0.12);
        const float4 bck = lod.sample(tm.get_sampler2D(solid), st + 0.5 * t);
        const float4 fnt = lod.sample(tm.get_sampler2D(alpha), st + t);
        const float3 tex = lerp(bck.rgb, fnt.rgb, fnt.a);
        float3 emission = 10.0 * (exp2(3.5 * tex) - 1.0);
        if (sd.bounce > 0) {
            emission += quake_sun(dir, sun_dir, sun_color);
        }
        return emission;
    }
}

struct QuakeCubemapSkyEnvMap : IEnvMap {
    TextureID rt;
    TextureID bk;
    TextureID lf;
    TextureID ft;
    TextureID up;
    TextureID dn;
    float3 sun_dir;
    float3 sun_color;

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        const float3 dir = -sd.wi;
        TextureID id = rt;
        float2 uv;

        switch (cubemap_side(dir)) {
        case 0: {
            id = rt;
            uv = 0.5 + 0.5 * float2(-dir.y, -dir.z) / abs(dir.x);
            break;
        }
        case 1: {
            id = lf;
            uv = 0.5 + 0.5 * float2(dir.y, -dir.z) / abs(dir.x);
            break;
        }
        case 2: {
            id = bk;
            uv = 0.5 + 0.5 * float2(dir.x, -dir.z) / abs(dir.y);
            break;
        }
        case 3: {
            id = ft;
            uv = 0.5 + 0.5 * float2(-dir.x, -dir.z) / abs(dir.y);
            break;
        }
        case 4: {
            id = up;
            uv = 0.5 + 0.5 * float2(-dir.y, dir.x) / abs(dir.z);
            break;
        }
        case 5: {
            id = dn;
            uv = 0.5 + 0.5 * float2(-dir.y, -dir.x) / abs(dir.z);
            break;
        }
        }

        float3 emission = lod.sample(tm.get_sampler2D(id), uv).rgb;
        if (sd.bounce > 0) {
            emission += quake_sun(dir, sun_dir, sun_color);
        }
        return emission;
    }
}

}
