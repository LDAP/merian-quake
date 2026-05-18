// Sky shading types used by quake-material.slang. The actual link-time
// `extern static const QuakeSky sky` declaration lives next to the material.
// Scene-side override modules can `import shader.quake_sky;` to get these
// types in scope, then specialize the value as e.g.:
//
//   namespace merian {
//       export static const QuakeSky sky = CubemapSky(...);
//   }

__exported import merian_shaders.utils.texture_manager;
import merian_shaders.utils.envmap;

namespace merian {

interface QuakeSky {
    float3 eval(const float3 dir, const float time, const TextureManager tm);
};

struct BlackSky : QuakeSky {
    float3 eval(const float3 dir, const float time, const TextureManager tm) {
        return float3(0);
    }
};

struct ClassicSky : QuakeSky {
    TextureID tex_bck;
    TextureID tex_fnt;

    float3 eval(const float3 dir, const float time, const TextureManager tm) {
        const float2 st = 0.5 + 1. * float2(dir.x, dir.y) / abs(dir.z);
        const float2 t = time * float2(0.12, 0.12);
        const float4 bck = tm.get_sampler2D(tex_bck).SampleLevel(st + 0.5 * t, 0);
        const float4 fnt = tm.get_sampler2D(tex_fnt).SampleLevel(st + t, 0);
        const float3 tex = lerp(bck.rgb, fnt.rgb, fnt.a);

        return 10.0 * (exp2(3.5 * tex) - 1.0);
    }
};

struct CubemapSky : QuakeSky {
    TextureID sky_rt;
    TextureID sky_bk;
    TextureID sky_lf;
    TextureID sky_ft;
    TextureID sky_up;
    TextureID sky_dn;

    float3 eval(const float3 dir, const float time, const TextureManager tm) {
        TextureID tex_id = 0;
        float2 st;

        switch (cubemap_side(dir)) {
        case 0: {
            tex_id = sky_rt;
            st = 0.5 + 0.5 * float2(-dir.y, -dir.z) / abs(dir.x);
            break;
        }
        case 1: {
            tex_id = sky_lf;
            st = 0.5 + 0.5 * float2(dir.y, -dir.z) / abs(dir.x);
            break;
        }
        case 2: {
            tex_id = sky_bk;
            st = 0.5 + 0.5 * float2(dir.x, -dir.z) / abs(dir.y);
            break;
        }
        case 3: {
            tex_id = sky_ft;
            st = 0.5 + 0.5 * float2(-dir.x, -dir.z) / abs(dir.y);
            break;
        }
        case 4: {
            tex_id = sky_up;
            st = 0.5 + 0.5 * float2(-dir.y, dir.x) / abs(dir.z);
            break;
        }
        case 5: {
            tex_id = sky_dn;
            st = 0.5 + 0.5 * float2(-dir.y, -dir.x) / abs(dir.z);
            break;
        }
        }

        return tm.get_sampler2D(tex_id).SampleLevel(st, 0).rgb;
    }
};

}
