#pragma once

#include "common.cuh"
#include "physics.cuh"

// ============================================================================
// GPU Render Buffers (Linear HDR + Signed Depth + Bokeh Scatter + sRGB Output)
// ============================================================================
struct GpuRenderBuffers {
    float4* d_hdr_scene     = nullptr; // .xyz = linear HDR RGB, .w = ray hit depth z
    float4* d_bokeh_accum   = nullptr; // .xyz = scattered aperture bokeh & rain streaks, .w = alpha/weight
    uchar3* d_srgb_out      = nullptr; // 1920x1080 24-bit RGB frame
};

// ============================================================================
// Thin-Lens Circle of Confusion (Signed Pixels at 1920x1080, 85mm f/1.4)
// Negative = Foreground (z < z_focus), Positive = Background (z > z_focus)
// ============================================================================
__device__ __forceinline__ float compute_signed_coc_px(float z, const CameraParams& cam) {
    z = fmaxf(z, 0.25f);
    float raw = cam.coc_scale * (z - cam.focus_z) / z;
    // Deadzone of +-0.45 px right around the Hero Leaf focal slab for razor-sharp macro detail
    if (fabsf(raw) < 0.45f) return 0.0f;
    return clampf(raw, -cam.max_coc_px, cam.max_coc_px);
}

// 9-Blade Rounded Circular Macro Lens Aperture Edge Radius Function r_aperture(angle)
__device__ __forceinline__ float aperture_9blade_radius(float theta) {
    const float n_blades = 9.0f;
    const float sector = 6.28318530f / n_blades;
    float a = fmodf(theta + 6.28318530f, sector) - 0.5f * sector;
    // Blend 68% pure circle + 32% 9-gon chord -> authentic rounded-blade f/1.4 cinema lens bokeh!
    float poly = cosf(0.5f * sector) / fmaxf(cosf(a), 0.5f);
    return lerpf(1.0f, poly, 0.34f);
}

// ============================================================================
// Atmospheric Golden-Hour Sky, Sun Disk & Distant Misty Emerald Reed Horizon
// sun_elevation = 18 deg, warm late-afternoon Tyndall glow
// ============================================================================
__device__ __forceinline__ Vec3 eval_golden_sky_and_foliage(
    const Vec3& rd,
    const Vec3& sun_dir,
    bool include_sharp_sun_disk,
    float& out_virtual_depth
) {
    float mu = clampf(dot(rd, sun_dir), -1.0f, 1.0f);
    float elev = clampf(rd.y, -0.25f, 1.0f);

    // Rich late-afternoon pond sky: deep teal-cerulean zenith -> warm golden-amber near the 18-deg sun!
    Vec3 zenith_col(0.045f, 0.115f, 0.165f);
    Vec3 mid_sky   (0.140f, 0.215f, 0.210f);
    Vec3 horizon_warm(0.52f, 0.38f, 0.20f);

    float t_sky = powf(clampf(elev * 2.0f, 0.0f, 1.0f), 0.70f);
    Vec3 sky = lerp3(horizon_warm, mid_sky, smoothstepf(0.0f, 0.45f, t_sky));
    sky = lerp3(sky, zenith_col, smoothstepf(0.35f, 1.0f, t_sky));

    // Warm golden-hour Mie sun glow concentrated around sun_dir (not washing out opposite sky!)
    float mie_wide = powf(fmaxf(0.0f, mu), 6.0f) * 0.85f;
    float mie_core = powf(fmaxf(0.0f, mu), 36.0f) * 2.90f;
    sky += Vec3(1.55f, 0.92f, 0.32f) * (mie_wide + mie_core);

    if (include_sharp_sun_disk) {
        float sun_disk = smoothstepf(0.99925f, 0.99980f, mu);
        sky += Vec3(32.0f, 21.5f, 8.8f) * sun_disk;
    }

    out_virtual_depth = 22.0f;

    // Lush Ghibli-style distant emerald bamboo/willow/lotus bank framing the pond horizon:
    // Shades the pond water into dark emerald-ink while letting 18-deg golden sunbeams pour through!
    float bank_az = atan2f(rd.x, fmaxf(rd.z, 0.02f));
    float sun_gap = expf(-3.2f * (bank_az - atan2f(sun_dir.x, sun_dir.z)) * (bank_az - atan2f(sun_dir.x, sun_dir.z)));
    float canopy_h = (0.24f - 0.09f * sun_gap)
        + 0.055f * sinf(bank_az * 4.0f + 0.7f)
        + 0.035f * cosf(bank_az * 9.0f - 0.4f)
        + 0.045f * fbm2d(bank_az * 12.0f, 3.1f, 3);

    if (rd.z > -0.1f && elev < canopy_h + 0.045f) {
        float foliage_mask = smoothstepf(canopy_h + 0.035f, canopy_h - 0.025f, elev);
        float tex  = fbm2d(bank_az * 28.0f, elev * 34.0f, 4);
        float tex2 = fbm2d(bank_az * 64.0f + 5.2f, elev * 58.0f, 3);

        Vec3 deep_emerald_shade(0.009f, 0.042f, 0.028f);
        Vec3 rich_viridian     (0.028f, 0.115f, 0.058f);
        Vec3 sunlit_gold_leaf  (0.340f, 0.420f, 0.105f);

        float sun_kiss = powf(fmaxf(0.0f, mu), 3.0f) * smoothstepf(canopy_h - 0.11f, canopy_h + 0.02f, elev);
        Vec3 bank_col = lerp3(deep_emerald_shade, rich_viridian, clampf(tex * 1.15f, 0.0f, 1.0f));
        bank_col = lerp3(bank_col, sunlit_gold_leaf, clampf(sun_kiss * 0.85f + tex2 * sun_kiss * 0.45f, 0.0f, 1.0f));

        // Atmospheric golden-emerald mist in sun-gap
        Vec3 haze_tint = lerp3(Vec3(0.04f, 0.11f, 0.08f), Vec3(0.72f, 0.48f, 0.20f), powf(fmaxf(0.0f, mu), 3.5f));
        bank_col = lerp3(bank_col, haze_tint, 0.26f + 0.24f * sun_gap);

        // Dappled sunlit canopy glints -> bloom into soft circular background bokeh orbs!
        float glint = powf(fmaxf(0.0f, noise2d(bank_az * 82.0f, elev * 96.0f) - 0.77f) * 4.3f, 3.0f);
        bank_col += Vec3(4.2f, 3.2f, 1.2f) * glint * (0.25f + 0.85f * fmaxf(0.0f, mu));

        sky = lerp3(sky, bank_col, foliage_mask);
        if (foliage_mask > 0.25f) out_virtual_depth = 11.5f;
    }

    return sky;
}

// ============================================================================
// Bicubic / Smooth Pond Wave Height & Normal Sampler (1024x1024 Grid + Far Ripples)
// ============================================================================
__device__ __forceinline__ void sample_pond_surface(
    const float* __restrict__ wave_eta,
    const float2* __restrict__ wave_grad,
    float wx, float wz,
    float sim_time,
    float& out_h,
    Vec3& out_normal
) {
    float gx = (wx - WAVE_X_MIN) * (1.0f / WAVE_DX) - 0.5f;
    float gz = (wz - WAVE_Z_MIN) * (1.0f / WAVE_DX) - 0.5f;

    float h_sim = 0.0f;
    float dhdx = 0.0f, dhdz = 0.0f;

    // Smooth transition weight inside 1024x1024 PDE wave grid
    float wx_in = smoothstepf(4.0f, 44.0f, gx) * (1.0f - smoothstepf(float(WAVE_N - 46), float(WAVE_N - 6), gx));
    float wz_in = smoothstepf(4.0f, 44.0f, gz) * (1.0f - smoothstepf(float(WAVE_N - 46), float(WAVE_N - 6), gz));
    float w_grid = wx_in * wz_in;

    if (w_grid > 0.001f) {
        int ix = clampf(floorf(gx), 1.0f, float(WAVE_N - 3));
        int iz = clampf(floorf(gz), 1.0f, float(WAVE_N - 3));
        float fx = gx - float(ix);
        float fz = gz - float(iz);
        // Smoothstep sub-texel filter for silky-smooth liquid specular normals!
        float sx = fx * fx * (3.0f - 2.0f * fx);
        float sz = fz * fz * (3.0f - 2.0f * fz);

        int r0 = iz * WAVE_N + ix;
        int r1 = (iz + 1) * WAVE_N + ix;

        h_sim = lerpf(lerpf(wave_eta[r0], wave_eta[r0 + 1], sx),
                      lerpf(wave_eta[r1], wave_eta[r1 + 1], sx), sz);

        float2 g00 = wave_grad[r0];
        float2 g10 = wave_grad[r0 + 1];
        float2 g01 = wave_grad[r1];
        float2 g11 = wave_grad[r1 + 1];
        dhdx = lerpf(lerpf(g00.x, g10.x, sx), lerpf(g01.x, g11.x, sx), sz) * w_grid;
        dhdz = lerpf(lerpf(g00.y, g10.y, sx), lerpf(g01.y, g11.y, sz), sz) * w_grid;
        h_sim *= w_grid;
    }

    // Subtle ambient pond swell + distant sun-sparkle capillary wavelets beyond 6.5m
    float rain_act = eval_rain_intensity(sim_time);
    float swell_amp = 0.0011f + 0.0014f * rain_act;
    float p1 = wx * 4.2f + wz * 3.6f - sim_time * 1.8f;
    float p2 = wx * -5.8f + wz * 4.9f - sim_time * 2.3f;
    float p3 = wx * 13.5f + wz * 11.2f - sim_time * 4.1f;

    float far_boost = 1.0f + 1.8f * (1.0f - w_grid);
    h_sim += swell_amp * far_boost * (0.55f * sinf(p1) + 0.35f * sinf(p2) + 0.18f * sinf(p3));
    dhdx  += swell_amp * far_boost * (0.55f * 4.2f * cosf(p1) - 0.35f * 5.8f * cosf(p2) + 0.18f * 13.5f * cosf(p3));
    dhdz  += swell_amp * far_boost * (0.55f * 3.6f * cosf(p1) + 0.35f * 4.9f * cosf(p2) + 0.18f * 11.2f * cosf(p3));

    // Distant procedural interfering rain ring wavelets for rich golden bokeh pond sparkles
    if (w_grid < 0.95f) {
        float cell_x = floorf(wx * 1.5f);
        float cell_z = floorf(wz * 1.5f);
        float h_seed = hash21(cell_x, cell_z);
        float ring_t = fmodf(sim_time * 0.85f + h_seed * 7.3f, 2.2f);
        Vec2 rc((cell_x + 0.3f + 0.4f * h_seed) / 1.5f, (cell_z + 0.5f) / 1.5f);
        float rr = sqrtf((wx - rc.x) * (wx - rc.x) + (wz - rc.y) * (wz - rc.y) + 1e-5f);
        float wave_front = rr - ring_t * 0.38f;
        float env = expf(-18.0f * wave_front * wave_front) * expf(-ring_t * 1.1f) * (1.0f - w_grid) * (0.35f + 0.65f * rain_act);
        float k_ring = 48.0f;
        float d_ring = env * 0.0024f * k_ring * cosf(k_ring * wave_front);
        dhdx += d_ring * (wx - rc.x) / rr;
        dhdz += d_ring * (wz - rc.y) / rr;
    }

    out_h = h_sim;
    out_normal = normalize(Vec3(-dhdx, 1.0f, -dhdz));
}

// ============================================================================
// Bounded Exact Ray-Intersection with Sculpted Lotus Leaf Blade + Stem
// ============================================================================
struct LeafHit {
    float t;
    int leaf_id;
    float u, v;
    LeafSample sample;
    Vec3 normal;
    bool is_stem;
};

__device__ __forceinline__ bool intersect_all_lotus_leaves(
    const Vec3& ro,
    const Vec3& rd,
    const LotusLeafState* __restrict__ leaves,
    float t_min,
    float t_max,
    LeafHit& best_hit
) {
    best_hit.t = t_max;
    best_hit.leaf_id = -1;
    best_hit.is_stem = false;
    bool hit_any = false;

    for (int lid = 0; lid < NUM_LEAVES; ++lid) {
        const LotusLeafState& L = leaves[lid];

        // 1. Check gently curved lotus petiole stem beneath leaf center down into pond
        // Offset -0.011f along axis_n so stem attaches cleanly underneath central umbilicus dimple!
        if (L.world_center.y > 0.018f) {
            Vec3 p_top = L.world_center - L.axis_u * (0.015f * L.radius) - L.axis_n * 0.011f;
            Vec3 p_bot = Vec3(L.base_pos.x - 0.025f, -0.16f, L.base_pos.z + 0.025f);
            Vec3 p_mid = 0.5f * (p_top + p_bot) + Vec3(0.014f * sinf(float(lid) * 1.9f), 0.0f, -0.012f);
            Vec3 prev_pt = p_bot;
            float stem_r = 0.0046f + 0.0095f * L.radius;
            for (int seg_i = 1; seg_i <= 4; ++seg_i) {
                float ts = float(seg_i) * 0.25f;
                float omt = 1.0f - ts;
                Vec3 cur_pt = (omt * omt) * p_bot + (2.0f * omt * ts) * p_mid + (ts * ts) * p_top;
                Vec3 ba = cur_pt - prev_pt;
                Vec3 oa = ro - prev_pt;
                float baba = dot(ba, ba);
                float bard = dot(ba, rd);
                float baoa = dot(ba, oa);
                float rdoa = dot(rd, oa);
                float oaoa = dot(oa, oa);
                float a = baba - bard * bard;
                float b = baba * rdoa - baoa * bard;
                float c = baba * oaoa - baoa * baoa - stem_r * stem_r * baba;
                float h_disc = b * b - a * c;
                if (h_disc > 0.0f && fabsf(a) > 1e-7f) {
                    float t_s = (-b - sqrtf(h_disc)) / a;
                    float y_s = baoa + t_s * bard;
                    if (t_s > t_min && t_s < best_hit.t && y_s >= 0.0f && y_s <= baba) {
                        best_hit.t = t_s;
                        best_hit.leaf_id = lid;
                        best_hit.is_stem = true;
                        Vec3 hp = ro + rd * t_s;
                        Vec3 axis_pt = prev_pt + ba * (y_s / baba);
                        best_hit.normal = normalize(hp - axis_pt);
                        hit_any = true;
                    }
                }
                prev_pt = cur_pt;
            }
        }

        // 2. Tight bounding sphere test around ruffled leaf blade
        Vec3 oc = ro - L.world_center;
        float b_sph = dot(oc, rd);
        float r_bound = L.radius * 1.14f;
        float c_sph = dot(oc, oc) - r_bound * r_bound;
        float disc = b_sph * b_sph - c_sph;
        if (disc <= 0.0f) continue;

        float sq = sqrtf(disc);
        float t0 = fmaxf(t_min, -b_sph - sq);
        float t1 = fminf(best_hit.t, -b_sph + sq);
        if (t0 >= t1) continue;

        // March across the thin curved heightfield slab inside the bounding sphere
        const int STEPS = 14;
        float dt = (t1 - t0) / float(STEPS);
        float prev_t = t0;
        Vec3 p0 = ro + rd * t0 - L.world_center;
        float prev_u = dot(p0, L.axis_u);
        float prev_v = dot(p0, L.axis_v);
        LeafSample prev_ls = eval_lotus_leaf_shape(L, prev_u, prev_v);
        float prev_f = dot(p0, L.axis_n) - prev_ls.height;

        for (int s = 1; s <= STEPS; ++s) {
            float cur_t = t0 + float(s) * dt;
            Vec3 p = ro + rd * cur_t - L.world_center;
            float u = dot(p, L.axis_u);
            float v = dot(p, L.axis_v);
            LeafSample ls = eval_lotus_leaf_shape(L, u, v);
            float cur_f = dot(p, L.axis_n) - ls.height;

            // Detect zero-crossing of signed height above leaf surface
            if ((prev_f * cur_f <= 0.0f) && (fminf(prev_ls.s_norm, ls.s_norm) <= 1.02f)) {
                // 5-iteration secant/bisection polish for sub-10-micron exact intersection!
                float ta = prev_t, tb = cur_t;
                float fa = prev_f;
                LeafSample mid_ls = ls;
                float mu = u, mv = v;
                for (int k = 0; k < 5; ++k) {
                    float tm = 0.5f * (ta + tb);
                    Vec3 pm = ro + rd * tm - L.world_center;
                    mu = dot(pm, L.axis_u);
                    mv = dot(pm, L.axis_v);
                    mid_ls = eval_lotus_leaf_shape(L, mu, mv);
                    float fm = dot(pm, L.axis_n) - mid_ls.height;
                    if (fa * fm <= 0.0f) { tb = tm; }
                    else { ta = tm; fa = fm; }
                }
                float thit = 0.5f * (ta + tb);
                if (mid_ls.s_norm <= 1.0f && thit > t_min && thit < best_hit.t) {
                    best_hit.t = thit;
                    best_hit.leaf_id = lid;
                    best_hit.u = mu;
                    best_hit.v = mv;
                    best_hit.sample = mid_ls;
                    best_hit.normal = eval_lotus_leaf_world_normal(L, mu, mv);
                    best_hit.is_stem = false;
                    hit_any = true;
                    break;
                }
            }
            prev_t = cur_t;
            prev_f = cur_f;
            prev_ls = ls;
        }
    }
    return hit_any;
}

// ============================================================================
// Bounded SDF Ray-Marching for the 2 Half-Opened Pale Ivory-Pink Lotus Buds
// ============================================================================
struct BudHit {
    float t;
    int bud_id;
    BudMaterialInfo mat;
    Vec3 normal;
};

__device__ __forceinline__ bool intersect_lotus_buds(
    const Vec3& ro,
    const Vec3& rd,
    const LotusBudState* __restrict__ buds,
    float t_min,
    float t_max,
    BudHit& best_hit
) {
    best_hit.t = t_max;
    best_hit.bud_id = -1;
    bool hit_any = false;

    for (int bid = 0; bid < NUM_BUDS; ++bid) {
        const LotusBudState& B = buds[bid];
        // Enclosing sphere covering both blossom head and upper curved stem
        Vec3 bound_c = 0.5f * (B.world_pos + Vec3(0.0f, 0.06f, 0.0f) + B.stem_anchor);
        float bound_r = length(B.world_pos + Vec3(0.0f, B.height, 0.0f) - bound_c) + 0.08f;

        Vec3 oc = ro - bound_c;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - bound_r * bound_r;
        float disc = b * b - c;
        if (disc <= 0.0f) continue;

        float sq = sqrtf(disc);
        float t = fmaxf(t_min, -b - sq);
        float t_end = fminf(best_hit.t, -b + sq);
        if (t >= t_end) continue;

        for (int step = 0; step < 36; ++step) {
            Vec3 p = ro + rd * t;
            BudMaterialInfo info = eval_lotus_bud_sdf(B, p);
            if (info.dist < 0.00035f) {
                if (t < best_hit.t) {
                    best_hit.t = t;
                    best_hit.bud_id = bid;
                    best_hit.mat = info;
                    // Tetrahedral SDF gradient normal
                    const float e = 0.00045f;
                    float d1 = eval_lotus_bud_sdf(B, p + Vec3( e, -e, -e)).dist;
                    float d2 = eval_lotus_bud_sdf(B, p + Vec3(-e, -e,  e)).dist;
                    float d3 = eval_lotus_bud_sdf(B, p + Vec3(-e,  e, -e)).dist;
                    float d4 = eval_lotus_bud_sdf(B, p + Vec3( e,  e,  e)).dist;
                    best_hit.normal = normalize(Vec3(d1 - d2 - d3 + d4, -d1 - d2 + d3 + d4, -d1 + d2 - d3 + d4));
                    hit_any = true;
                }
                break;
            }
            t += fmaxf(info.dist * 0.72f, 0.0012f);
            if (t >= t_end) break;
        }
    }
    return hit_any;
}

// ============================================================================
// Focused Refractive Water-Pearl Caustics Projected onto Lotus Leaf Surface
// Queries 3x3 cells of the leaf's 64x64 pearl grid along +sun_dir!
// ============================================================================
__device__ __forceinline__ float compute_leaf_pearl_caustic(
    int lid,
    const Vec3& hit_pos,
    const Vec3& leaf_normal,
    const LotusLeafState& L,
    const Vec3& sun_dir,
    const DropletParticle* __restrict__ particles,
    const int* __restrict__ leaf_grid_head,
    const int* __restrict__ particle_next
) {
    // Tangent component of 18-deg sunbeam across the lotus blade
    Vec3 sun_tang = normalize(sun_dir - leaf_normal * dot(sun_dir, leaf_normal));

    // Query leaf (u, v) around hit_pos shifted slightly toward +sun_tang (where the focusing pearl sits!)
    Vec3 rel_hit = hit_pos - L.world_center;
    float hu = dot(rel_hit, L.axis_u);
    float hv = dot(rel_hit, L.axis_v);

    float R_grid = L.radius * 1.12f;
    int cx = int(floorf((hu + R_grid) / (2.0f * R_grid) * float(LEAF_GRID_RES)));
    int cz = int(floorf((hv + R_grid) / (2.0f * R_grid) * float(LEAF_GRID_RES)));
    if (cx < 0 || cx >= LEAF_GRID_RES || cz < 0 || cz >= LEAF_GRID_RES) return 1.0f;

    float caustic_mult = 1.0f;

    for (int dz = -2; dz <= 2; ++dz) {
        int nz = cz + dz;
        if (nz < 0 || nz >= LEAF_GRID_RES) continue;
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= LEAF_GRID_RES) continue;

            int p_idx = leaf_grid_head[(lid * LEAF_GRID_RES + nz) * LEAF_GRID_RES + nx];
            int iter = 0;
            while (p_idx != -1 && iter < 18) {
                ++iter;
                const DropletParticle& q = particles[p_idx];
                // Project pearl base onto tangent plane & compute Snell-refracted focal spot offset (-sun_tang * 1.05 * R)
                Vec3 pearl_foot = q.pos - leaf_normal * dot(q.pos - hit_pos, leaf_normal);
                Vec3 d_foot = hit_pos - pearl_foot;
                float r_foot = length(d_foot);

                // Focused paraxial caustic hotspot cast forward along -sun_tang by the spherical water lens (IOR=1.333)!
                Vec3 caustic_center = pearl_foot - sun_tang * (1.05f * q.radius);
                float r_caus = length(hit_pos - caustic_center) / fmaxf(q.radius, 1e-5f);
                float rn_base = r_foot / fmaxf(q.radius, 1e-5f);

                if (rn_base < 2.35f) {
                    // Dark refractive meniscus shadow ring where outer rays bend inward into the focal spot
                    float shadow_ring = 0.28f + 0.72f * smoothstepf(0.85f, 1.55f, rn_base);
                    // Radiant golden-emerald focal hotspot + crescent!
                    float focal_spot = 7.8f * expf(-7.5f * r_caus * r_caus)
                                     + 3.2f * expf(-22.0f * (r_caus - 0.38f) * (r_caus - 0.38f));
                    caustic_mult = fmaxf(caustic_mult * shadow_ring, 1.0f + focal_spot);
                }
                p_idx = particle_next[p_idx];
            }
        }
    }
    return caustic_mult;
}

// ============================================================================
// Shade Sculpted Lotus Leaf Blade:
// - Lush Viridian/Emerald Botanical Albedo (ZERO flat neon green!)
// - Superhydrophobic Epicuticular Wax Velvet Grazing Sheen
// - Warm Golden-Green Backlit Subsurface Translucency Thinned Along Veins
// - Focused Refractive Pearl Caustic Hotspots
// ============================================================================
__device__ __forceinline__ Vec3 shade_lotus_leaf(
    const LeafHit& hit,
    const Vec3& ro,
    const Vec3& rd,
    const LotusLeafState* __restrict__ leaves,
    const Vec3& sun_dir,
    const DropletParticle* __restrict__ particles,
    const int* __restrict__ leaf_grid_head,
    const int* __restrict__ particle_next,
    bool compute_caustics
) {
    Vec3 hit_p = ro + rd * hit.t;
    Vec3 V = -rd;
    Vec3 N = hit.normal;
    bool backside = false;
    if (dot(N, V) < 0.0f) {
        N = -N;
        backside = true;
    }

    if (hit.is_stem) {
        float ndl = fmaxf(0.0f, dot(N, sun_dir));
        float rim = powf(1.0f - fmaxf(0.0f, dot(N, V)), 3.0f);
        Vec3 stem_alb(0.045f, 0.115f, 0.048f);
        return stem_alb * (0.28f + 1.85f * ndl) + Vec3(0.32f, 0.42f, 0.14f) * rim * 0.45f;
    }

    const LotusLeafState& L = leaves[hit.leaf_id];

    // Procedural botanical micro-cellular variation & fine tertiary vein network
    float micro_n = fbm2d(hit.u * 68.0f + float(hit.leaf_id) * 3.7f, hit.v * 68.0f, 3);
    float micro_pap = noise2d(hit.u * 380.0f, hit.v * 380.0f) - 0.5f;

    // Perturb normal subtly for waxy cellular micro-relief
    N = normalize(N + (L.axis_u * micro_pap + L.axis_v * (micro_n - 0.5f)) * 0.045f);

    // Rich Ghibli-meets-BBC-Earth botanical emerald palette (viridian jade -> warm sunlit olive -> citron veins)
    Vec3 deep_emerald(0.022f, 0.098f, 0.064f);
    Vec3 mid_jade    (0.048f, 0.168f, 0.082f);
    Vec3 vein_citron (0.215f, 0.305f, 0.095f);
    Vec3 umbil_cream (0.440f, 0.435f, 0.210f);

    Vec3 albedo = lerp3(deep_emerald, mid_jade, clampf(micro_n * 1.15f + 0.15f * hit.sample.s_norm, 0.0f, 1.0f));
    albedo = lerp3(albedo, vein_citron, hit.sample.vein_mask * 0.78f);
    albedo = lerp3(albedo, umbil_cream, hit.sample.umbilicus * 0.88f);
    // Delicate warm russet-gold organic botanical margin on the outermost 2% rim
    float rim_wear = smoothstepf(0.965f, 1.0f, hit.sample.s_norm);
    albedo = lerp3(albedo, Vec3(0.24f, 0.16f, 0.055f), rim_wear * 0.65f);
    if (backside) {
        // Underside of real lotus leaf has paler sage-celadon wax bloom
        albedo = lerp3(albedo, Vec3(0.11f, 0.19f, 0.13f), 0.42f);
    }

    // Focused pearl caustics on top sunlit leaf surface
    float caustic = 1.0f;
    if (compute_caustics && !backside && particles != nullptr) {
        caustic = compute_leaf_pearl_caustic(
            hit.leaf_id, hit_p, N, L, sun_dir, particles, leaf_grid_head, particle_next
        );
    }

    // 1. Warm Golden-Hour Direct Solar Illumination (sun_elevation = 18 deg)
    const Vec3 sun_radiance(3.65f, 2.72f, 1.48f);
    float ndotl = fmaxf(0.0f, dot(N, sun_dir));

    // Soft self-shadow check from raised ruffled rim / neighboring leaves
    float sun_vis = 1.0f;
    if (ndotl > 0.01f || true) {
        LeafHit sh_hit;
        if (intersect_all_lotus_leaves(hit_p + N * 0.0025f, sun_dir, leaves, 0.004f, 2.5f, sh_hit)) {
            // Leaves are translucent! Even shadowed regions receive warm green-gold filtered light!
            sun_vis = 0.28f;
        }
    }

    Vec3 direct_diff = albedo * sun_radiance * (ndotl * sun_vis * caustic);

    // 2. Superhydrophobic Micro-Papillae Wax Velvet Sheen (Epicuticular Nanotubule Grazing Glow!)
    float ndotv = clampf(dot(N, V), 0.0f, 1.0f);
    float grazing = powf(1.0f - ndotv, 3.3f);
    Vec3 H = normalize(sun_dir + V);
    float ndoth = fmaxf(0.0f, dot(N, H));
    // Broad velvety wax lobe + silvery-celadon superhydrophobic air-plastron rim sheen
    Vec3 wax_spec = Vec3(0.32f, 0.40f, 0.36f) * (0.14f * powf(ndoth, 9.0f) * ndotl * sun_vis)
                  + Vec3(0.22f, 0.34f, 0.29f) * (grazing * (0.35f + 0.65f * fmaxf(0.0f, dot(V, -sun_dir) * 0.5f + 0.5f)));

    // 3. Backlit Subsurface Translucency (Warm Golden-Green Transmission Thinned Along Radial Veins!)
    // Optical thickness thins along primary/secondary vein channels and ruffled outer blade
    float opt_thick = lerpf(1.35f, 0.62f, clampf(hit.sample.vein_mask * 0.75f + 0.35f * hit.sample.s_norm, 0.0f, 1.0f));
    float forward_hg = powf(clampf(dot(-V, sun_dir) * 0.5f + 0.5f, 0.0f, 1.0f), 2.6f);
    float trans_wrap = clampf(-dot(N, sun_dir) * 0.65f + 0.55f, 0.0f, 1.0f);
    Vec3 sss_tint = lerp3(Vec3(0.28f, 0.54f, 0.085f), Vec3(0.68f, 0.66f, 0.16f), hit.sample.vein_mask);
    Vec3 sss_glow = sss_tint * sun_radiance * expf(-1.25f * opt_thick)
                  * (0.22f + 1.05f * forward_hg) * trans_wrap * (0.45f + 0.55f * sun_vis);

    // 4. Hemispherical Sky & Emerald Pond Bounce Ambient
    Vec3 sky_amb = Vec3(0.18f, 0.26f, 0.28f) * (0.5f + 0.5f * N.y);
    Vec3 pond_bounce = Vec3(0.04f, 0.14f, 0.09f) * (0.5f - 0.5f * N.y);
    Vec3 ambient = albedo * (sky_amb + pond_bounce) * 1.55f;

    // 5. Golden Rim-Lighting along Ruffled Leaf Margin
    float edge_rim = smoothstepf(0.88f, 1.0f, hit.sample.s_norm) * powf(1.0f - ndotv, 2.0f)
                   * (0.3f + 0.7f * forward_hg);
    Vec3 margin_rim = Vec3(1.25f, 0.96f, 0.36f) * edge_rim * 0.52f * sun_vis;

    return direct_diff + wax_spec + sss_glow + ambient + margin_rim;
}

// ============================================================================
// Shade Sacred Half-Opened Pale Ivory-Pink Lotus Bud (Nelumbo nucifera)
// ============================================================================
__device__ __forceinline__ Vec3 shade_lotus_bud(
    const BudHit& hit,
    const Vec3& ro,
    const Vec3& rd,
    const Vec3& sun_dir
) {
    Vec3 V = -rd;
    Vec3 N = hit.normal;
    if (dot(N, V) < 0.0f) N = -N;

    const Vec3 sun_rad(3.7f, 2.75f, 1.50f);
    float ndotl = fmaxf(0.0f, dot(N, sun_dir));
    float ndotv = clampf(dot(N, V), 0.0f, 1.0f);
    float forward_sun = powf(clampf(dot(-V, sun_dir) * 0.5f + 0.5f, 0.0f, 1.0f), 2.4f);

    if (hit.mat.part_id == 2) {
        // Curved lotus bud stem
        Vec3 stem_col(0.048f, 0.125f, 0.052f);
        return stem_col * (Vec3(0.24f, 0.30f, 0.28f) + sun_rad * ndotl * 0.65f)
             + Vec3(0.45f, 0.52f, 0.18f) * powf(1.0f - ndotv, 3.2f) * 0.35f;
    }
    if (hit.mat.part_id == 3) {
        // Warm golden stamen heart glowing inside half-parted petals
        return Vec3(1.15f, 0.72f, 0.14f) * (0.45f + 0.85f * ndotl + 0.65f * forward_sun);
    }

    // Botanical Nelumbo nucifera petal gradient:
    // Creamy warm ivory-jade base -> soft shell blush -> luminous rose-carmine acuminate tip!
    float h = clampf(hit.mat.petal_h_norm, 0.0f, 1.0f);
    Vec3 ivory_base(0.62f, 0.56f, 0.43f);
    Vec3 blush_mid (0.74f, 0.36f, 0.42f);
    Vec3 rose_tip  (0.68f, 0.11f, 0.24f);

    if (hit.mat.part_id == 1) {
        // Outer sepaloid base tinged with delicate celadon-jade
        ivory_base = Vec3(0.24f, 0.36f, 0.19f);
    }

    Vec3 petal_alb = lerp3(ivory_base, blush_mid, smoothstepf(0.12f, 0.58f, h));
    petal_alb = lerp3(petal_alb, rose_tip, smoothstepf(0.52f, 0.96f, h));
    // Delicate longitudinal petal silk venation & petal-edge crease shading
    petal_alb *= (0.88f + 0.14f * hit.mat.rib_mask);

    // Translucent porcelain-silk petal subsurface scattering + golden-hour rim light!
    float wrap_diff = clampf(dot(N, sun_dir) * 0.55f + 0.45f, 0.0f, 1.0f);
    Vec3 sss_petal = lerp3(Vec3(0.68f, 0.44f, 0.28f), Vec3(0.75f, 0.18f, 0.28f), h)
                   * sun_rad * (0.18f + 0.62f * forward_sun) * (0.30f + 0.70f * h) * 0.36f;

    float silk_rim = powf(1.0f - ndotv, 3.0f);
    Vec3 rim_col = Vec3(1.15f, 0.78f, 0.45f) * silk_rim * (0.30f + 0.70f * forward_sun) * 0.46f;

    Vec3 H = normalize(sun_dir + V);
    float spec = powf(fmaxf(0.0f, dot(N, H)), 24.0f) * ndotl * 0.14f;

    // Inter-whorl ambient occlusion so overlapping petals have sculptural depth
    float ao = lerpf(0.58f, 1.0f, smoothstepf(0.04f, 0.65f, h)) * lerpf(1.0f, 0.74f, hit.mat.whorl_norm);

    return (petal_alb * (Vec3(0.16f, 0.19f, 0.21f) + sun_rad * wrap_diff * 0.48f) * ao)
         + sss_petal + rim_col + Vec3(0.9f, 0.8f, 0.65f) * spec;
}

// ============================================================================
// Exact Ray-Metaball / Oblate-Ellipsoid Water Pearl Intersection on Leaf & Pond
// Smoothly blends coalescing droplets (sminf) with IOR = 1.333 Fresnel & Refraction!
// ============================================================================
struct PearlHit {
    float t;
    Vec3 normal;
    int leaf_id; // >=0 if sitting on a lotus leaf, -1 if crown splash bead in air
    float radius;
};

__device__ __forceinline__ float eval_leaf_pearls_metaball_sdf(
    const Vec3& p,
    int lid,
    const LotusLeafState& L,
    const DropletParticle* __restrict__ particles,
    const int* __restrict__ leaf_grid_head,
    const int* __restrict__ particle_next,
    float& out_dom_radius
) {
    Vec3 rel = p - L.world_center;
    float u = dot(rel, L.axis_u);
    float v = dot(rel, L.axis_v);
    float R_grid = L.radius * 1.12f;
    int cx = int(floorf((u + R_grid) / (2.0f * R_grid) * float(LEAF_GRID_RES)));
    int cz = int(floorf((v + R_grid) / (2.0f * R_grid) * float(LEAF_GRID_RES)));
    if (cx < 0 || cx >= LEAF_GRID_RES || cz < 0 || cz >= LEAF_GRID_RES) return 1.0f;

    float d_blend = 1.0f;
    out_dom_radius = 0.001f;

    for (int dz = -1; dz <= 1; ++dz) {
        int nz = cz + dz;
        if (nz < 0 || nz >= LEAF_GRID_RES) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= LEAF_GRID_RES) continue;

            int idx = leaf_grid_head[(lid * LEAF_GRID_RES + nz) * LEAF_GRID_RES + nx];
            int iter = 0;
            while (idx != -1 && iter < 16) {
                ++iter;
                const DropletParticle& q = particles[idx];
                // Superhydrophobic oblate sessile pearl: wider horizontally, slightly flattened along leaf normal!
                Vec3 dp = p - q.pos;
                float dn = dot(dp, L.axis_n);
                Vec3 dtang = dp - L.axis_n * dn;
                // Gravity + contact wobble oblateness (larger pearls flatten into lustrous liquid cushions!)
                float oblate = 1.0f + 42.0f * q.radius + 0.25f * q.squash * sinf(q.age * 38.0f);
                float scaled_n = dn * oblate;
                float dist_ellip = (sqrtf(dot(dtang, dtang) + scaled_n * scaled_n + 1e-12f) - q.radius) / oblate;

                if (dist_ellip < d_blend) out_dom_radius = q.radius;
                // Smooth liquid meniscus necking between touching/coalescing pearls!
                d_blend = sminf(d_blend, dist_ellip, 0.00085f);
                idx = particle_next[idx];
            }
        }
    }
    return d_blend;
}

__device__ __forceinline__ bool intersect_leaf_water_pearls(
    const Vec3& ro,
    const Vec3& rd,
    int lid,
    float t_leaf_hit,
    const LotusLeafState& L,
    const DropletParticle* __restrict__ particles,
    const int* __restrict__ leaf_grid_head,
    const int* __restrict__ particle_next,
    PearlHit& out_hit
) {
    // Pearls sit within ~14mm above the leaf surface along the ray just before t_leaf_hit
    float cos_inc = fmaxf(0.16f, fabsf(dot(rd, L.axis_n)));
    float window = 0.020f / cos_inc;
    float t_start = fmaxf(0.05f, t_leaf_hit - window);
    float t_end   = t_leaf_hit + 0.0010f;
    if (t_start >= t_end) return false;

    float t = t_start;
    float dom_r = 0.001f;
    for (int i = 0; i < 26; ++i) {
        Vec3 p = ro + rd * t;
        float d = eval_leaf_pearls_metaball_sdf(p, lid, L, particles, leaf_grid_head, particle_next, dom_r);
        if (d < 0.00008f) {
            out_hit.t = t;
            out_hit.leaf_id = lid;
            out_hit.radius = dom_r;
            const float e = 0.00014f;
            float r_dummy;
            float d1 = eval_leaf_pearls_metaball_sdf(p + Vec3( e, -e, -e), lid, L, particles, leaf_grid_head, particle_next, r_dummy);
            float d2 = eval_leaf_pearls_metaball_sdf(p + Vec3(-e, -e,  e), lid, L, particles, leaf_grid_head, particle_next, r_dummy);
            float d3 = eval_leaf_pearls_metaball_sdf(p + Vec3(-e,  e, -e), lid, L, particles, leaf_grid_head, particle_next, r_dummy);
            float d4 = eval_leaf_pearls_metaball_sdf(p + Vec3( e,  e,  e), lid, L, particles, leaf_grid_head, particle_next, r_dummy);
            out_hit.normal = normalize(Vec3(d1 - d2 - d3 + d4, -d1 - d2 + d3 + d4, -d1 + d2 - d3 + d4));
            return true;
        }
        t += fmaxf(d * 0.68f, 0.00032f);
        if (t >= t_end) break;
    }
    return false;
}

// Ray-ellipsoid intersection against free-flight spilling pearls & crown-splash droplets above pond
__device__ __forceinline__ bool intersect_crown_splash_beads(
    const Vec3& ro,
    const Vec3& rd,
    float t_max,
    const DropletParticle* __restrict__ particles,
    const int* __restrict__ world_grid_head,
    const int* __restrict__ particle_next,
    PearlHit& out_hit
) {
    // Only test ray segment near pond surface y in [-0.02, 0.45]
    float t0 = 0.25f, t1 = fminf(t_max, 5.5f);
    if (fabsf(rd.y) > 1e-4f) {
        float ta = (0.45f - ro.y) / rd.y;
        float tb = (-0.03f - ro.y) / rd.y;
        t0 = fmaxf(0.25f, fminf(ta, tb));
        t1 = fminf(t1, fmaxf(ta, tb));
    }
    if (t0 >= t1) return false;

    bool found = false;
    out_hit.t = t1;
    out_hit.leaf_id = -1;

    // Sample 4 coarse points along the low-altitude ray segment in the 128x128 world XZ grid
    int prev_cell = -1;
    for (int s = 0; s < 5; ++s) {
        float ts = lerpf(t0, t1, float(s) * 0.25f);
        Vec3 ps = ro + rd * ts;
        int cx = int(floorf((ps.x - WAVE_X_MIN) / WAVE_SIZE * float(WORLD_GRID_RES)));
        int cz = int(floorf((ps.z - WAVE_Z_MIN) / WAVE_SIZE * float(WORLD_GRID_RES)));
        if (cx < 1 || cx >= WORLD_GRID_RES - 1 || cz < 1 || cz >= WORLD_GRID_RES - 1) continue;
        int base_cell = cz * WORLD_GRID_RES + cx;
        if (base_cell == prev_cell) continue;
        prev_cell = base_cell;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) {
                int idx = world_grid_head[(cz + dz) * WORLD_GRID_RES + (cx + dx)];
                int iter = 0;
                while (idx != -1 && iter < 12) {
                    ++iter;
                    const DropletParticle& q = particles[idx];
                    float r_eff = q.radius * 1.15f;
                    Vec3 oc = ro - q.pos;
                    float b = dot(oc, rd);
                    float c = dot(oc, oc) - r_eff * r_eff;
                    float disc = b * b - c;
                    if (disc > 0.0f) {
                        float th = -b - sqrtf(disc);
                        if (th > 0.2f && th < out_hit.t) {
                            out_hit.t = th;
                            out_hit.normal = normalize((ro + rd * th) - q.pos);
                            out_hit.radius = r_eff;
                            found = true;
                        }
                    }
                    idx = particle_next[idx];
                }
            }
        }
    }
    return found;
}

// ============================================================================
// Trace Environment Reflection Ray (for Pond Water & Refractive Water Pearls)
// Reflects all 7 Lotus Leaves, both Pink Lotus Buds, Golden Sun & Misty Horizon!
// ============================================================================
__device__ __forceinline__ Vec3 trace_reflection_or_refraction_env(
    const Vec3& ro,
    const Vec3& rd,
    const LotusLeafState* __restrict__ leaves,
    const LotusBudState* __restrict__ buds,
    const Vec3& sun_dir,
    bool sharp_sun
) {
    float best_t = 1e5f;
    Vec3 col(0.0f);
    bool hit_obj = false;

    LeafHit lhit;
    if (intersect_all_lotus_leaves(ro, rd, leaves, 0.004f, best_t, lhit)) {
        best_t = lhit.t;
        col = shade_lotus_leaf(lhit, ro, rd, leaves, sun_dir, nullptr, nullptr, nullptr, false);
        hit_obj = true;
    }

    BudHit bhit;
    if (intersect_lotus_buds(ro, rd, buds, 0.004f, best_t, bhit)) {
        best_t = bhit.t;
        col = shade_lotus_bud(bhit, ro, rd, sun_dir);
        hit_obj = true;
    }

    if (!hit_obj) {
        float vdepth = 24.0f;
        col = eval_golden_sky_and_foliage(rd, sun_dir, sharp_sun, vdepth);
    }
    return col;
}

// ============================================================================
// Kernel R1: Multi-Sampled Primary Ray-Tracing Pass (1920x1080)
// Evaluates 7 Sculpted Lotus Leaves, 2 Pink Lotus Buds, Refractive Water Pearls
// (IOR = 1.333 + Leaf Caustics), and 1024x1024 Dispersive Emerald-Ink Pond!
// ============================================================================
__global__ void render_primary_raytrace_kernel(
    float4* __restrict__ hdr_scene,
    CameraParams cam,
    const LotusLeafState* __restrict__ leaves,
    const LotusBudState* __restrict__ buds,
    const DropletParticle* __restrict__ particles,
    const int* __restrict__ leaf_grid_head,
    const int* __restrict__ world_grid_head,
    const int* __restrict__ particle_next,
    const float* __restrict__ wave_eta,
    const float2* __restrict__ wave_grad,
    float sim_time,
    int spp
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMG_W || py >= IMG_H) return;

    // 4x Rotated-Grid Stratified Supersampling offsets for razor-clean edges
    const float2 rgss[4] = {
        make_float2(0.125f, 0.375f),
        make_float2(0.625f, 0.125f),
        make_float2(0.375f, 0.875f),
        make_float2(0.875f, 0.625f)
    };

    Vec3 accum_col(0.0f);
    float primary_depth = 20.0f;

    int num_samples = max(1, min(spp, 4));
    for (int s = 0; s < num_samples; ++s) {
        float2 jitter = (num_samples == 1) ? make_float2(0.5f, 0.5f) : rgss[s];
        float u_ndc = ((float(px) + jitter.x) / float(IMG_W) * 2.0f - 1.0f) * cam.aspect * cam.tan_half_fov_y;
        float v_ndc = (1.0f - (float(py) + jitter.y) / float(IMG_H) * 2.0f) * cam.tan_half_fov_y;

        Vec3 ro = cam.pos;
        Vec3 rd = normalize(cam.forward + cam.right * u_ndc + cam.up * v_ndc);

        float best_t = 1e5f;
        int hit_kind = 0; // 0 = sky, 1 = leaf, 2 = bud, 3 = pond, 4 = pearl

        LeafHit lhit;
        if (intersect_all_lotus_leaves(ro, rd, leaves, 0.15f, best_t, lhit)) {
            best_t = lhit.t;
            hit_kind = 1;
        }

        BudHit bhit;
        if (intersect_lotus_buds(ro, rd, buds, 0.15f, best_t, bhit)) {
            best_t = bhit.t;
            hit_kind = 2;
        }

        // Intersect 1024x1024 dispersive capillary-gravity pond surface y = h(x,z)
        float t_pond = 1e5f;
        Vec3 pond_n(0.f, 1.f, 0.f);
        float pond_h = 0.0f;
        if (rd.y < -0.0015f) {
            float t_plane = (0.0f - ro.y) / rd.y;
            if (t_plane > 0.2f && t_plane < best_t && t_plane < 26.0f) {
                // 3-step secant heightfield refinement for exact ripple crest intersection
                float tp = t_plane;
                for (int k = 0; k < 3; ++k) {
                    Vec3 pp = ro + rd * tp;
                    sample_pond_surface(wave_eta, wave_grad, pp.x, pp.z, sim_time, pond_h, pond_n);
                    tp += (pond_h - pp.y) / fminf(rd.y, -0.02f);
                }
                if (tp > 0.2f && tp < best_t) {
                    Vec3 pp = ro + rd * tp;
                    sample_pond_surface(wave_eta, wave_grad, pp.x, pp.z, sim_time, pond_h, pond_n);
                    t_pond = tp;
                    best_t = tp;
                    hit_kind = 3;
                }
            }
        }

        // Check refractive water pearls on hit lotus leaf!
        PearlHit phit;
        if (hit_kind == 1 && !lhit.is_stem) {
            if (intersect_leaf_water_pearls(
                ro, rd, lhit.leaf_id, lhit.t, leaves[lhit.leaf_id],
                particles, leaf_grid_head, particle_next, phit
            )) {
                best_t = phit.t;
                hit_kind = 4;
            }
        }

        // Also check free-flight spilling leaf pearls & crown-splash droplets above pond!
        PearlHit splash_hit;
        if (intersect_crown_splash_beads(ro, rd, best_t, particles, world_grid_head, particle_next, splash_hit)) {
            phit = splash_hit;
            best_t = splash_hit.t;
            hit_kind = 4;
        }

        Vec3 sample_col(0.0f);
        float sample_z = best_t * dot(rd, cam.forward);

        if (hit_kind == 0) {
            float sky_z = 22.0f;
            sample_col = eval_golden_sky_and_foliage(rd, cam.sun_dir, true, sky_z);
            sample_z = sky_z;
        }
        else if (hit_kind == 1) {
            sample_col = shade_lotus_leaf(
                lhit, ro, rd, leaves, cam.sun_dir,
                particles, leaf_grid_head, particle_next, true
            );
        }
        else if (hit_kind == 2) {
            sample_col = shade_lotus_bud(bhit, ro, rd, cam.sun_dir);
        }
        else if (hit_kind == 4) {
            // ================================================================
            // Exact Dielectric Refractive Ray-Traced Water Pearl (IOR = 1.333)
            // Fresnel Reflection + Inverted Sky / Magnified Leaf Vein Refraction
            // + Internal Caustic Gleam!
            // ================================================================
            Vec3 hp = ro + rd * phit.t;
            Vec3 N = phit.normal;
            if (dot(N, -rd) < 0.0f) N = -N;

            float F = fresnel_dielectric(dot(-rd, N), 1.333f);
            Vec3 refl_dir = reflect_vec(rd, N);
            Vec3 refl_col = trace_reflection_or_refraction_env(hp + N * 0.0004f, refl_dir, leaves, buds, cam.sun_dir, true);

            Vec3 refr_dir;
            if (!refract_vec(rd, N, 1.0f / 1.333f, refr_dir)) {
                refr_dir = refl_dir;
            }
            // Second refraction exiting back/bottom of spherical droplet (thick water lens inversion!)
            Vec3 chord_exit = hp + refr_dir * (phit.radius * 1.75f);
            Vec3 exit_n = normalize(hp + N * (-phit.radius * 0.85f) - chord_exit);
            Vec3 final_refr_dir;
            if (!refract_vec(refr_dir, exit_n, 1.333f, final_refr_dir)) {
                final_refr_dir = refr_dir;
            }

            Vec3 refr_col(0.0f);
            LeafHit under_leaf;
            if (phit.leaf_id >= 0 && intersect_all_lotus_leaves(hp, final_refr_dir, leaves, 0.0001f, 0.08f, under_leaf)) {
                refr_col = shade_lotus_leaf(
                    under_leaf, hp, final_refr_dir, leaves, cam.sun_dir,
                    particles, leaf_grid_head, particle_next, true
                ) * 1.18f;
            } else {
                refr_col = trace_reflection_or_refraction_env(chord_exit, final_refr_dir, leaves, buds, cam.sun_dir, false);
            }

            // Pure silver-clear water pearl: diamond sun highlight + dark superhydrophobic rim meniscus!
            float ndotv_p = clampf(dot(-rd, N), 0.0f, 1.0f);
            float rim_gleam = powf(1.0f - ndotv_p, 3.2f);
            Vec3 Hp = normalize(cam.sun_dir - rd);
            float sun_pin = powf(fmaxf(0.0f, dot(N, Hp)), 280.0f);
            float forward_mie = powf(clampf(dot(rd, cam.sun_dir), 0.0f, 1.0f), 12.0f);
            Vec3 internal_glint = Vec3(14.0f, 10.5f, 6.2f) * sun_pin
                                + Vec3(1.8f, 1.45f, 0.85f) * forward_mie * 0.75f
                                + Vec3(0.75f, 0.88f, 0.92f) * rim_gleam * 0.52f;
            // Subtle dark Fresnel contact ring gives crisp jewel-like silhouette against emerald blade
            float meniscus_contrast = 0.62f + 0.38f * smoothstepf(0.08f, 0.45f, ndotv_p);

            sample_col = lerp3(refr_col * Vec3(0.92f, 0.98f, 0.95f) * meniscus_contrast, refl_col, clampf(F + 0.10f, 0.0f, 1.0f)) + internal_glint;
        }
        else if (hit_kind == 3) {
            // ================================================================
            // Dark, Tranquil Emerald-Ink Pond Surface:
            // Ray-Traced Lotus & Golden Sky Reflections + Volumetric Jade Depth
            // ================================================================
            Vec3 hp = ro + rd * t_pond;
            Vec3 N = pond_n;
            if (dot(N, -rd) < 0.0f) N = Vec3(0.0f, 1.0f, 0.0f);

            float F = fresnel_dielectric(dot(-rd, N), 1.333f);
            // Attenuate grazing sky-wash slightly so pond reads as deep emerald-ink mirror!
            float refl_weight = clampf(F * 0.74f, 0.04f, 0.78f);
            Vec3 refl_dir = reflect_vec(rd, N);
            if (refl_dir.y < 0.002f) refl_dir = normalize(Vec3(refl_dir.x, 0.005f, refl_dir.z));

            Vec3 refl_col = trace_reflection_or_refraction_env(hp + Vec3(0.0f, 0.003f, 0.0f), refl_dir, leaves, buds, cam.sun_dir, true);

            float sun_shadow = 1.0f;
            LeafHit pond_sh;
            if (intersect_all_lotus_leaves(hp + Vec3(0.f, 0.003f, 0.f), cam.sun_dir, leaves, 0.005f, 3.0f, pond_sh)) {
                sun_shadow = 0.22f;
            }
            float wave_caustic = clampf(1.0f + 22.0f * pond_h, 0.55f, 1.85f);
            Vec3 deep_ink   (0.004f, 0.022f, 0.019f);
            Vec3 sunlit_jade(0.012f, 0.058f, 0.042f);
            Vec3 water_body = lerp3(deep_ink, sunlit_jade, sun_shadow * 0.60f) * wave_caustic;

            // Golden sun sparkle lobe on capillary-gravity ripple crests (creates lush bokeh orbs in far pond!)
            Vec3 H = normalize(cam.sun_dir - rd);
            float spec_tight = powf(fmaxf(0.0f, dot(N, H)), 190.0f) * sun_shadow;
            float spec_warm  = powf(fmaxf(0.0f, dot(N, H)), 28.0f) * sun_shadow;
            Vec3 sun_glint = Vec3(24.0f, 16.0f, 6.4f) * spec_tight + Vec3(0.75f, 0.48f, 0.16f) * spec_warm * 0.42f;

            sample_col = lerp3(water_body, refl_col, refl_weight) + sun_glint;

            // Seamless horizon blend where distant pond meets misty emerald foliage bank!
            float horizon_fade = smoothstepf(14.0f, 25.5f, t_pond);
            if (horizon_fade > 0.001f) {
                float vz = 22.0f;
                Vec3 h_sky = eval_golden_sky_and_foliage(Vec3(rd.x, fmaxf(rd.y, 0.005f), rd.z), cam.sun_dir, false, vz);
                sample_col = lerp3(sample_col, h_sky, horizon_fade);
            }
        }

        // Subtle atmospheric perspective ONLY in the far background (> 5.2m), preserving rich dark foreground/midground contrast!
        float mist_f = (1.0f - expf(-0.045f * fmaxf(0.0f, sample_z - 5.2f))) * 0.42f;
        float sun_align = powf(clampf(dot(rd, cam.sun_dir) * 0.5f + 0.5f, 0.0f, 1.0f), 3.5f);
        Vec3 mist_col = lerp3(Vec3(0.03f, 0.09f, 0.07f), Vec3(0.62f, 0.42f, 0.18f), sun_align);
        sample_col = lerp3(sample_col, mist_col, clampf(mist_f, 0.0f, 0.42f));

        accum_col += sample_col;
        if (s == 0 || sample_z < primary_depth) primary_depth = sample_z;
    }

    accum_col *= (1.0f / float(num_samples));
    hdr_scene[py * IMG_W + px] = make_float4(accum_col.x, accum_col.y, accum_col.z, primary_depth);
}

// ============================================================================
// Helper: Atomic Add into HDR Bokeh / Rain Layer
// ============================================================================
__device__ __forceinline__ void atomic_add_bokeh(float4* __restrict__ buf, int px, int py, const Vec3& col, float w = 0.0f) {
    if (px < 0 || px >= IMG_W || py < 0 || py >= IMG_H) return;
    int idx = py * IMG_W + px;
    atomicAdd(&buf[idx].x, col.x);
    atomicAdd(&buf[idx].y, col.y);
    atomicAdd(&buf[idx].z, col.z);
    if (w != 0.0f) atomicAdd(&buf[idx].w, w);
}

// ============================================================================
// Kernel R2: Motion-Blurred Refractive Falling Rain Streaks (Slanted by W(x,z,t))
//            Convolved with True 9-Blade f/1.4 Optical Circle-of-Confusion Bokeh!
// ============================================================================
__global__ void splat_rain_streaks_kernel(
    float4* __restrict__ bokeh_accum,
    const float4* __restrict__ hdr_scene,
    const DropletParticle* __restrict__ particles,
    CameraParams cam
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    const DropletParticle& p = particles[idx];
    if (p.state == STATE_LEAF_BEAD) return;
    if (p.active_band == 0 && (idx & 7) != 0) return; // Sparse faint far-mist streaks in background

    // 180-degree cinema shutter exposure time = 1 / 120 s = 0.00833 s
    const float dt_shutter = (p.state == STATE_FALLING_RAIN) ? 0.0082f : 0.0045f;
    Vec3 p0 = p.pos - p.vel * (0.5f * dt_shutter);
    Vec3 p1 = p.pos + p.vel * (0.5f * dt_shutter);

    // Project endpoints to camera space
    Vec3 d0 = p0 - cam.pos;
    Vec3 d1 = p1 - cam.pos;
    float z0 = dot(d0, cam.forward);
    float z1 = dot(d1, cam.forward);
    if (z0 < 0.42f || z1 < 0.42f || z0 > 15.0f) return;

    float inv_x_scale = 1.0f / (cam.aspect * cam.tan_half_fov_y);
    float inv_y_scale = 1.0f / cam.tan_half_fov_y;

    float sx0 = (dot(d0, cam.right) / z0 * inv_x_scale * 0.5f + 0.5f) * float(IMG_W);
    float sy0 = (0.5f - dot(d0, cam.up) / z0 * inv_y_scale * 0.5f) * float(IMG_H);
    float sx1 = (dot(d1, cam.right) / z1 * inv_x_scale * 0.5f + 0.5f) * float(IMG_W);
    float sy1 = (0.5f - dot(d1, cam.up) / z1 * inv_y_scale * 0.5f) * float(IMG_H);

    if ((sx0 < -40.f && sx1 < -40.f) || (sx0 > IMG_W + 40.f && sx1 > IMG_W + 40.f) ||
        (sy0 < -40.f && sy1 < -40.f) || (sy0 > IMG_H + 40.f && sy1 > IMG_H + 40.f)) {
        return;
    }

    float z_mid = 0.5f * (z0 + z1);
    float coc_px = fabsf(compute_signed_coc_px(z_mid, cam));
    float geom_rad_px = fmaxf(0.55f, (p.radius / z_mid) * inv_y_scale * 0.5f * float(IMG_H));
    float blur_rad = sqrtf(geom_rad_px * geom_rad_px + coc_px * coc_px);
    blur_rad = clampf(blur_rad, 0.70f, 22.0f);

    // Physical optical energy conservation: larger bokeh radius spreads flux over pi * R^2!
    Vec3 view_dir = normalize(d0);
    float sun_rim = powf(clampf(dot(view_dir, cam.sun_dir) * 0.5f + 0.5f, 0.0f, 1.0f), 3.2f);
    // Warm golden-rimmed silver crystalline raindrop radiance
    Vec3 drop_tint = lerp3(Vec3(0.62f, 0.74f, 0.80f), Vec3(2.45f, 1.85f, 0.95f), sun_rim);

    float area_norm = (geom_rad_px * geom_rad_px + 0.65f) / (blur_rad * blur_rad + 0.65f);
    float base_opacity = clampf(0.52f * area_norm * (p.radius * 1000.0f), 0.0045f, 0.42f);

    float dx = sx1 - sx0;
    float dy = sy1 - sy0;
    float seg_len = sqrtf(dx * dx + dy * dy + 1e-6f);

    int steps = clampf(ceilf(seg_len / fmaxf(0.85f, blur_rad * 0.45f)), 1.0f, 42.0f);
    float step_weight = 1.0f / float(steps);
    int ir = int(ceilf(blur_rad));

    for (int s = 0; s < steps; ++s) {
        float t = (float(s) + 0.5f) / float(steps);
        float cx = lerpf(sx0, sx1, t);
        float cy = lerpf(sy0, sy1, t);

        int icx = int(floorf(cx));
        int icy = int(floorf(cy));

        // Sample underlying background depth for accurate occlusion behind foreground leaves!
        if (icx >= 0 && icx < IMG_W && icy >= 0 && icy < IMG_H) {
            float scene_z = hdr_scene[icy * IMG_W + icx].w;
            if (z_mid > scene_z + 0.04f && z_mid > cam.focus_z - 0.1f) continue;
        }

        if (blur_rad <= 1.35f) {
            // Crisp near-focus refractive rain needle (anti-aliased 2x2 bilinear splat)
            float fx = cx - floorf(cx);
            float fy = cy - floorf(cy);
            Vec3 c_add = drop_tint * (base_opacity * step_weight * float(steps) * 0.18f);
            atomic_add_bokeh(bokeh_accum, icx,     icy,     c_add * ((1.f - fx) * (1.f - fy)));
            atomic_add_bokeh(bokeh_accum, icx + 1, icy,     c_add * (fx * (1.f - fy)));
            atomic_add_bokeh(bokeh_accum, icx,     icy + 1, c_add * ((1.f - fx) * fy));
            atomic_add_bokeh(bokeh_accum, icx + 1, icy + 1, c_add * (fx * fy));
        } else {
            // Out-of-focus foreground/background 9-blade polygonal/circular bokeh ribbon!
            Vec3 c_disc = drop_tint * (base_opacity * step_weight * 2.4f);
            int stride = (ir > 10) ? 2 : 1; // Subsample ultra-wide bokeh smoothly, gathered in R4
            float stride_comp = float(stride * stride);
            for (int oy = -ir; oy <= ir; oy += stride) {
                int py = icy + oy;
                if (py < 0 || py >= IMG_H) continue;
                for (int ox = -ir; ox <= ir; ox += stride) {
                    int px = icx + ox;
                    if (px < 0 || px >= IMG_W) continue;

                    float rx = (float(px) + 0.5f - cx) / blur_rad;
                    float ry = (float(py) + 0.5f - cy) / blur_rad;
                    float r_norm = sqrtf(rx * rx + ry * ry + 1e-8f);
                    float ang = atan2f(ry, rx);
                    float r_lim = aperture_9blade_radius(ang);
                    if (r_norm <= r_lim) {
                        // Authentic optical spherical-aberration soft outer ring!
                        float u_r = r_norm / r_lim;
                        float ring_prof = (0.72f + 0.52f * u_r * u_r) * smoothstepf(1.0f, 0.84f, u_r);
                        atomic_add_bokeh(bokeh_accum, px, py, c_disc * (ring_prof * stride_comp));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Kernel R3: Specular Pond-Sparkle & Foliage Glint Bokeh Scatter (2x2 Tile)
// Extracts bright out-of-focus HDR sun sparkles and splats crisp, noise-free
// 9-blade rounded circular aperture bokeh discs!
// ============================================================================
__global__ void scatter_specular_bokeh_discs_kernel(
    float4* __restrict__ hdr_scene,
    float4* __restrict__ bokeh_accum,
    CameraParams cam
) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = bx * 2;
    int y0 = by * 2;
    if (x0 >= IMG_W - 1 || y0 >= IMG_H - 1) return;

    // Pool 2x2 neighborhood for bright out-of-focus specular highlight
    Vec3 sum_hi(0.0f);
    float max_coc = 0.0f;
    float hi_z = 20.0f;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int idx = (y0 + dy) * IMG_W + (x0 + dx);
            float4 sc = hdr_scene[idx];
            float coc = fabsf(compute_signed_coc_px(sc.w, cam));
            float lum = 0.2126f * sc.x + 0.7152f * sc.y + 0.0722f * sc.z;

            if (coc > 3.2f && lum > 1.25f) {
                float extract_frac = smoothstepf(1.25f, 3.2f, lum) * smoothstepf(3.2f, 6.5f, coc) * 0.85f;
                Vec3 cur(sc.x, sc.y, sc.z);
                sum_hi += cur * extract_frac;
                // Attenuate point spike in base scene so energy is strictly conserved!
                Vec3 rem = cur * (1.0f - extract_frac);
                hdr_scene[idx] = make_float4(rem.x, rem.y, rem.z, sc.w);
                if (coc > max_coc) {
                    max_coc = coc;
                    hi_z = sc.w;
                }
            }
        }
    }

    float hi_lum = 0.2126f * sum_hi.x + 0.7152f * sum_hi.y + 0.0722f * sum_hi.z;
    if (hi_lum < 0.05f || max_coc <= 3.2f) return;

    float R = clampf(max_coc, 3.5f, cam.max_coc_px);
    int ir = int(ceilf(R));
    float cx = float(x0) + 1.0f;
    float cy = float(y0) + 1.0f;

    // Subtle optical cat's-eye mechanical vignetting toward extreme 85mm f/1.4 frame corners
    float vx_ndc = (cx / float(IMG_W) - 0.5f) * 2.0f;
    float vy_ndc = (cy / float(IMG_H) - 0.5f) * 2.0f;

    float norm_factor = 1.0f / (3.14159265f * R * R);
    Vec3 disc_energy = sum_hi * norm_factor * 1.35f;

    for (int oy = -ir; oy <= ir; ++oy) {
        int py = int(cy) + oy;
        if (py < 0 || py >= IMG_H) continue;
        for (int ox = -ir; ox <= ir; ++ox) {
            int px = int(cx) + ox;
            if (px < 0 || px >= IMG_W) continue;

            // Depth occlusion: background pond bokeh discs do not bleed over sharp Hero Leaf!
            float dst_z = hdr_scene[py * IMG_W + px].w;
            float dst_coc = fabsf(compute_signed_coc_px(dst_z, cam));
            if (hi_z > dst_z + 0.35f && dst_coc < 2.5f) continue;

            float rx = (float(px) + 0.5f - cx) / R;
            float ry = (float(py) + 0.5f - cy) / R;

            // Slight optical cat's-eye shift near lens periphery
            float rx2 = rx - vx_ndc * 0.24f;
            float ry2 = ry - vy_ndc * 0.24f;
            if (rx2 * rx2 + ry2 * ry2 > 1.28f) continue;

            float r_norm = sqrtf(rx * rx + ry * ry + 1e-8f);
            float r_lim = aperture_9blade_radius(atan2f(ry, rx));
            if (r_norm <= r_lim) {
                float u = r_norm / r_lim;
                // Luminous creamy disc with delicate f/1.4 optical edge ring
                float w = (0.76f + 0.58f * powf(u, 3.0f)) * smoothstepf(1.0f, 0.86f, u);
                atomic_add_bokeh(bokeh_accum, px, py, disc_energy * w);
            }
        }
    }
}

// ============================================================================
// Kernel R4: True Optical Circle-of-Confusion Gather (64-Tap 9-Blade Vogel Disk)
//            + Volumetric 18-deg Golden Sunbeam Shafts + AgX Botanical Tonemap!
// ============================================================================
__global__ void dof_gather_and_tonemap_kernel(
    uchar3* __restrict__ srgb_out,
    const float4* __restrict__ hdr_scene,
    const float4* __restrict__ bokeh_accum,
    CameraParams cam,
    float sim_time
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMG_W || py >= IMG_H) return;

    int center_idx = py * IMG_W + px;
    float4 center_sc = hdr_scene[center_idx];
    float center_z = center_sc.w;
    float center_signed_coc = compute_signed_coc_px(center_z, cam);
    float center_abs_coc = fabsf(center_signed_coc);

    // Isotropic circular ring probe for foreground blur bleeding onto sharp focus plane (NO 1D cross!)
    float max_fg_coc = (center_signed_coc < 0.0f) ? -center_signed_coc : 0.0f;
    for (int i = 0; i < 12; ++i) {
        float ang = float(i) * (6.2831853f / 12.0f);
        float r_probe = (i & 1) ? 14.0f : 7.0f;
        int sx = int(clampf(float(px) + cosf(ang) * r_probe, 0.f, float(IMG_W - 1)));
        int sy = int(clampf(float(py) + sinf(ang) * r_probe, 0.f, float(IMG_H - 1)));
        float c_fg = compute_signed_coc_px(hdr_scene[sy * IMG_W + sx].w, cam);
        if (c_fg < 0.0f) max_fg_coc = fmaxf(max_fg_coc, -c_fg);
    }

    float gather_rad = fmaxf(center_abs_coc, max_fg_coc);
    Vec3 gathered_col(center_sc.x, center_sc.y, center_sc.z);

    if (gather_rad > 0.65f) {
        Vec3 sum_col = gathered_col;
        float sum_w = 1.0f;

        // 64-tap Golden-Angle Vogel Spiral warped to our 9-Blade Rounded Macro Aperture!
        constexpr int TAPS = 64;
        const float golden_ang = 2.39996323f;
        float pixel_rot = hash21(float(px), float(py)) * 6.2831853f;

        for (int i = 0; i < TAPS; ++i) {
            float r_unit = sqrtf((float(i) + 0.5f) / float(TAPS));
            float theta = float(i) * golden_ang + pixel_rot;
            float blade_r = r_unit * aperture_9blade_radius(theta) * gather_rad;

            int sx = int(floorf(float(px) + 0.5f + cosf(theta) * blade_r));
            int sy = int(floorf(float(py) + 0.5f + sinf(theta) * blade_r));
            if (sx < 0 || sx >= IMG_W || sy < 0 || sy >= IMG_H) continue;

            float4 tap_sc = hdr_scene[sy * IMG_W + sx];
            float tap_signed_coc = compute_signed_coc_px(tap_sc.w, cam);
            float tap_abs_coc = fabsf(tap_signed_coc);

            // Physically accurate foreground/background CoC gather weights:
            // 1. Background taps only contribute if their own CoC covers blade_r AND they aren't closer sharp objects!
            // 2. Foreground taps (tap_signed_coc < 0) bleed freely across sharper background/hero pixels!
            float w = 0.0f;
            if (tap_signed_coc < -0.5f) {
                w = smoothstepf(blade_r - 1.5f, blade_r + 1.5f, tap_abs_coc);
            } else {
                float depth_gate = smoothstepf(center_z - 0.25f, center_z + 0.04f, tap_sc.w);
                float coc_gate   = smoothstepf(blade_r - 1.5f, blade_r + 1.5f, tap_abs_coc);
                w = depth_gate * coc_gate;
            }

            sum_col += Vec3(tap_sc.x, tap_sc.y, tap_sc.z) * w;
            sum_w += w;
        }
        gathered_col = sum_col * (1.0f / sum_w);
    }

    // Smoothly composite scattered 9-blade bokeh highlights & shutter-integrated rain streaks
    // (3x3 tent filter removes any sub-pixel splat discretization!)
    Vec3 bokeh_layer(0.0f);
    const float tent[3] = { 0.25f, 0.50f, 0.25f };
    for (int dy = -1; dy <= 1; ++dy) {
        int sy = min(max(py + dy, 0), IMG_H - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = min(max(px + dx, 0), IMG_W - 1);
            float4 b = bokeh_accum[sy * IMG_W + sx];
            bokeh_layer += Vec3(b.x, b.y, b.z) * (tent[dx + 1] * tent[dy + 1]);
        }
    }

    Vec3 hdr = gathered_col + bokeh_layer;

    // ========================================================================
    // Warm Late-Afternoon Golden-Hour Volumetric Sunbeams (sun_elevation = 18 deg)
    // Subtle crepuscular Tyndall shafts slanting across misty pond air!
    // ========================================================================
    float u_uv = float(px) / float(IMG_W);
    float v_uv = float(py) / float(IMG_H);
    float shaft_coord = (u_uv * 0.82f + v_uv * 0.57f) * 9.5f - sim_time * 0.08f;
    float shaft_pattern = 0.55f + 0.45f * sinf(shaft_coord) * cosf(shaft_coord * 1.7f + 0.9f);
    float depth_vol = clampf((fminf(center_z, 8.0f) - 1.1f) / 7.5f, 0.03f, 1.0f);
    float corner_glow = expf(-((u_uv - 0.22f) * (u_uv - 0.22f) * 2.2f + (v_uv - 0.06f) * (v_uv - 0.06f) * 2.8f));
    Vec3 sunbeam = Vec3(0.24f, 0.15f, 0.052f) * (0.08f + 0.26f * corner_glow) * shaft_pattern * depth_vol;
    hdr += sunbeam;

    // Subtle natural 85mm optical lens vignette
    float vx = u_uv - 0.5f;
    float vy = v_uv - 0.5f;
    float vig = 1.0f - 0.26f * (vx * vx + vy * vy) * 2.1f;
    hdr *= vig;

    // ========================================================================
    // AgX-Inspired Botanical Cinema Tone Mapping (Lush Emerald & Creamy Golds)
    // ========================================================================
    hdr *= 0.92f; // Balanced moody BBC-Earth macro cinema exposure!
    Vec3 inset(
        0.86f * hdr.x + 0.09f * hdr.y + 0.05f * hdr.z,
        0.06f * hdr.x + 0.88f * hdr.y + 0.06f * hdr.z,
        0.05f * hdr.x + 0.09f * hdr.y + 0.86f * hdr.z
    );

    auto filmic_curve = [](float x) -> float {
        x = fmaxf(0.0f, x);
        // ACES / Kodak-2383 S-curve with rich velvet shadow toe & creamy highlight shoulder
        return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    };

    Vec3 mapped(filmic_curve(inset.x), filmic_curve(inset.y), filmic_curve(inset.z));

    // Gentle saturation restore after filmic highlight desaturation
    float lum = 0.2126f * mapped.x + 0.7152f * mapped.y + 0.0722f * mapped.z;
    mapped = clamp3(lerp3(Vec3(lum), mapped, 1.14f), 0.0f, 1.0f);

    // Exact sRGB transfer function + Interleaved Gradient Noise (IGN) 1-LSB dithering
    auto linear_to_srgb = [](float c) -> float {
        c = clampf(c, 0.0f, 1.0f);
        return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
    };

    float ign = fmodf(52.9829189f * fmodf(0.06711056f * float(px) + 0.00583715f * float(py), 1.0f), 1.0f) - 0.5f;
    float dither = ign * (1.0f / 255.0f);

    uint8_t r8 = uint8_t(clampf(linear_to_srgb(mapped.x) + dither, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint8_t g8 = uint8_t(clampf(linear_to_srgb(mapped.y) + dither, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint8_t b8 = uint8_t(clampf(linear_to_srgb(mapped.z) + dither, 0.0f, 1.0f) * 255.0f + 0.5f);

    srgb_out[center_idx] = make_uchar3(r8, g8, b8);
}
