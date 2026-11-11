#pragma once

#include "common.cuh"
#include <cstdio>

// ============================================================================
// GPU Buffers & State Container for Coupled 65,536-Particle + 1024^2 Wave Sim
// ============================================================================
struct GpuSimContext {
    DropletParticle* d_particles = nullptr;
    LotusLeafState*  d_leaves    = nullptr;
    LotusBudState*   d_buds      = nullptr;

    // Leaf torque & vertical force accumulators (3 floats per leaf: fx_torque, fz_torque, fy_force)
    float* d_leaf_forces = nullptr;

    // 1024x1024 Dispersive Capillary-Gravity Shallow-Water Heightfield
    float*  d_wave_eta  = nullptr; // Surface elevation h(x,z) in meters
    float*  d_wave_vel  = nullptr; // Vertical velocity dh/dt
    float*  d_wave_lap  = nullptr; // Laplacian(eta) for biharmonic capillary operator
    float2* d_wave_grad = nullptr; // Smooth surface slope (dh/dx, dh/dz)

    // Spatial Acceleration Grids
    // 1. Per-leaf 64x64 grid for O(1) coalescence, ray-traced pearls, and leaf caustics
    int* d_leaf_grid_head = nullptr; // [NUM_LEAVES * LEAF_GRID_RES * LEAF_GRID_RES]
    // 2. Pond-surface 128x128 world XZ grid for ray-traced crown-splash & near-water pearls
    int* d_world_grid_head = nullptr; // [WORLD_GRID_RES * WORLD_GRID_RES]
    int* d_particle_next   = nullptr; // [NUM_PARTICLES] linked-list next pointer
    int* d_merge_claim     = nullptr; // [NUM_PARTICLES] atomic lock for volume-conserving merge

    // Acoustic impact events buffer
    GpuImpactEvent* d_events = nullptr;
    int* d_event_count       = nullptr;
};

// ============================================================================
// Helper: Sample 1024x1024 Wave Height Bilinearly on Device
// ============================================================================
__device__ __forceinline__ float sample_wave_height(const float* __restrict__ eta, float wx, float wz) {
    float gx = (wx - WAVE_X_MIN) * (1.0f / WAVE_DX) - 0.5f;
    float gz = (wz - WAVE_Z_MIN) * (1.0f / WAVE_DX) - 0.5f;
    if (gx < 0.0f || gx >= float(WAVE_N - 2) || gz < 0.0f || gz >= float(WAVE_N - 2)) {
        return 0.0f;
    }
    int ix = int(floorf(gx));
    int iz = int(floorf(gz));
    float fx = gx - float(ix);
    float fz = gz - float(iz);
    int row0 = iz * WAVE_N + ix;
    int row1 = (iz + 1) * WAVE_N + ix;
    float h00 = eta[row0];
    float h10 = eta[row0 + 1];
    float h01 = eta[row1];
    float h11 = eta[row1 + 1];
    return lerpf(lerpf(h00, h10, fx), lerpf(h01, h11, fx), fz);
}

// ============================================================================
// Deposit Localized Pressure Impulse into 1024x1024 Pond Wave Grid
// ============================================================================
__device__ __forceinline__ void inject_pond_impulse(
    float* __restrict__ wave_vel,
    float wx, float wz,
    float drop_radius_m,
    float impact_speed
) {
    float gx = (wx - WAVE_X_MIN) * (1.0f / WAVE_DX);
    float gz = (wz - WAVE_Z_MIN) * (1.0f / WAVE_DX);
    int cx = int(floorf(gx));
    int cz = int(floorf(gz));
    if (cx < 6 || cx >= WAVE_N - 6 || cz < 6 || cz >= WAVE_N - 6) return;

    // Physical cavity impulse scaled by droplet momentum ~ r^1.8 * v
    float r_mm = drop_radius_m * 1000.0f;
    float strength = -0.34f * powf(clampf(r_mm, 0.40f, 5.0f), 1.65f) * clampf(impact_speed, 0.6f, 9.5f) * 0.22f;
    // Heavy spilling pearls create broader, deeper craters
    int rad_cells = (r_mm > 2.0f) ? 5 : 4;
    float inv_sig2 = (r_mm > 2.0f) ? 0.14f : 0.24f;

    for (int dz = -rad_cells; dz <= rad_cells; ++dz) {
        for (int dx = -rad_cells; dx <= rad_cells; ++dx) {
            float d2 = float(dx * dx + dz * dz);
            // Zero-mean crater + rim impulse (conserves mean pond elevation!)
            float w = (1.0f - 0.50f * d2 * inv_sig2) * expf(-0.5f * d2 * inv_sig2);
            atomicAdd(&wave_vel[(cz + dz) * WAVE_N + (cx + dx)], strength * w);
        }
    }
}

// ============================================================================
// Respawn a Recycled Particle into the Falling Rain / Mist Column
// ============================================================================
__device__ __forceinline__ void respawn_raindrop(
    DropletParticle& p,
    float sim_time,
    bool initial_fill
) {
    float rain_int = eval_rain_intensity(sim_time);
    float u_band = rand01(p.rng);
    p.state = STATE_FALLING_RAIN;
    p.leaf_id = -1;
    p.age = 0.0f;
    p.squash = 0.0f;

    if (u_band < rain_int) {
        p.active_band = 1;
        // Concentrate macro density gently around visible camera frustum [-2.6, 2.6] x [0.55, 6.3]
        float u1 = rand01(p.rng);
        float u2 = rand01(p.rng);
        // Depth stratified so both foreground bokeh streaks, hero leaf, and sunlit pond get rich coverage
        float z = lerpf(0.62f, 6.25f, powf(u2, 0.85f));
        float x_span = 0.45f + 0.46f * z;
        float x = (u1 * 2.0f - 1.0f) * x_span;
        float y = initial_fill ? lerpf(0.25f, 5.8f, rand01(p.rng))
                               : lerpf(2.20f, 5.60f, rand01(p.rng));
        // Offset spawn X upwind during gust so slanted streaks sweep squarely across frame
        Vec3 w = eval_wind_field(x, z, sim_time);
        x -= w.x * (y * 0.14f);

        p.pos = Vec3(x, y, z);
        // Marshall-Palmer raindrop size distribution (0.45mm to 1.65mm radius)
        float ur = rand01(p.rng);
        p.radius = 0.00048f + 0.00118f * powf(ur, 2.1f);
        // Terminal fall speed ~ 4.8 - 7.8 m/s depending on drop radius
        float v_term = 4.5f + 2200.0f * p.radius;
        p.vel = Vec3(w.x * 0.88f, -v_term, w.z * 0.88f);
    } else {
        // Upper atmospheric / distant mist reservoir: still simulated every step on GPU!
        p.active_band = 0;
        float z = lerpf(6.8f, 16.5f, rand01(p.rng));
        float x = (rand01(p.rng) * 2.0f - 1.0f) * (1.2f + 0.42f * z);
        float y = initial_fill ? lerpf(0.8f, 8.5f, rand01(p.rng))
                               : lerpf(3.2f, 8.5f, rand01(p.rng));
        Vec3 w = eval_wind_field(x, z, sim_time);
        p.pos = Vec3(x, y, z);
        p.radius = 0.00042f + 0.00065f * rand01(p.rng);
        p.vel = Vec3(w.x, -(4.2f + 1500.0f * p.radius), w.z);
    }
}

// ============================================================================
// Kernel 0: Initialize 65,536 Unified Water Elements (Pre-seeded Leaf Dew + Rain)
// ============================================================================
__global__ void init_particles_kernel(
    DropletParticle* __restrict__ particles,
    const LotusLeafState* __restrict__ leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    DropletParticle p;
    p.rng = pcg_hash(uint32_t(idx * 1973u + 9277u));
    p.pad0 = 0;

    // Seed ~2,600 well-spaced superhydrophobic pearls across the 7 lotus leaves at t = 0
    if (idx < 2600) {
        int lid = idx % NUM_LEAVES;
        if (idx < 820) lid = 0; // Rich jewel-like dew & rolling pearls on Hero Leaf 0!

        // Golden-angle phyllotaxis spiral distribution so initial pearls don't overlap instantaneously
        float local_k = (lid == 0) ? float(idx) / 820.0f : float(idx / NUM_LEAVES) / 300.0f;
        local_k = clampf(local_k, 0.01f, 0.98f);
        float ang = float(idx) * 2.39996323f + 0.18f * (rand01(p.rng) - 0.5f);
        float rad_frac = 0.08f + 0.84f * sqrtf(local_k);

        if (idx < 8) {
            // Hero coalescing pearl group in Hero Leaf 0's central hollow -> giant pearl spills ~ t = 2.9s!
            lid = 0;
            ang = leaves[0].pour_angle + float(idx - 4) * 0.16f;
            rad_frac = 0.12f + 0.05f * float(idx);
            p.radius = 0.00285f + 0.00055f * rand01(p.rng);
        } else if (idx < 28) {
            // Prominent jewel-like silver pearls (2.6 - 4.4 mm radius) across Hero Leaf 0's sunlit bowl!
            lid = 0;
            // Place primarily on front/sunlit visible petals of Hero Leaf 0
            ang = -1.57f + (float(idx - 8) / 20.0f) * 3.14f + 0.15f * (rand01(p.rng) - 0.5f);
            rad_frac = 0.18f + 0.58f * rand01(p.rng);
            p.radius = 0.00265f + 0.00165f * rand01(p.rng);
        } else if (idx < 140) {
            // Medium glistening silver pearls (1.8 - 3.1 mm radius) across all 7 leaves
            lid = (idx < 75) ? 0 : (idx % NUM_LEAVES);
            p.radius = 0.00185f + 0.00125f * rand01(p.rng);
        } else {
            // Fine superhydrophobic dew micro-pearls (0.8 - 1.6 mm radius)
            p.radius = 0.00080f + 0.00085f * powf(rand01(p.rng), 1.6f);
        }

        const LotusLeafState& L_use = leaves[lid];
        float u = cosf(ang) * (rad_frac * L_use.radius * 0.90f);
        float v = sinf(ang) * (rad_frac * L_use.radius * 0.90f);
        LeafSample ls = eval_lotus_leaf_shape(L_use, u, v);
        Vec3 n_w = eval_lotus_leaf_world_normal(L_use, u, v);

        p.state = STATE_LEAF_BEAD;
        p.leaf_id = lid;
        p.leaf_u = u;
        p.leaf_v = v;
        p.vel = Vec3(0.0f);
        p.age = rand01(p.rng) * 2.0f;
        p.squash = 0.0f;
        p.active_band = 1;
        // High contact angle (~150 deg): pearl center sits ~0.84 * R above leaf cuticle
        p.pos = L_use.world_center + L_use.axis_u * u + L_use.axis_n * ls.height + L_use.axis_v * v + n_w * (p.radius * 0.84f);
    } else {
        respawn_raindrop(p, 0.0f, true);
    }

    particles[idx] = p;
}

// ============================================================================
// Kernel 1: Coupled Particle Advection, Lotus Leaf Collision, Superhydrophobic
//           Rolling, Pond Impacts, Crown Splashes & Acoustic Event Emission
// ============================================================================
__global__ void step_particles_kernel(
    DropletParticle* __restrict__ particles,
    const LotusLeafState* __restrict__ leaves,
    float* __restrict__ leaf_forces,
    const float* __restrict__ wave_eta,
    float* __restrict__ wave_vel,
    GpuImpactEvent* __restrict__ events,
    int* __restrict__ event_count,
    int frame_idx,
    float sub_frac,
    float sim_time,
    float dt,
    bool record_events
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    DropletParticle p = particles[idx];
    p.age += dt;
    p.squash *= expf(-18.0f * dt);

    const Vec3 gravity(0.0f, -9.81f, 0.0f);

    // ------------------------------------------------------------------------
    // STATE 0: FALLING RAIN STREAKS
    // ------------------------------------------------------------------------
    if (p.state == STATE_FALLING_RAIN) {
        Vec3 wind = eval_wind_field(p.pos.x, p.pos.z, sim_time);
        // Aerodynamic drag pulls droplet horizontal velocity smoothly toward breeze W(x,z,t)
        float drag_k = 1.65f / fmaxf(p.radius * 1000.0f, 0.45f);
        Vec3 v_target(wind.x, -(4.6f + 2150.0f * p.radius), wind.z);
        p.vel += (v_target - p.vel) * fminf(1.0f, drag_k * dt);

        Vec3 prev_pos = p.pos;
        p.pos += p.vel * dt;

        // Check collision against the 7 sculpted lotus leaves (only if in macro pond zone)
        bool hit_leaf = false;
        if (p.active_band == 1 && p.pos.y < 0.72f && p.pos.y > -0.05f) {
            for (int lid = 0; lid < NUM_LEAVES; ++lid) {
                const LotusLeafState& L = leaves[lid];
                Vec3 d_prev = prev_pos - L.world_center;
                Vec3 d_curr = p.pos - L.world_center;
                // Quick bounding sphere reject
                float r_bound = L.radius * 1.12f;
                if (length_sq(d_curr) > r_bound * r_bound) continue;

                float u_curr = dot(d_curr, L.axis_u);
                float v_curr = dot(d_curr, L.axis_v);
                LeafSample ls = eval_lotus_leaf_shape(L, u_curr, v_curr);
                if (ls.s_norm > 0.992f) continue;

                float y_prev = dot(d_prev, L.axis_n) - ls.height;
                float y_curr = dot(d_curr, L.axis_n) - ls.height;

                if (y_prev >= -0.008f && y_curr <= p.radius * 0.9f && y_curr > -0.06f) {
                    Vec3 n_leaf = eval_lotus_leaf_world_normal(L, u_curr, v_curr);
                    float vn = fmaxf(0.5f, -dot(p.vel, n_leaf));

                    // Apply physical impact impulse & torque to leaf
                    float drop_mass = 4.18879f * p.radius * p.radius * p.radius * 1000.0f; // kg
                    float imp = drop_mass * vn * 48.0f; // Impulse scaled for visible spring response
                    Vec3 rel = p.pos - L.world_center;
                    atomicAdd(&leaf_forces[lid * 3 + 0], -rel.z * imp);
                    atomicAdd(&leaf_forces[lid * 3 + 1],  rel.x * imp);
                    atomicAdd(&leaf_forces[lid * 3 + 2], -imp);

                    // Record physical acoustic modal-membrane strike event
                    if (record_events && event_count != nullptr) {
                        int e_idx = atomicAdd(event_count, 1);
                        if (e_idx < MAX_IMPACT_EVENTS) {
                            events[e_idx] = GpuImpactEvent{
                                frame_idx, sub_frac, p.pos.x, p.pos.z,
                                IMPACT_LOTUS_LEAF, lid,
                                p.radius * 1000.0f, vn, clampf(ls.s_norm, 0.0f, 1.0f)
                            };
                        }
                    }

                    // Superhydrophobic outcome (contact angle ~ 150 deg):
                    // 26% crisp elastic rebound into micro-crown bead, 74% adhere & roll as pearl!
                    if (vn > 3.2f && rand01(p.rng) < 0.26f) {
                        p.state = STATE_CROWN_SPLASH;
                        Vec3 refl = reflect_vec(p.vel, n_leaf);
                        float ang_j = rand01(p.rng) * 6.2831853f;
                        Vec3 tang = normalize(L.axis_u * cosf(ang_j) + L.axis_v * sinf(ang_j));
                        p.vel = refl * 0.24f + n_leaf * 0.55f + tang * (0.35f + 0.45f * rand01(p.rng));
                        p.pos = L.world_center + L.axis_u * u_curr + L.axis_n * ls.height + L.axis_v * v_curr + n_leaf * (p.radius * 1.2f);
                        p.radius *= 0.85f;
                        p.age = 0.0f;
                        p.squash = 0.28f;
                    } else {
                        p.state = STATE_LEAF_BEAD;
                        p.leaf_id = lid;
                        p.leaf_u = u_curr;
                        p.leaf_v = v_curr;
                        float vu = dot(p.vel, L.axis_u) * 0.22f;
                        float vv = dot(p.vel, L.axis_v) * 0.22f;
                        p.vel = Vec3(vu, 0.0f, vv);
                        p.age = 0.0f;
                        p.squash = 0.35f;
                        p.radius = fmaxf(p.radius, 0.00085f);
                        p.pos = L.world_center + L.axis_u * u_curr + L.axis_n * ls.height + L.axis_v * v_curr + n_leaf * (p.radius * 0.84f);
                    }
                    hit_leaf = true;
                    break;
                }
            }
        }

        if (!hit_leaf) {
            float h_water = (p.active_band == 1) ? sample_wave_height(wave_eta, p.pos.x, p.pos.z) : 0.0f;
            if (p.pos.y <= h_water) {
                if (p.active_band == 1) {
                    float vn = fmaxf(0.8f, -p.vel.y);
                    inject_pond_impulse(wave_vel, p.pos.x, p.pos.z, p.radius, vn);

                    // Distance & storm-density adaptive acoustic capture so MAX_IMPACT_EVENTS (262,144)
                    // never overflows and covers 100% of [0.0s, 20.0s]!
                    if (record_events && event_count != nullptr) {
                        float r_int = eval_rain_intensity(sim_time);
                        float keep_prob = (r_int < 0.15f) ? 1.0f : ((p.pos.z < 2.8f) ? 0.32f : 0.10f);
                        if (rand01(p.rng) < keep_prob) {
                            int e_idx = atomicAdd(event_count, 1);
                            if (e_idx < MAX_IMPACT_EVENTS) {
                                events[e_idx] = GpuImpactEvent{
                                    frame_idx, sub_frac, p.pos.x, p.pos.z,
                                    IMPACT_POND_DROP, -1,
                                    p.radius * 1000.0f, vn, 0.0f
                                };
                            }
                        }
                    }

                    // Energetic pond hits spawn a parabolic Rayleigh-jet crown droplet!
                    if (p.radius > 0.00095f && rand01(p.rng) < 0.45f) {
                        p.state = STATE_CROWN_SPLASH;
                        p.pos.y = h_water + 0.002f;
                        float az = rand01(p.rng) * 6.2831853f;
                        float vr = 0.32f + 0.55f * rand01(p.rng);
                        float vy = 0.72f + 0.78f * rand01(p.rng);
                        p.vel = Vec3(cosf(az) * vr, vy, sinf(az) * vr);
                        p.radius = clampf(p.radius * (0.58f + 0.25f * rand01(p.rng)), 0.00045f, 0.0012f);
                        p.age = 0.0f;
                        p.squash = 0.22f;
                    } else {
                        respawn_raindrop(p, sim_time, false);
                    }
                } else {
                    respawn_raindrop(p, sim_time, false);
                }
            } else if (p.pos.y < -0.5f || fabsf(p.pos.x) > 12.0f || p.pos.z > 20.0f || p.pos.z < -1.0f) {
                respawn_raindrop(p, sim_time, false);
            }
        }
    }
    // ------------------------------------------------------------------------
    // STATE 1: SUPERHYDROPHOBIC LEAF BEADING & ROLLING ("LOTUS EFFECT")
    // ------------------------------------------------------------------------
    else if (p.state == STATE_LEAF_BEAD) {
        int lid = clampf(float(p.leaf_id), 0.0f, float(NUM_LEAVES - 1));
        const LotusLeafState& L = leaves[lid];

        const float eps = 0.0015f;
        LeafSample ls_c = eval_lotus_leaf_shape(L, p.leaf_u, p.leaf_v);
        float h_u = eval_lotus_leaf_shape(L, p.leaf_u + eps, p.leaf_v).height;
        float h_v = eval_lotus_leaf_shape(L, p.leaf_u, p.leaf_v + eps).height;
        float dhdu = (h_u - ls_c.height) / eps;
        float dhdv = (h_v - ls_c.height) / eps;

        // World tangent vectors along u and v on curved leaf surface
        Vec3 tu = normalize(L.axis_u + L.axis_n * dhdu);
        Vec3 tv = normalize(L.axis_v + L.axis_n * dhdv);

        // Tangential gravitational acceleration + gentle wind push on exposed pearl dome
        Vec3 wind = eval_wind_field(p.pos.x, p.pos.z, sim_time);
        Vec3 f_spec = gravity + wind * (0.28f + 95.0f * p.radius);
        float acc_u = dot(f_spec, tu);
        float acc_v = dot(f_spec, tv);

        // Superhydrophobic Cassie-Baxter contact angle (~150 deg) hysteresis:
        // Small/medium pearls (< 2.8 mm) glisten & slowly drift along vein channels;
        // heavy merged pearls (> 3.4 mm) roll into the basin and spill off the pouring lip!
        float r_mm = p.radius * 1000.0f;
        float pin_thresh = clampf(3.25f - 0.85f * r_mm, 0.08f, 2.85f);
        float acc_mag = sqrtf(acc_u * acc_u + acc_v * acc_v + 1e-10f);

        // Gentle pouring-lip attraction for heavy coalesced pearls (r > 3.1 mm, or final pearl release at t ~ 17.2s)
        bool final_hero_release = (lid == 0 && sim_time > 16.9f && sim_time < 18.2f && r_mm > 2.35f);
        if (r_mm > 3.30f || final_hero_release) {
            float spout_u = cosf(L.pour_angle) * L.radius;
            float spout_v = sinf(L.pour_angle) * L.radius;
            Vec2 to_spout(spout_u - p.leaf_u, spout_v - p.leaf_v);
            float d_sp = sqrtf(to_spout.x * to_spout.x + to_spout.y * to_spout.y + 1e-6f);
            float pour_boost = 0.45f * fmaxf(0.2f, r_mm - 3.10f);
            if (lid == 0 && ((sim_time > 2.2f && sim_time < 3.2f) || final_hero_release)) {
                pour_boost += 2.25f;
            }
            acc_u += (to_spout.x / d_sp) * pour_boost;
            acc_v += (to_spout.y / d_sp) * pour_boost;
            acc_mag = sqrtf(acc_u * acc_u + acc_v * acc_v + 1e-10f);
        }

        if (acc_mag > pin_thresh) {
            float scale = (acc_mag - pin_thresh) / acc_mag;
            // Rolling sphere inertia factor 5/7 = 0.714
            p.vel.x += acc_u * scale * 0.714f * dt;
            p.vel.z += acc_v * scale * 0.714f * dt;
        } else {
            p.vel.x *= expf(-12.0f * dt);
            p.vel.z *= expf(-12.0f * dt);
        }

        // Low viscous rolling resistance on wax micro-papillae
        float roll_damp = expf(-1.65f * dt);
        p.vel.x *= roll_damp;
        p.vel.z *= roll_damp;

        // Clamp max rolling speed on leaf
        float spd = sqrtf(p.vel.x * p.vel.x + p.vel.z * p.vel.z);
        if (spd > 0.65f) {
            p.vel.x *= (0.65f / spd);
            p.vel.z *= (0.65f / spd);
        }

        p.leaf_u += p.vel.x * dt;
        p.leaf_v += p.vel.z * dt;

        LeafSample ls_new = eval_lotus_leaf_shape(L, p.leaf_u, p.leaf_v);
        Vec3 n_w = eval_lotus_leaf_world_normal(L, p.leaf_u, p.leaf_v);
        p.pos = L.world_center + L.axis_u * p.leaf_u + L.axis_n * ls_new.height + L.axis_v * p.leaf_v + n_w * (p.radius * 0.80f);

        // Apply continuous gravitational pearl weight torque onto the flexible lotus leaf stem!
        float pearl_weight = 4.18879f * p.radius * p.radius * p.radius * 1000.0f * 9.81f * 6.5f;
        Vec3 rel = p.pos - L.world_center;
        atomicAdd(&leaf_forces[lid * 3 + 0], -rel.z * pearl_weight * dt);
        atomicAdd(&leaf_forces[lid * 3 + 1],  rel.x * pearl_weight * dt);
        atomicAdd(&leaf_forces[lid * 3 + 2], -pearl_weight * dt);

        // Check if heavy silver pearl spills off the ruffled lotus leaf lip into the pond!
        if (ls_new.s_norm >= 1.00f) {
            p.state = STATE_CROWN_SPLASH; // Free-flight spilling pearl!
            Vec3 world_v = tu * p.vel.x + tv * p.vel.z;
            Vec3 radial_out = normalize(L.axis_u * p.leaf_u + L.axis_v * p.leaf_v);
            p.vel = world_v * 1.15f + radial_out * 0.18f + Vec3(0.0f, -0.12f, 0.0f);
            p.leaf_id = -1;
            p.age = 0.0f;
            p.squash = 0.25f;

            // Upward recoil impulse on leaf as heavy pearl detaches!
            atomicAdd(&leaf_forces[lid * 3 + 2], pearl_weight * 0.08f);
        }
    }
    // ------------------------------------------------------------------------
    // STATE 2: FREE-FLIGHT SPILLING PEARLS & CROWN SPLASH MICRO-BEADS
    // ------------------------------------------------------------------------
    else if (p.state == STATE_CROWN_SPLASH) {
        Vec3 wind = eval_wind_field(p.pos.x, p.pos.z, sim_time);
        p.vel += (gravity + (wind - Vec3(p.vel.x, 0.0f, p.vel.z)) * 0.65f) * dt;
        p.pos += p.vel * dt;

        float h_water = sample_wave_height(wave_eta, p.pos.x, p.pos.z);
        if (p.pos.y <= h_water && p.vel.y < 0.0f) {
            float vn = fmaxf(0.6f, -p.vel.y);
            bool is_heavy_pearl = (p.radius >= 0.00185f);

            inject_pond_impulse(wave_vel, p.pos.x, p.pos.z, p.radius * (is_heavy_pearl ? 1.35f : 1.0f), vn);

            if (record_events && event_count != nullptr) {
                int e_idx = atomicAdd(event_count, 1);
                if (e_idx < MAX_IMPACT_EVENTS) {
                    events[e_idx] = GpuImpactEvent{
                        frame_idx, sub_frac, p.pos.x, p.pos.z,
                        is_heavy_pearl ? IMPACT_HERO_PEARL : IMPACT_POND_DROP,
                        -1, p.radius * 1000.0f, vn, 0.0f
                    };
                }
            }

            // Heavy spilling leaf pearl rebounds a crisp central Rayleigh jet droplet once!
            if (is_heavy_pearl) {
                p.pos.y = h_water + 0.003f;
                float az = rand01(p.rng) * 6.2831853f;
                p.vel = Vec3(cosf(az) * 0.22f, 1.15f + 0.35f * rand01(p.rng), sinf(az) * 0.22f);
                p.radius = 0.00125f;
                p.age = 0.0f;
                p.squash = 0.30f;
            } else {
                respawn_raindrop(p, sim_time, false);
            }
        } else if (p.pos.y < -0.5f || p.age > 3.0f) {
            respawn_raindrop(p, sim_time, false);
        }
    }

    particles[idx] = p;
}

// ============================================================================
// Kernel 2: Build Spatial Hash Grids (Per-Leaf 64x64 + World Pond 128x128)
// ============================================================================
__global__ void build_spatial_grids_kernel(
    const DropletParticle* __restrict__ particles,
    const LotusLeafState* __restrict__ leaves,
    int* __restrict__ leaf_grid_head,
    int* __restrict__ world_grid_head,
    int* __restrict__ particle_next
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    const DropletParticle& p = particles[idx];
    particle_next[idx] = -1;

    if (p.state == STATE_LEAF_BEAD && p.leaf_id >= 0 && p.leaf_id < NUM_LEAVES) {
        int lid = p.leaf_id;
        float R = leaves[lid].radius * 1.12f;
        float gx = (p.leaf_u + R) / (2.0f * R) * float(LEAF_GRID_RES);
        float gz = (p.leaf_v + R) / (2.0f * R) * float(LEAF_GRID_RES);
        int cx = int(floorf(gx));
        int cz = int(floorf(gz));
        if (cx >= 0 && cx < LEAF_GRID_RES && cz >= 0 && cz < LEAF_GRID_RES) {
            int cell = (lid * LEAF_GRID_RES + cz) * LEAF_GRID_RES + cx;
            particle_next[idx] = atomicExch(&leaf_grid_head[cell], idx);
        }
    } else if (p.state == STATE_CROWN_SPLASH && p.pos.y > -0.04f && p.pos.y < 0.55f) {
        float gx = (p.pos.x - WAVE_X_MIN) / WAVE_SIZE * float(WORLD_GRID_RES);
        float gz = (p.pos.z - WAVE_Z_MIN) / WAVE_SIZE * float(WORLD_GRID_RES);
        int cx = int(floorf(gx));
        int cz = int(floorf(gz));
        if (cx >= 0 && cx < WORLD_GRID_RES && cz >= 0 && cz < WORLD_GRID_RES) {
            int cell = cz * WORLD_GRID_RES + cx;
            particle_next[idx] = atomicExch(&world_grid_head[cell], idx);
        }
    }
}

// ============================================================================
// Kernel 3: Volume-Conserving Superhydrophobic Pearl Coalescence on Leaves
//           r_new = (r1^3 + r2^3)^(1/3), Momentum & Center-of-Mass Preserving
// ============================================================================
__global__ void coalesce_leaf_pearls_kernel(
    DropletParticle* __restrict__ particles,
    const LotusLeafState* __restrict__ leaves,
    const int* __restrict__ leaf_grid_head,
    const int* __restrict__ particle_next,
    int* __restrict__ merge_claim,
    float sim_time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    DropletParticle p = particles[idx];
    if (p.state != STATE_LEAF_BEAD || p.leaf_id < 0 || p.leaf_id >= NUM_LEAVES) return;
    if (p.radius >= 0.0056f) return; // Cap single heavy sessile pearl at ~11 mm diameter cushion

    int lid = p.leaf_id;
    float R = leaves[lid].radius * 1.12f;
    int cx = int(floorf((p.leaf_u + R) / (2.0f * R) * float(LEAF_GRID_RES)));
    int cz = int(floorf((p.leaf_v + R) / (2.0f * R) * float(LEAF_GRID_RES)));
    if (cx < 0 || cx >= LEAF_GRID_RES || cz < 0 || cz >= LEAF_GRID_RES) return;

    for (int dz = -1; dz <= 1; ++dz) {
        int nz = cz + dz;
        if (nz < 0 || nz >= LEAF_GRID_RES) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= LEAF_GRID_RES) continue;

            int cell = (lid * LEAF_GRID_RES + nz) * LEAF_GRID_RES + nx;
            int other = leaf_grid_head[cell];
            int iter = 0;
            while (other != -1 && iter < 24) {
                ++iter;
                if (other > idx) {
                    DropletParticle q = particles[other];
                    if (q.state == STATE_LEAF_BEAD && q.leaf_id == lid) {
                        float du = p.leaf_u - q.leaf_u;
                        float dv = p.leaf_v - q.leaf_v;
                        float d2 = du * du + dv * dv;
                        float touch = 1.04f * (p.radius + q.radius);
                        if (d2 < touch * touch && (p.radius + q.radius) < 0.0082f) {
                            // Race-free dual atomic claim
                            if (atomicCAS(&merge_claim[idx], 0, 1) == 0) {
                                if (atomicCAS(&merge_claim[other], 0, 1) == 0) {
                                    float v1 = p.radius * p.radius * p.radius;
                                    float v2 = q.radius * q.radius * q.radius;
                                    float v_sum = v1 + v2;
                                    float r_new = cbrtf(v_sum);
                                    float w1 = v1 / v_sum;
                                    float w2 = v2 / v_sum;

                                    p.leaf_u = w1 * p.leaf_u + w2 * q.leaf_u;
                                    p.leaf_v = w1 * p.leaf_v + w2 * q.leaf_v;
                                    p.vel    = p.vel * w1 + q.vel * w2;
                                    p.radius = fminf(r_new, 0.0056f);
                                    p.squash = 0.34f; // Capillary coalescence wobble!

                                    const LotusLeafState& L = leaves[lid];
                                    LeafSample ls = eval_lotus_leaf_shape(L, p.leaf_u, p.leaf_v);
                                    Vec3 nw = eval_lotus_leaf_world_normal(L, p.leaf_u, p.leaf_v);
                                    p.pos = L.world_center + L.axis_u * p.leaf_u + L.axis_n * ls.height + L.axis_v * p.leaf_v + nw * (p.radius * 0.80f);
                                    particles[idx] = p;

                                    // Recycle absorbed droplet cleanly into upper sky reservoir
                                    respawn_raindrop(q, sim_time, false);
                                    particles[other] = q;
                                    return;
                                } else {
                                    atomicExch(&merge_claim[idx], 0);
                                }
                            }
                        }
                    }
                }
                other = particle_next[other];
            }
        }
    }
}

// ============================================================================
// Kernel 4A & 4B: 1024x1024 Shallow-Water Dispersive Capillary-Gravity Wave PDE
// Solves d^2(eta)/dt^2 = c_g^2 * Lap(eta) - kappa_cap * Lap(Lap(eta))
//                        + nu_visc * Lap(vel) - gamma * vel
// Reproduces dispersive capillary wavelets outrunning gravity troughs!
// ============================================================================
__global__ void wave_laplacian_kernel(
    const float* __restrict__ eta,
    float* __restrict__ lap_eta
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WAVE_N || z >= WAVE_N) return;

    int idx = z * WAVE_N + x;
    if (x < 2 || x >= WAVE_N - 2 || z < 2 || z >= WAVE_N - 2) {
        lap_eta[idx] = 0.0f;
        return;
    }

    // Isotropic 9-point Oono-Puri stencil (eliminates grid-axis diamond anisotropy!)
    float c  = eta[idx];
    float n  = eta[(z - 1) * WAVE_N + x];
    float s  = eta[(z + 1) * WAVE_N + x];
    float w  = eta[z * WAVE_N + (x - 1)];
    float e  = eta[z * WAVE_N + (x + 1)];
    float nw = eta[(z - 1) * WAVE_N + (x - 1)];
    float ne = eta[(z - 1) * WAVE_N + (x + 1)];
    float sw = eta[(z + 1) * WAVE_N + (x - 1)];
    float se = eta[(z + 1) * WAVE_N + (x + 1)];

    const float inv_dx2 = 1.0f / (WAVE_DX * WAVE_DX);
    float lap = (0.5f * (n + s + w + e) + 0.25f * (nw + ne + sw + se) - 3.0f * c) * inv_dx2;
    lap_eta[idx] = lap;
}

__global__ void wave_step_and_grad_kernel(
    float* __restrict__ eta,
    float* __restrict__ vel,
    const float* __restrict__ lap_eta,
    float2* __restrict__ grad_eta,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WAVE_N || z >= WAVE_N) return;

    int idx = z * WAVE_N + x;
    if (x < 3 || x >= WAVE_N - 3 || z < 3 || z >= WAVE_N - 3) {
        eta[idx] = 0.0f;
        vel[idx] = 0.0f;
        grad_eta[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    const float inv_dx2 = 1.0f / (WAVE_DX * WAVE_DX);

    // Biharmonic Lap(Lap(eta)) for surface-tension capillary dispersion (sigma / rho)
    float lc  = lap_eta[idx];
    float ln  = lap_eta[(z - 1) * WAVE_N + x];
    float ls  = lap_eta[(z + 1) * WAVE_N + x];
    float lw  = lap_eta[z * WAVE_N + (x - 1)];
    float le  = lap_eta[z * WAVE_N + (x + 1)];
    float lnw = lap_eta[(z - 1) * WAVE_N + (x - 1)];
    float lne = lap_eta[(z - 1) * WAVE_N + (x + 1)];
    float lsw = lap_eta[(z + 1) * WAVE_N + (x - 1)];
    float lse = lap_eta[(z + 1) * WAVE_N + (x + 1)];
    float biharm = (0.5f * (ln + ls + lw + le) + 0.25f * (lnw + lne + lsw + lse) - 3.0f * lc) * inv_dx2;

    // Viscous high-wavenumber damping Lap(vel)
    float vc = vel[idx];
    float lap_v = (vel[(z - 1) * WAVE_N + x] + vel[(z + 1) * WAVE_N + x] +
                   vel[z * WAVE_N + (x - 1)] + vel[z * WAVE_N + (x + 1)] - 4.0f * vc) * inv_dx2;

    // Physical shallow-water gravity speed c_g = 0.42 m/s + capillary dispersion kappa = 2.6e-6 m^4/s^2
    // Low damping preserves glassy concentric interference rings across the tranquil 14-20s finale!
    const float cg2       = 0.176f;
    const float kappa_cap = 2.65e-6f;
    const float nu_visc   = 6.8e-7f;
    const float gamma_0   = 0.095f;

    float acc = cg2 * lc - kappa_cap * biharm + nu_visc * lap_v - gamma_0 * vc;

    // Absorbing PML sponge layer on outer 48 cells so zero boundary reflections occur!
    int edge_dist = min(min(x, WAVE_N - 1 - x), min(z, WAVE_N - 1 - z));
    float sponge = 1.0f;
    if (edge_dist < 48) {
        float s = float(48 - edge_dist) / 48.0f;
        sponge = expf(-3.5f * s * s * dt * 60.0f);
    }

    float v_next = clampf((vc + acc * dt) * sponge, -0.35f, 0.35f);
    float h_next = clampf((eta[idx] + v_next * dt) * sponge, -0.018f, 0.018f);

    vel[idx] = v_next;
    eta[idx] = h_next;

    // Compute smooth surface slope for pond ray-tracing
    float dhdx = (eta[z * WAVE_N + (x + 1)] - eta[z * WAVE_N + (x - 1)]) * (0.5f / WAVE_DX);
    float dhdz = (eta[(z + 1) * WAVE_N + x] - eta[(z - 1) * WAVE_N + x]) * (0.5f / WAVE_DX);
    grad_eta[idx] = make_float2(dhdx, dhdz);
}
