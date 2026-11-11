#include "common.cuh"
#include "physics.cuh"
#include "renderer.cuh"
#include "audio_synth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d -> %s\n",                           \
                         __FILE__, __LINE__, cudaGetErrorString(err__));                 \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

namespace {

void init_scene(
    LotusLeafState leaves[NUM_LEAVES],
    LotusBudState buds[NUM_BUDS],
    CameraParams& cam
) {
    // ========================================================================
    // 85mm f/1.4 Macro Telephoto Camera Looking Across Emerald-Ink Lotus Pond
    // ========================================================================
    cam.pos = Vec3(0.0f, 0.48f, 0.0f);
    Vec3 target(0.0f, 0.052f, FOCAL_PLANE_Z);
    cam.forward = normalize(target - cam.pos);
    Vec3 world_up(0.0f, 1.0f, 0.0f);
    // Screen +X = world +X so +X summer wind gust sweeps cleanly LEFT-TO-RIGHT!
    cam.right = normalize(cross(world_up, cam.forward));
    cam.up    = normalize(cross(cam.forward, cam.right));
    cam.tan_half_fov_y = 0.168f; // 85mm cinema macro vertical FOV
    cam.aspect = float(IMG_W) / float(IMG_H);
    cam.focus_z = FOCAL_PLANE_Z;
    cam.coc_scale = 36.0f;       // True f/1.4 shallow DOF Circle-of-Confusion scale
    cam.max_coc_px = 26.0f;

    // Late-afternoon golden-hour sunbeams: sun_elevation = 18 deg (0.314159 rad)
    const float sun_elev = 18.0f * float(M_PI) / 180.0f;
    const float sun_az   = -24.0f * float(M_PI) / 180.0f; // 3/4 warm backlight from upper-left!
    cam.sun_dir = normalize(Vec3(
        std::cos(sun_elev) * std::sin(sun_az),
        std::sin(sun_elev),
        std::cos(sun_elev) * std::cos(sun_az)
    ));

    // ========================================================================
    // 7 Sculpted, Organically Ruffled Lotus Leaves (Varying Sizes, Heights & Modes)
    // Note: Negative base_tilt_x tilts leaf normal toward -Z (toward the camera!)
    // ========================================================================
    auto setup_leaf = [&](int id, Vec3 pos, float rad, float cup, float tx, float tz,
                          float yaw, float pour_ang, float p1, float p2, float p3, float f0_hz) {
        LotusLeafState& L = leaves[id];
        L.base_pos = pos;
        L.radius = rad;
        L.cup_depth = cup;
        L.base_tilt_x = tx;
        L.base_tilt_z = tz;
        L.yaw = yaw;
        L.pour_angle = pour_ang;
        L.phase1 = p1; L.phase2 = p2; L.phase3 = p3;
        L.fundamental_hz = f0_hz;
        L.bob_y = 0.0f; L.vel_y = 0.0f;
        L.dyn_tilt_x = 0.0f; L.dyn_tilt_z = 0.0f;
        L.omega_x = 0.0f; L.omega_z = 0.0f;
        update_leaf_frame(L);
    };

    // Leaf 0: HERO MIDGROUND LOTUS LEAF (Locked sharp at z_focus = 1.90m, cupped top tilted toward lens!)
    setup_leaf(0, Vec3(-0.045f, 0.048f, 1.90f), 0.225f, 0.036f, -0.210f, -0.070f,  0.10f, -0.62f, 0.4f, 1.7f, 2.9f, 188.0f);
    // Leaf 1: Foreground Left Floating Pad (Skimming pond surface in creamy foreground bokeh)
    setup_leaf(1, Vec3(-0.245f, 0.008f, 1.35f), 0.138f, 0.015f, -0.055f,  0.035f,  1.10f,  0.35f, 1.2f, 0.5f, 4.1f, 338.0f);
    // Leaf 2: Midground Right Companion Emergent Leaf (Tilted to glow with warm 18-deg backlit vein SSS!)
    setup_leaf(2, Vec3( 0.340f, 0.138f, 2.26f), 0.185f, 0.034f, -0.250f,  0.145f, -0.28f, -2.10f, 2.3f, 3.1f, 0.8f, 242.0f);
    // Leaf 3: Mid-Foreground Right Floating Pad
    setup_leaf(3, Vec3( 0.295f, 0.009f, 1.63f), 0.122f, 0.014f, -0.060f,  0.045f,  2.20f, -1.75f, 3.8f, 1.9f, 5.2f, 396.0f);
    // Leaf 4: Mid-Background Left Stately Emergent Umbrella
    setup_leaf(4, Vec3(-0.460f, 0.185f, 2.82f), 0.225f, 0.040f, -0.195f, -0.140f,  0.45f, -0.40f, 0.9f, 4.4f, 1.6f, 166.0f);
    // Leaf 5: Background Center-Right Floating Pad Amidst Sunlit Pond Sparkles
    setup_leaf(5, Vec3( 0.140f, 0.010f, 3.30f), 0.190f, 0.020f, -0.055f,  0.025f, -1.40f, -1.20f, 4.7f, 2.2f, 3.3f, 212.0f);
    // Leaf 6: Far Background Upper-Center Canopy Leaf (Cleanly framed inside 16:9 telephoto view!)
    setup_leaf(6, Vec3(-0.075f, 0.142f, 3.85f), 0.235f, 0.042f, -0.185f, -0.070f,  0.20f, -0.80f, 5.4f, 0.7f, 2.5f, 148.0f);

    // ========================================================================
    // 2 Half-Opened Pale Ivory-Pink Sacred Lotus Buds (Nelumbo nucifera)
    // ========================================================================
    buds[0].base_pos    = Vec3( 0.195f, 0.118f, 2.04f);
    buds[0].stem_anchor = Vec3( 0.165f, -0.14f, 2.08f);
    buds[0].radius      = 0.050f;
    buds[0].height      = 0.132f;
    buds[0].tilt_x      = -0.065f;
    buds[0].tilt_z      = -0.085f;
    buds[0].yaw         = 0.42f;
    buds[0].open_amount = 0.50f;
    update_bud_frame(buds[0], 0.0f, 0.0f);

    buds[1].base_pos    = Vec3(-0.265f, 0.158f, 3.08f);
    buds[1].stem_anchor = Vec3(-0.295f, -0.14f, 3.12f);
    buds[1].radius      = 0.056f;
    buds[1].height      = 0.146f;
    buds[1].tilt_x      = -0.065f;
    buds[1].tilt_z      = 0.105f;
    buds[1].yaw         = -0.65f;
    buds[1].open_amount = 0.56f;
    update_bud_frame(buds[1], 0.0f, 0.0f);
}

// Integrate coupled spring-damper rigid-stem bobbing & tilting under droplet torque + wind gust
void step_leaves_and_buds_dynamics(
    LotusLeafState leaves[NUM_LEAVES],
    LotusBudState buds[NUM_BUDS],
    const float host_leaf_forces[NUM_LEAVES * 3],
    float sim_time,
    float dt_frame
) {
    for (int i = 0; i < NUM_LEAVES; ++i) {
        LotusLeafState& L = leaves[i];
        Vec3 w = eval_wind_field(L.base_pos.x, L.base_pos.z, sim_time);

        // Flexible emergent stems sway & tilt more than low water-supported floating pads!
        float stem_flex = (L.base_pos.y > 0.05f) ? 1.0f : 0.32f;
        float tx_torque = host_leaf_forces[i * 3 + 0] * 1.4f + w.z * 0.18f * stem_flex;
        // Wind in +X tilts leaf clockwise around Z (dyn_tilt_z < 0 -> dips right lip!)
        float tz_torque = host_leaf_forces[i * 3 + 1] * 1.4f - w.x * 0.115f * stem_flex;
        float fy_force  = host_leaf_forces[i * 3 + 2] * 0.045f
                        + 0.004f * std::sin(sim_time * 1.9f + float(i) * 1.3f);

        const float k_rot  = 18.0f;
        const float c_rot  = 5.2f;
        const float k_bob  = 24.0f;
        const float c_bob  = 6.4f;

        float acc_tx = tx_torque - k_rot * L.dyn_tilt_x - c_rot * L.omega_x;
        float acc_tz = tz_torque - k_rot * L.dyn_tilt_z - c_rot * L.omega_z;
        float acc_y  = fy_force  - k_bob * L.bob_y      - c_bob * L.vel_y;

        L.omega_x    += acc_tx * dt_frame;
        L.omega_z    += acc_tz * dt_frame;
        L.vel_y      += acc_y  * dt_frame;

        L.dyn_tilt_x = std::clamp(L.dyn_tilt_x + L.omega_x * dt_frame, -0.22f, 0.22f);
        L.dyn_tilt_z = std::clamp(L.dyn_tilt_z + L.omega_z * dt_frame, -0.26f, 0.22f);
        L.bob_y      = std::clamp(L.bob_y      + L.vel_y   * dt_frame, -0.022f, 0.018f);

        update_leaf_frame(L);
    }

    for (int b = 0; b < NUM_BUDS; ++b) {
        Vec3 w = eval_wind_field(buds[b].base_pos.x, buds[b].base_pos.z, sim_time);
        float flutter = 0.012f * std::sin(sim_time * 3.4f + float(b) * 2.1f);
        float sway_x = std::clamp(w.z * 0.045f + flutter, -0.16f, 0.16f);
        float sway_z = std::clamp(-w.x * 0.068f + flutter * 0.8f, -0.24f, 0.16f);
        update_bud_frame(buds[b], sway_x, sway_z);
    }
}

void write_ppm_image(const std::string& path, const std::vector<uchar3>& pixels, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), size_t(w) * size_t(h) * 3u);
}

} // namespace

int main(int argc, char** argv) {
    bool diag_mode          = false;
    bool preview_mode       = false;
    bool export_events_only = false;
    std::string out_dir = ".";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--diag") diag_mode = true;
        else if (arg == "--preview") preview_mode = true;
        else if (arg == "--export-events") export_events_only = true;
        else if (arg == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
    }

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        std::cerr << "No CUDA-capable GPU detected!\n";
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "[GPU] " << prop.name << " (SM " << prop.major << "." << prop.minor
              << ", VRAM: " << (prop.totalGlobalMem >> 20) << " MiB)\n";

    // Initialize scene state
    LotusLeafState h_leaves[NUM_LEAVES];
    LotusBudState  h_buds[NUM_BUDS];
    CameraParams   cam{};
    init_scene(h_leaves, h_buds, cam);

    // Allocate GPU Simulation & Render Context (~112 MiB total VRAM)
    GpuSimContext sim{};
    GpuRenderBuffers ren{};

    CUDA_CHECK(cudaMalloc(&sim.d_particles, NUM_PARTICLES * sizeof(DropletParticle)));
    CUDA_CHECK(cudaMalloc(&sim.d_leaves,    NUM_LEAVES * sizeof(LotusLeafState)));
    CUDA_CHECK(cudaMalloc(&sim.d_buds,      NUM_BUDS * sizeof(LotusBudState)));
    CUDA_CHECK(cudaMalloc(&sim.d_leaf_forces, NUM_LEAVES * 3 * sizeof(float)));

    const size_t wave_cells = size_t(WAVE_N) * size_t(WAVE_N);
    CUDA_CHECK(cudaMalloc(&sim.d_wave_eta,  wave_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sim.d_wave_vel,  wave_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sim.d_wave_lap,  wave_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sim.d_wave_grad, wave_cells * sizeof(float2)));

    CUDA_CHECK(cudaMalloc(&sim.d_leaf_grid_head,  NUM_LEAVES * LEAF_GRID_RES * LEAF_GRID_RES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_world_grid_head, WORLD_GRID_RES * WORLD_GRID_RES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_particle_next,   NUM_PARTICLES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_merge_claim,     NUM_PARTICLES * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&sim.d_events,      MAX_IMPACT_EVENTS * sizeof(GpuImpactEvent)));
    CUDA_CHECK(cudaMalloc(&sim.d_event_count, sizeof(int)));

    const size_t num_pixels = size_t(IMG_W) * size_t(IMG_H);
    CUDA_CHECK(cudaMalloc(&ren.d_hdr_scene,   num_pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&ren.d_bokeh_accum, num_pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&ren.d_srgb_out,    num_pixels * sizeof(uchar3)));

    uchar3* h_pinned_rgb = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_rgb, num_pixels * sizeof(uchar3)));

    CUDA_CHECK(cudaMemset(sim.d_wave_eta,  0, wave_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(sim.d_wave_vel,  0, wave_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(sim.d_wave_lap,  0, wave_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(sim.d_wave_grad, 0, wave_cells * sizeof(float2)));
    CUDA_CHECK(cudaMemset(sim.d_event_count, 0, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(sim.d_leaves, h_leaves, NUM_LEAVES * sizeof(LotusLeafState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_buds,   h_buds,   NUM_BUDS * sizeof(LotusBudState),    cudaMemcpyHostToDevice));

    // Launch configuration helpers
    const int p_threads = 256;
    const int p_blocks  = (NUM_PARTICLES + p_threads - 1) / p_threads;
    const dim3 wave_block(16, 16);
    const dim3 wave_grid((WAVE_N + 15) / 16, (WAVE_N + 15) / 16);
    const dim3 img_block(16, 16);
    const dim3 img_grid((IMG_W + 15) / 16, (IMG_H + 15) / 16);
    const dim3 half_img_grid(((IMG_W / 2) + 15) / 16, ((IMG_H / 2) + 15) / 16);

    init_particles_kernel<<<p_blocks, p_threads>>>(sim.d_particles, sim.d_leaves);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto step_one_frame_physics = [&](int frame_idx, float sim_time, bool record_events) {
        const float dt_frame = 1.0f / float(FPS);
        CUDA_CHECK(cudaMemset(sim.d_leaf_forces, 0, NUM_LEAVES * 3 * sizeof(float)));

        // 2 particle substeps + 6 shallow-water capillary-gravity wave substeps per 60fps frame
        constexpr int P_SUB = 2;
        const float dt_p = dt_frame / float(P_SUB);
        for (int ps = 0; ps < P_SUB; ++ps) {
            float sub_frac = float(ps) / float(P_SUB);
            step_particles_kernel<<<p_blocks, p_threads>>>(
                sim.d_particles, sim.d_leaves, sim.d_leaf_forces,
                sim.d_wave_eta, sim.d_wave_vel,
                sim.d_events, sim.d_event_count,
                frame_idx, sub_frac, sim_time + sub_frac * dt_frame, dt_p, record_events
            );
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemset(sim.d_leaf_grid_head,  0xFF, NUM_LEAVES * LEAF_GRID_RES * LEAF_GRID_RES * sizeof(int)));
            CUDA_CHECK(cudaMemset(sim.d_world_grid_head, 0xFF, WORLD_GRID_RES * WORLD_GRID_RES * sizeof(int)));
            CUDA_CHECK(cudaMemset(sim.d_merge_claim,     0,    NUM_PARTICLES * sizeof(int)));

            build_spatial_grids_kernel<<<p_blocks, p_threads>>>(
                sim.d_particles, sim.d_leaves,
                sim.d_leaf_grid_head, sim.d_world_grid_head, sim.d_particle_next
            );
            CUDA_CHECK(cudaGetLastError());

            coalesce_leaf_pearls_kernel<<<p_blocks, p_threads>>>(
                sim.d_particles, sim.d_leaves,
                sim.d_leaf_grid_head, sim.d_particle_next, sim.d_merge_claim,
                sim_time
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // Rebuild clean spatial grid after merges for exact pearl ray-tracing & caustics
        CUDA_CHECK(cudaMemset(sim.d_leaf_grid_head,  0xFF, NUM_LEAVES * LEAF_GRID_RES * LEAF_GRID_RES * sizeof(int)));
        CUDA_CHECK(cudaMemset(sim.d_world_grid_head, 0xFF, WORLD_GRID_RES * WORLD_GRID_RES * sizeof(int)));
        build_spatial_grids_kernel<<<p_blocks, p_threads>>>(
            sim.d_particles, sim.d_leaves,
            sim.d_leaf_grid_head, sim.d_world_grid_head, sim.d_particle_next
        );
        CUDA_CHECK(cudaGetLastError());

        constexpr int W_SUB = 6;
        const float dt_w = dt_frame / float(W_SUB);
        for (int ws = 0; ws < W_SUB; ++ws) {
            wave_laplacian_kernel<<<wave_grid, wave_block>>>(sim.d_wave_eta, sim.d_wave_lap);
            wave_step_and_grad_kernel<<<wave_grid, wave_block>>>(
                sim.d_wave_eta, sim.d_wave_vel, sim.d_wave_lap, sim.d_wave_grad, dt_w
            );
        }
        CUDA_CHECK(cudaGetLastError());

        float h_forces[NUM_LEAVES * 3] = {};
        CUDA_CHECK(cudaMemcpy(h_forces, sim.d_leaf_forces, sizeof(h_forces), cudaMemcpyDeviceToHost));
        step_leaves_and_buds_dynamics(h_leaves, h_buds, h_forces, sim_time, dt_frame);
        CUDA_CHECK(cudaMemcpy(sim.d_leaves, h_leaves, NUM_LEAVES * sizeof(LotusLeafState), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.d_buds,   h_buds,   NUM_BUDS * sizeof(LotusBudState),    cudaMemcpyHostToDevice));
    };

    auto render_one_frame = [&](float sim_time, int spp) {
        CUDA_CHECK(cudaMemset(ren.d_bokeh_accum, 0, num_pixels * sizeof(float4)));

        render_primary_raytrace_kernel<<<img_grid, img_block>>>(
            ren.d_hdr_scene, cam, sim.d_leaves, sim.d_buds,
            sim.d_particles, sim.d_leaf_grid_head, sim.d_world_grid_head, sim.d_particle_next,
            sim.d_wave_eta, sim.d_wave_grad, sim_time, spp
        );
        CUDA_CHECK(cudaGetLastError());

        splat_rain_streaks_kernel<<<p_blocks, p_threads>>>(
            ren.d_bokeh_accum, ren.d_hdr_scene, sim.d_particles, cam
        );
        CUDA_CHECK(cudaGetLastError());

        scatter_specular_bokeh_discs_kernel<<<half_img_grid, img_block>>>(
            ren.d_hdr_scene, ren.d_bokeh_accum, cam
        );
        CUDA_CHECK(cudaGetLastError());

        dof_gather_and_tonemap_kernel<<<img_grid, img_block>>>(
            ren.d_srgb_out, ren.d_hdr_scene, ren.d_bokeh_accum, cam, sim_time
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_pinned_rgb, ren.d_srgb_out, num_pixels * sizeof(uchar3), cudaMemcpyDeviceToHost));
    };

    // ========================================================================
    // Warm-up Pre-Roll (75 frames = 1.25s) so Pond Ripples & Leaf Dew Settle Naturally
    // ========================================================================
    std::cout << "[Sim] Running 75-step physics warm-up pre-roll...\n";
    for (int w = -75; w < 0; ++w) {
        step_one_frame_physics(w, 0.0f, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Mode 1: Fast Diagnostic Self-Test (--diag)
    // Checks memory bounds, NaNs/Infs in particles, wave grid, HDR & sRGB buffers!
    // ========================================================================
    if (diag_mode) {
        std::cout << "[Diag] Running 30-step GPU physics + full 1920x1080 render + NaN/bounds audit...\n";
        auto t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < 30; ++f) {
            step_one_frame_physics(f, float(f) / float(FPS), true);
        }
        render_one_frame(0.5f, 4);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::vector<DropletParticle> hp(NUM_PARTICLES);
        CUDA_CHECK(cudaMemcpy(hp.data(), sim.d_particles, NUM_PARTICLES * sizeof(DropletParticle), cudaMemcpyDeviceToHost));
        int nan_p = 0, c_rain = 0, c_bead = 0, c_splash = 0;
        for (const auto& p : hp) {
            if (!std::isfinite(p.pos.x) || !std::isfinite(p.pos.y) || !std::isfinite(p.pos.z) ||
                !std::isfinite(p.vel.x) || !std::isfinite(p.vel.y) || !std::isfinite(p.vel.z) ||
                !std::isfinite(p.radius) || p.radius <= 0.0f) {
                nan_p++;
            }
            if (p.state == STATE_FALLING_RAIN) c_rain++;
            else if (p.state == STATE_LEAF_BEAD) c_bead++;
            else if (p.state == STATE_CROWN_SPLASH) c_splash++;
        }

        std::vector<float> h_eta(wave_cells);
        CUDA_CHECK(cudaMemcpy(h_eta.data(), sim.d_wave_eta, wave_cells * sizeof(float), cudaMemcpyDeviceToHost));
        int nan_w = 0;
        float max_eta = 0.0f;
        for (float v : h_eta) {
            if (!std::isfinite(v)) nan_w++;
            else max_eta = std::max(max_eta, std::abs(v));
        }

        std::vector<float4> h_hdr(num_pixels);
        CUDA_CHECK(cudaMemcpy(h_hdr.data(), ren.d_hdr_scene, num_pixels * sizeof(float4), cudaMemcpyDeviceToHost));
        int nan_hdr = 0;
        for (const auto& px : h_hdr) {
            if (!std::isfinite(px.x) || !std::isfinite(px.y) || !std::isfinite(px.z) || !std::isfinite(px.w)) {
                nan_hdr++;
            }
        }

        int ev_cnt = 0;
        CUDA_CHECK(cudaMemcpy(&ev_cnt, sim.d_event_count, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "[Diag] Completed in " << ms << " ms\n"
                  << "       Particles (total=" << NUM_PARTICLES << "): rain=" << c_rain
                  << ", leaf_beads=" << c_bead << ", crown_splash=" << c_splash
                  << ", NaNs=" << nan_p << "\n"
                  << "       WaveGrid (1024x1024): max|eta|=" << (max_eta * 1000.0f) << " mm, NaNs=" << nan_w << "\n"
                  << "       HDR Frame (1920x1080): NaNs=" << nan_hdr << "\n"
                  << "       Recorded Acoustic Events (0.5s): " << ev_cnt << "\n";

        if (nan_p > 0 || nan_w > 0 || nan_hdr > 0) {
            std::cerr << "[Diag] FAILED NaN check!\n";
            return EXIT_FAILURE;
        }
        std::cout << "[Diag] ALL CHECKS PASSED ZERO NaNs!\n";
        return EXIT_SUCCESS;
    }

    // ========================================================================
    // Mode 2: Keyframe Art-Direction Preview + Full 20s Audio Audit (--preview)
    // Renders the 6 mandatory QA frames (0.5s, 3.5s, 7.5s, 11.0s, 15.0s, 18.5s)
    // + synthesizes the coupled 48kHz 24-bit WAV in seconds!
    // ========================================================================
    const int qa_frames[6] = { 30, 210, 450, 660, 900, 1110 };
    const char* qa_names[6] = {
        "qa_t00_5s.ppm", "qa_t03_5s.ppm", "qa_t07_5s.ppm",
        "qa_t11_0s.ppm", "qa_t15_0s.ppm", "qa_t18_5s.ppm"
    };

    FILE* ffmpeg_pipe = nullptr;
    std::string temp_video_mp4 = out_dir + "/lotus_rain_video_track.mp4";
    std::string wav_path       = out_dir + "/lotus_rain_65536.wav";
    std::string events_bin     = out_dir + "/gpu_impact_events.bin";
    std::string final_mp4      = out_dir + "/lotus_rain_65536_cuda.mp4";

    if (!preview_mode && !export_events_only) {
        std::string cmd = "ffmpeg -y -loglevel warning -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 60 -i - "
                          "-c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p \"" + temp_video_mp4 + "\"";
        std::cout << "[Render] Opening FFmpeg H.264 encoder pipe:\n  " << cmd << "\n";
        ffmpeg_pipe = POPEN(cmd.c_str(), "wb");
        if (!ffmpeg_pipe) {
            std::cerr << "[Error] Failed to open ffmpeg pipe!\n";
            return EXIT_FAILURE;
        }
#ifdef _WIN32
        _setmode(_fileno(ffmpeg_pipe), _O_BINARY);
#endif
    }

    auto t_start = std::chrono::steady_clock::now();
    std::vector<uchar3> host_copy(num_pixels);

    for (int f = 0; f < TOTAL_FRAMES; ++f) {
        float sim_time = float(f) / float(FPS);
        step_one_frame_physics(f, sim_time, true);

        bool is_qa = false;
        int qa_idx = -1;
        for (int k = 0; k < 6; ++k) {
            if (f == qa_frames[k]) { is_qa = true; qa_idx = k; break; }
        }

        if (!preview_mode && !export_events_only) {
            render_one_frame(sim_time, 4);
            std::fwrite(h_pinned_rgb, sizeof(uchar3), num_pixels, ffmpeg_pipe);
        } else if (preview_mode && is_qa) {
            render_one_frame(sim_time, 4);
            std::memcpy(host_copy.data(), h_pinned_rgb, num_pixels * sizeof(uchar3));
            std::string ppath = out_dir + "/" + qa_names[qa_idx];
            write_ppm_image(ppath, host_copy, IMG_W, IMG_H);
            std::cout << "[Preview] Wrote full-res 1920x1080 QA frame t=" << sim_time << "s -> " << ppath << "\n";
        }

        if ((f + 1) % 120 == 0 || f == TOTAL_FRAMES - 1) {
            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double fps_rate = double(f + 1) / std::max(0.001, elapsed);
            std::cout << "[Progress] Frame " << (f + 1) << "/" << TOTAL_FRAMES
                      << " (t=" << (float(f + 1) / float(FPS)) << "s, "
                      << fps_rate << " fps)\n";
        }
    }

    if (ffmpeg_pipe) {
        int rc = PCLOSE(ffmpeg_pipe);
        if (rc != 0) {
            std::cerr << "[Error] FFmpeg video encode exited with code " << rc << "\n";
            return EXIT_FAILURE;
        }
    }

    // ========================================================================
    // Export Timestamped GPU Impact Events -> Binary File & Acoustic Synth
    // ========================================================================
    static_assert(sizeof(GpuImpactEvent) == 36, "GpuImpactEvent binary layout must be 36 bytes");
    int raw_event_count = 0;
    CUDA_CHECK(cudaMemcpy(&raw_event_count, sim.d_event_count, sizeof(int), cudaMemcpyDeviceToHost));
    int clamped_events = std::min(raw_event_count, MAX_IMPACT_EVENTS);
    std::cout << "[Audio] Exporting " << clamped_events << " coupled GPU impact events (raw=" << raw_event_count << ")...\n";

    std::vector<GpuImpactEvent> host_events(clamped_events);
    if (clamped_events > 0) {
        CUDA_CHECK(cudaMemcpy(host_events.data(), sim.d_events, clamped_events * sizeof(GpuImpactEvent), cudaMemcpyDeviceToHost));
    }

    {
        FILE* evf = std::fopen(events_bin.c_str(), "wb");
        if (evf) {
            if (clamped_events > 0) {
                std::fwrite(host_events.data(), sizeof(GpuImpactEvent), clamped_events, evf);
            }
            std::fclose(evf);
            std::cout << "[Audio] Saved binary GPU impact events -> " << events_bin << " (" << clamped_events << " records)\n";
        }
    }

    if (!export_events_only) {
        AudioDiagnostics adiag = synthesize_coupled_pond_audio_wav(
            host_events, h_leaves, wav_path, DURATION_SEC, 48000
        );

        std::cout << "[Audio] Synthesized 48kHz 24-bit stereo WAV: " << wav_path << "\n"
                  << "        Pond plinks: " << adiag.pond_events
                  << " | Hero pearl plunges: " << adiag.pearl_events
                  << " | Lotus leaf strikes: " << adiag.leaf_events << "\n"
                  << "        RMS(0-4s)=" << adiag.rms_0_4s
                  << " -> RMS(4-14s)=" << adiag.rms_4_14s
                  << " -> RMS(14-20s)=" << adiag.rms_14_20s
                  << " | Peak |sample|=" << adiag.max_abs_sample
                  << " | NaNs=" << (adiag.has_nan ? "YES" : "NO") << "\n";
    }

    // ========================================================================
    // Mux H.264 Video + 48kHz 24-bit WAV into Final lotus_rain_65536_cuda.mp4
    // ========================================================================
    if (!preview_mode && !export_events_only) {
        std::string mux_cmd = "ffmpeg -y -loglevel warning -i \"" + temp_video_mp4 + "\" -i \"" + wav_path + "\" "
                              "-c:v copy -c:a aac -b:a 320k -ar 48000 -shortest \"" + final_mp4 + "\"";
        std::cout << "[Mux] Muxing video & coupled stereo audio:\n  " << mux_cmd << "\n";
        int mrc = std::system(mux_cmd.c_str());
        if (mrc != 0) {
            std::cerr << "[Error] FFmpeg mux failed with code " << mrc << "\n";
            return EXIT_FAILURE;
        }
        std::remove(temp_video_mp4.c_str());
        std::cout << "[Success] Master cinema output ready: " << final_mp4 << "\n";
    }

    // Cleanup GPU memory
    cudaFreeHost(h_pinned_rgb);
    cudaFree(ren.d_hdr_scene);
    cudaFree(ren.d_bokeh_accum);
    cudaFree(ren.d_srgb_out);
    cudaFree(sim.d_particles);
    cudaFree(sim.d_leaves);
    cudaFree(sim.d_buds);
    cudaFree(sim.d_leaf_forces);
    cudaFree(sim.d_wave_eta);
    cudaFree(sim.d_wave_vel);
    cudaFree(sim.d_wave_lap);
    cudaFree(sim.d_wave_grad);
    cudaFree(sim.d_leaf_grid_head);
    cudaFree(sim.d_world_grid_head);
    cudaFree(sim.d_particle_next);
    cudaFree(sim.d_merge_claim);
    cudaFree(sim.d_events);
    cudaFree(sim.d_event_count);

    return EXIT_SUCCESS;
}
