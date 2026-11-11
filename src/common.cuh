#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HD __host__ __device__ __forceinline__

// ============================================================================
// Simulation & Cinema Constants
// ============================================================================
constexpr int IMG_W = 1920;
constexpr int IMG_H = 1080;
constexpr int FPS = 60;
constexpr int TOTAL_FRAMES = 1200; // 20.0 seconds @ 60 fps
constexpr float DURATION_SEC = 20.0f;

constexpr int NUM_PARTICLES = 65536; // 2^16 unified water elements
constexpr int NUM_LEAVES = 7;
constexpr int NUM_BUDS = 2;

// 1024x1024 Shallow-Water Capillary-Gravity Wave Grid
constexpr int WAVE_N = 1024;
constexpr float WAVE_X_MIN = -3.20f;
constexpr float WAVE_X_MAX =  3.20f;
constexpr float WAVE_Z_MIN =  0.35f;
constexpr float WAVE_Z_MAX =  6.75f;
constexpr float WAVE_SIZE  =  6.40f;
constexpr float WAVE_DX    =  WAVE_SIZE / float(WAVE_N); // 6.25 mm cell size

// Spatial acceleration grids
constexpr int LEAF_GRID_RES = 64;
constexpr int WORLD_GRID_RES = 128;
constexpr int MAX_IMPACT_EVENTS = 524288;

// Optical 85mm f/1.4 Macro Telephoto Camera
constexpr float FOCAL_PLANE_Z = 1.90f; // Locked sharp on Hero Midground Lotus Leaf

// Particle states
enum ParticleState : int {
    STATE_FALLING_RAIN = 0,
    STATE_LEAF_BEAD    = 1,
    STATE_CROWN_SPLASH = 2
};

// Acoustic impact event types
enum ImpactType : int {
    IMPACT_POND_DROP   = 0,
    IMPACT_HERO_PEARL  = 1,
    IMPACT_LOTUS_LEAF  = 2
};

// ============================================================================
// Vector Math
// ============================================================================
struct Vec2 {
    float x, y;
    HD Vec2() : x(0.f), y(0.f) {}
    HD Vec2(float _x, float _y) : x(_x), y(_y) {}
    HD Vec2 operator+(const Vec2& b) const { return Vec2(x + b.x, y + b.y); }
    HD Vec2 operator-(const Vec2& b) const { return Vec2(x - b.x, y - b.y); }
    HD Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
};

struct Vec3 {
    float x, y, z;
    HD Vec3() : x(0.f), y(0.f), z(0.f) {}
    HD Vec3(float v) : x(v), y(v), z(v) {}
    HD Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    HD Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    HD Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    HD Vec3 operator-() const { return Vec3(-x, -y, -z); }
    HD Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    HD Vec3 operator/(float s) const { float inv = 1.0f / s; return Vec3(x * inv, y * inv, z * inv); }
    HD Vec3 operator*(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
    HD Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    HD Vec3& operator-=(const Vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    HD Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

HD Vec3 operator*(float s, const Vec3& v) { return Vec3(v.x * s, v.y * s, v.z * s); }
HD float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
HD float length_sq(const Vec3& v) { return dot(v, v); }
HD float length(const Vec3& v) { return sqrtf(fmaxf(dot(v, v), 1e-20f)); }
HD Vec3 normalize(const Vec3& v) {
    float l2 = dot(v, v);
    if (l2 < 1e-20f) return Vec3(0.f, 1.f, 0.f);
    float inv = 1.0f / sqrtf(l2);
    return v * inv;
}

HD float clampf(float x, float a, float b) { return fminf(fmaxf(x, a), b); }
HD Vec3 clamp3(const Vec3& v, float a, float b) {
    return Vec3(clampf(v.x, a, b), clampf(v.y, a, b), clampf(v.z, a, b));
}
HD float lerpf(float a, float b, float t) { return a + (b - a) * t; }
HD Vec3 lerp3(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }

HD float smoothstepf(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0 + 1e-12f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Smooth minimum (polynomial smin) for liquid metaball coalescence
HD float sminf(float a, float b, float k) {
    float h = clampf(0.5f + 0.5f * (b - a) / fmaxf(k, 1e-6f), 0.0f, 1.0f);
    return lerpf(b, a, h) - k * h * (1.0f - h);
}

HD float wrap_angle(float a) {
    a = fmodf(a + 3.14159265f, 6.28318530f);
    if (a < 0.0f) a += 6.28318530f;
    return a - 3.14159265f;
}

HD Vec3 reflect_vec(const Vec3& I, const Vec3& N) {
    return I - 2.0f * dot(I, N) * N;
}

HD bool refract_vec(const Vec3& I, const Vec3& N, float eta, Vec3& T) {
    float cosi = -dot(I, N);
    float sin2t = eta * eta * (1.0f - cosi * cosi);
    if (sin2t > 1.0f) return false;
    float cost = sqrtf(fmaxf(0.0f, 1.0f - sin2t));
    T = eta * I + (eta * cosi - cost) * N;
    return true;
}

// Exact dielectric unpolarized Fresnel reflectance (water IOR = 1.333)
HD float fresnel_dielectric(float cosi, float ior) {
    cosi = clampf(cosi, -1.0f, 1.0f);
    float etai = 1.0f, etat = ior;
    if (cosi < 0.0f) {
        cosi = -cosi;
        float tmp = etai; etai = etat; etat = tmp;
    }
    float sint = (etai / etat) * sqrtf(fmaxf(0.0f, 1.0f - cosi * cosi));
    if (sint >= 1.0f) return 1.0f;
    float cost = sqrtf(fmaxf(0.0f, 1.0f - sint * sint));
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost) + 1e-8f);
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost) + 1e-8f);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// ============================================================================
// Deterministic GPU PRNG & Procedural Botanical Noise (Zero External Textures)
// ============================================================================
HD uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

HD float rand01(uint32_t& state) {
    state = pcg_hash(state);
    return float(state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

HD float hash21(float x, float y) {
    int ix = int(floorf(x));
    int iy = int(floorf(y));
    uint32_t h = pcg_hash(uint32_t(ix * 73856093) ^ uint32_t(iy * 19349663));
    return float(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

HD float noise2d(float x, float y) {
    float ix = floorf(x), iy = floorf(y);
    float fx = x - ix, fy = y - iy;
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    float a = hash21(ix, iy);
    float b = hash21(ix + 1.0f, iy);
    float c = hash21(ix, iy + 1.0f);
    float d = hash21(ix + 1.0f, iy + 1.0f);
    return lerpf(lerpf(a, b, ux), lerpf(c, d, ux), uy);
}

HD float fbm2d(float x, float y, int octaves = 4) {
    float val = 0.0f;
    float amp = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        val += amp * noise2d(x, y);
        // Rotate octave slightly to eliminate grid axis alignment
        float nx = 1.6f * x - 1.2f * y + 13.7f;
        float ny = 1.2f * x + 1.6f * y + 7.3f;
        x = nx; y = ny;
        amp *= 0.5f;
    }
    return val;
}

// ============================================================================
// Dramatic 20-Second Arc: Rain Envelope & Time-Varying Summer Breeze W(x,z,t)
// ============================================================================
// 0.0s - 4.0s: Sparse sunlit droplets + hero pearl rolling off leaf
// 4.0s - 14.0s: Warm summer wind gust sweeps left-to-right + full 65,536 sun-shower
// 14.0s - 20.0s: Rain tapers off into tranquil golden ripples & final slow pearl drop
HD float eval_rain_intensity(float t) {
    if (t < 0.0f) return 0.024f;
    float ramp_up   = smoothstepf(3.8f, 6.8f, t);
    float ramp_down = 1.0f - smoothstepf(12.6f, 15.6f, t);
    float storm = ramp_up * ramp_down;
    float opening = lerpf(0.022f, 0.045f, clampf(t / 4.0f, 0.0f, 1.0f));
    float tail    = lerpf(0.032f, 0.014f, smoothstepf(14.5f, 19.8f, t));
    float base    = (t < 4.2f) ? opening : tail;
    return clampf(base + 0.96f * storm, 0.012f, 1.0f);
}

HD Vec3 eval_wind_field(float x, float z, float t) {
    // Gentle ambient summer air + sweeping warm left-to-right gust in [4.0s, 14.0s]
    float gust_env = smoothstepf(3.8f, 6.2f, t) * (1.0f - smoothstepf(12.2f, 14.8f, t));
    // Travelling gust front sweeping left-to-right (+X)
    float phase = (t - 4.0f) * 2.1f - x * 0.85f - z * 0.22f;
    float wave1 = 0.65f + 0.35f * sinf(phase) + 0.18f * sinf(phase * 2.3f + 1.1f);
    float turb_x = noise2d(x * 1.4f - t * 1.7f, z * 1.4f + t * 0.5f) - 0.5f;
    float turb_z = noise2d(x * 1.3f + 11.3f - t * 1.4f, z * 1.3f + 5.7f) - 0.5f;

    float ambient_x = 0.14f + 0.06f * sinf(t * 0.7f + z * 0.4f);
    float ambient_z = -0.05f + 0.04f * cosf(t * 0.5f + x * 0.5f);

    float wx = ambient_x + gust_env * (2.35f * wave1 + 0.65f * turb_x);
    float wy = gust_env * (0.12f * sinf(phase * 1.7f) + 0.15f * turb_z);
    float wz = ambient_z + gust_env * (-0.32f * wave1 + 0.45f * turb_z);
    return Vec3(wx, wy, wz);
}

// ============================================================================
// Scene Data Structures: 7 Lotus Leaves, 2 Lotus Buds, 65,536 Particles, Events
// ============================================================================
struct LotusLeafState {
    Vec3 base_pos;      // Resting world anchor
    float radius;       // Nominal leaf radius R0 (m)
    float cup_depth;    // Central bowl depth (m)
    float base_tilt_x;  // Resting pitch (rad)
    float base_tilt_z;  // Resting roll (rad)
    float yaw;          // Azimuthal rotation (rad)
    float pour_angle;   // Local angle of low pouring lip
    float phase1, phase2, phase3; // Ruffle harmonic phases
    float fundamental_hz; // Modal acoustic fundamental (140 - 420 Hz)

    // Dynamic coupled spring-damper state driven by droplet torque & wind
    float bob_y;
    float vel_y;
    float dyn_tilt_x;
    float dyn_tilt_z;
    float omega_x;
    float omega_z;

    // Precomputed orthonormal frame & world center for fast GPU ray/particle queries
    Vec3 world_center;
    Vec3 axis_u;
    Vec3 axis_n;
    Vec3 axis_v;
};

struct LotusBudState {
    Vec3 base_pos;      // Base of receptacle in world space
    Vec3 stem_anchor;   // Submerged stem anchor point
    float radius;       // Outer petal envelope radius (~0.052m)
    float height;       // Bud height (~0.14m)
    float tilt_x;
    float tilt_z;
    float yaw;
    float open_amount;  // Half-opened bloom factor (0.42 - 0.55)

    // Dynamic wind sway
    Vec3 world_pos;
    Vec3 axis_u;
    Vec3 axis_n;
    Vec3 axis_v;
};

struct alignas(16) DropletParticle {
    Vec3 pos;
    float radius;       // meters (e.g. 0.00045 - 0.0042)
    Vec3 vel;
    int state;          // ParticleState
    int leaf_id;        // 0..6 when STATE_LEAF_BEAD, else -1
    float leaf_u;       // Local coordinates on leaf
    float leaf_v;
    float age;
    float squash;       // Oblate contact oscillation
    uint32_t rng;
    int active_band;    // 1 = foreground/midground cinema volume, 0 = upper mist reservoir
    int pad0;
};

struct GpuImpactEvent {
    int frame;
    float sub_frac;     // [0, 1) fractional frame time
    float world_x;
    float world_z;
    int impact_type;    // ImpactType
    int leaf_id;
    float radius_mm;
    float normal_vel;
    float radial_norm;  // [0, 1] normalized strike radius on leaf
};

struct CameraParams {
    Vec3 pos;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float tan_half_fov_y;
    float aspect;
    float focus_z;
    float coc_scale;    // Thin-lens 85mm f/1.4 Circle-of-Confusion scale (in pixels)
    float max_coc_px;
    Vec3 sun_dir;       // sun_elevation = 18 deg
};

// ============================================================================
// Botanical Lotus Leaf Parametric Surface & Venation Relief Evaluator
// ============================================================================
HD void update_leaf_frame(LotusLeafState& L) {
    L.world_center = L.base_pos + Vec3(0.0f, L.bob_y, 0.0f);
    float tx = L.base_tilt_x + L.dyn_tilt_x;
    float tz = L.base_tilt_z + L.dyn_tilt_z;

    float cy = cosf(L.yaw), sy = sinf(L.yaw);
    float cx = cosf(tx),    sx = sinf(tx);
    float cz = cosf(tz),    sz = sinf(tz);

    // Rotation: Ry(yaw) * Rx(tx) * Rz(tz)
    Vec3 u0(cz, sz, 0.0f);
    Vec3 n0(-sz, cz, 0.0f);
    Vec3 v0(0.0f, 0.0f, 1.0f);

    Vec3 u1(u0.x, u0.y * cx - u0.z * sx, u0.y * sx + u0.z * cx);
    Vec3 n1(n0.x, n0.y * cx - n0.z * sx, n0.y * sx + n0.z * cx);
    Vec3 v1(v0.x, v0.y * cx - v0.z * sx, v0.y * sx + v0.z * cx);

    L.axis_u = normalize(Vec3(u1.x * cy + u1.z * sy, u1.y, -u1.x * sy + u1.z * cy));
    L.axis_n = normalize(Vec3(n1.x * cy + n1.z * sy, n1.y, -n1.x * sy + n1.z * cy));
    L.axis_v = normalize(cross(L.axis_n, L.axis_u));
}

HD void update_bud_frame(LotusBudState& B, float sway_x, float sway_z) {
    float tx = B.tilt_x + sway_x;
    float tz = B.tilt_z + sway_z;
    float cy = cosf(B.yaw), sy = sinf(B.yaw);
    float cx = cosf(tx),    sx = sinf(tx);
    float cz = cosf(tz),    sz = sinf(tz);

    Vec3 u0(cz, sz, 0.0f);
    Vec3 n0(-sz, cz, 0.0f);

    Vec3 u1(u0.x, u0.y * cx - u0.z * sx, u0.y * sx + u0.z * cx);
    Vec3 n1(n0.x, n0.y * cx - n0.z * sx, n0.y * sx + n0.z * cx);

    B.axis_u = normalize(Vec3(u1.x * cy + u1.z * sy, u1.y, -u1.x * sy + u1.z * cy));
    B.axis_n = normalize(Vec3(n1.x * cy + n1.z * sy, n1.y, -n1.x * sy + n1.z * cy));
    B.axis_v = normalize(cross(B.axis_n, B.axis_u));
    B.world_pos = B.base_pos + Vec3(sway_z * -0.06f, 0.0f, sway_x * 0.06f);
}

struct LeafSample {
    float height;       // Local surface height h(u, v)
    float r_edge;       // Local outer ruffled boundary radius R(theta)
    float s_norm;       // r / r_edge (> 1 means outside leaf blade)
    float vein_mask;    // [0, 1] radial primary + bifurcating secondary vein intensity
    float umbilicus;    // [0, 1] central pale stem attachment disc
};

HD LeafSample eval_lotus_leaf_shape(const LotusLeafState& L, float u, float v) {
    LeafSample out;
    // Slight peltate offset of stem umbilicus
    float du = u + 0.015f * L.radius;
    float dv = v;
    float r = sqrtf(du * du + dv * dv + 1e-10f);
    float theta = atan2f(dv, du);
    float d_pour = wrap_angle(theta - L.pour_angle);

    // Organic ruffled perimeter with subtle pouring-lip indentation
    float perim_wave = 1.0f
        + 0.044f * sinf(7.0f * theta + L.phase1)
        + 0.027f * sinf(11.0f * theta + L.phase2)
        + 0.014f * cosf(17.0f * theta + L.phase3)
        - 0.036f * expf(-6.5f * d_pour * d_pour);

    out.r_edge = L.radius * perim_wave;
    float s = r / fmaxf(out.r_edge, 1e-5f);
    out.s_norm = s;
    float sc = fminf(s, 1.15f);

    // 1. Peltate cupped bowl with central stem funnel dimple
    float bowl = L.cup_depth * (powf(sc, 1.42f) - 0.17f * expf(-24.0f * sc * sc));

    // 2. Pouring trough saddle along pour_angle: lets heavy coalesced pearls
    // collect in the central hollow and spill over the low lip when heavy/tilted
    float pour_lobe = expf(-4.8f * d_pour * d_pour);
    float pour_channel = -0.68f * L.cup_depth * powf(sc, 1.65f) * pour_lobe;

    // 3. Multi-harmonic ruffled margin waves
    float ruffle_amp = L.radius * powf(clampf(sc, 0.0f, 1.05f), 2.15f);
    float ruffle = ruffle_amp * (
        0.048f * sinf(7.0f * theta + L.phase1) +
        0.029f * cosf(12.0f * theta + L.phase2) +
        0.015f * sinf(19.0f * theta + L.phase3)
    ) * (1.0f - 0.65f * pour_lobe);

    // 4. 21 Primary Radial Veins + 42 Bifurcating Secondary Veins
    float v1_phase = sinf(21.0f * theta + 0.25f * sinf(3.0f * theta + L.phase1));
    float primary_vein = powf(fmaxf(0.0f, 1.0f - fabsf(v1_phase)), 5.5f) * smoothstepf(0.04f, 0.14f, sc) * (1.0f - 0.35f * sc);

    float v2_phase = sinf(42.0f * theta + 0.25f * sinf(3.0f * theta + L.phase1));
    float secondary_vein = powf(fmaxf(0.0f, 1.0f - fabsf(v2_phase)), 6.0f) * smoothstepf(0.42f, 0.68f, sc) * (1.0f - 0.15f * sc);

    out.vein_mask = clampf(primary_vein + 0.65f * secondary_vein, 0.0f, 1.0f);
    out.umbilicus = expf(-340.0f * sc * sc);

    // Subtle micro-relief: veins form smooth raised structural ribs, Shallow grooves between veins guide droplets
    float vein_relief = L.radius * 0.0042f * out.vein_mask - L.radius * 0.0018f * (1.0f - fabsf(v1_phase)) * sc;

    // Slight graceful downward lip curl on very outer 4% edge
    float edge_curl = -L.radius * 0.012f * smoothstepf(0.93f, 1.0f, sc);

    out.height = bowl + pour_channel + ruffle + vein_relief + edge_curl;
    return out;
}

HD Vec3 eval_lotus_leaf_world_normal(const LotusLeafState& L, float u, float v) {
    const float eps = 0.0018f;
    float h_c = eval_lotus_leaf_shape(L, u, v).height;
    float h_u = eval_lotus_leaf_shape(L, u + eps, v).height;
    float h_v = eval_lotus_leaf_shape(L, u, v + eps).height;
    float dhdu = (h_u - h_c) / eps;
    float dhdv = (h_v - h_c) / eps;
    Vec3 n_loc = normalize(Vec3(-dhdu, 1.0f, -dhdv));
    return normalize(L.axis_u * n_loc.x + L.axis_n * n_loc.y + L.axis_v * n_loc.z);
}

// ============================================================================
// Botanical Sacred Lotus Bud (Nelumbo nucifera) Parametric SDF Evaluator
// 3 concentric whorls of cupped ovoid-acuminate ivory-pink petals + curved stem
// ============================================================================
struct BudMaterialInfo {
    float dist;
    int part_id;        // 0 = petal, 1 = sepal/base, 2 = stem, 3 = golden stamen core
    float petal_h_norm; // [0, 1] base-to-tip height along petal (ivory -> carmine-rose tip)
    float whorl_norm;   // [0, 1] outer-to-inner whorl
    float rib_mask;     // Longitudinal delicate petal venation
};

HD BudMaterialInfo eval_lotus_bud_sdf(const LotusBudState& B, const Vec3& p_world) {
    BudMaterialInfo res;
    res.dist = 1e5f;
    res.part_id = 0;
    res.petal_h_norm = 0.0f;
    res.whorl_norm = 0.5f;
    res.rib_mask = 0.0f;

    // Transform to bud local coordinates (y = [0..height] from receptacle base to apex)
    Vec3 d = p_world - B.world_pos;
    float lx = dot(d, B.axis_u);
    float ly = dot(d, B.axis_n);
    float lz = dot(d, B.axis_v);

    // 1. Curved cylindrical green lotus stem descending into pond
    {
        // Parametric stem from world_pos (t=1) down to stem_anchor (t=0)
        Vec3 p0 = B.stem_anchor;
        Vec3 p2 = B.world_pos + B.axis_n * 0.008f;
        Vec3 p1 = 0.5f * (p0 + p2) - B.axis_u * 0.025f;
        // Project coarsely in 5 segments
        float best_d = 1e5f;
        Vec3 prev = p0;
        for (int i = 1; i <= 5; ++i) {
            float t = float(i) * 0.2f;
            float omt = 1.0f - t;
            Vec3 cur = (omt * omt) * p0 + (2.0f * omt * t) * p1 + (t * t) * p2;
            Vec3 seg = cur - prev;
            float h_seg = clampf(dot(p_world - prev, seg) / fmaxf(dot(seg, seg), 1e-8f), 0.0f, 1.0f);
            float stem_r = lerpf(0.0068f, 0.0054f, (float(i - 1) + h_seg) * 0.2f);
            float dist_seg = length(p_world - (prev + seg * h_seg)) - stem_r;
            if (dist_seg < best_d) best_d = dist_seg;
            prev = cur;
        }
        res.dist = best_d;
        res.part_id = 2;
    }

    // Bounding check for bud blossom head
    if (ly < -0.025f || ly > B.height + 0.025f || (lx * lx + lz * lz) > (B.radius * B.radius * 3.4f)) {
        return res;
    }

    float r_cyl = sqrtf(lx * lx + lz * lz + 1e-10f);
    float theta = atan2f(lz, lx);

    // 2. Greenish-gold receptacle cup at very base of bud
    {
        float dy = ly - 0.008f;
        float r_rec = 0.018f + 0.35f * clampf(ly, 0.0f, 0.022f);
        float d_rec = sqrtf(fmaxf(r_cyl - r_rec, 0.0f) * fmaxf(r_cyl - r_rec, 0.0f) + dy * dy) - 0.007f;
        if (ly >= -0.01f && ly <= 0.024f && d_rec < res.dist) {
            res.dist = d_rec;
            res.part_id = 1;
            res.petal_h_norm = clampf(ly / 0.024f, 0.0f, 1.0f);
        }
    }

    // 3. Golden stamen heart nestled inside the half-opened petals
    {
        float sy = (ly - 0.28f * B.height) / (0.24f * B.height);
        float sr = r_cyl / (0.34f * B.radius);
        float d_core = (sqrtf(sr * sr + sy * sy) - 1.0f) * (0.24f * B.radius);
        if (d_core < res.dist) {
            res.dist = d_core;
            res.part_id = 3;
            res.petal_h_norm = clampf(sy * 0.5f + 0.5f, 0.0f, 1.0f);
        }
    }

    // 4. Three concentric sculpted petal whorls (half-opened Nelumbo nucifera)
    // Whorl 0: 5 outer slightly-parted ivory-emerald-blush petals
    // Whorl 1: 8 middle cupped pale ivory-pink petals with rose acuminate tips
    // Whorl 2: 8 inner gracefully clasping petals converging toward pointed apex
    const int counts[3]    = { 5, 8, 8 };
    const float phases[3]  = { 0.0f, 0.39f, 0.78f };
    const float h_scale[3] = { 0.80f, 0.96f, 1.00f };
    const float r_scale[3] = { 1.04f, 0.90f, 0.70f };
    const float flare[3]   = { 0.38f * B.open_amount, 0.18f * B.open_amount, -0.04f };

    for (int w = 0; w < 3; ++w) {
        float Hw = B.height * h_scale[w];
        float hn = ly / Hw;
        if (hn < -0.05f || hn > 1.06f) continue;
        float hc = clampf(hn, 0.0f, 1.0f);

        float n_petals = float(counts[w]);
        float ang_cell = n_petals * theta + phases[w];
        // Wrap to individual petal center [-pi, pi]
        float local_ang = wrap_angle(ang_cell); // [-pi, pi] across one petal period
        float u_petal = fabsf(local_ang) / 3.14159265f; // 0 at petal centerline, 1 at gap between petals

        // Petal width narrows toward acuminate pointed tip!
        // Overlapping imbricate petals: width > 1.0 in lower/mid body, tapering to 0 at tip (hc -> 1)
        float half_width = (1.28f - 0.20f * float(w)) * sinf(powf(hc, 0.62f) * 3.14159265f) + 0.04f * (1.0f - hc);
        if (half_width < 0.02f && hn > 0.98f) continue;

        // Cupped ovoid radial profile of the petal shell
        float base_bulb = sinf(powf(hc, 0.58f) * 2.88f + 0.16f);
        // Acuminate tip slight outward-upward reflex on outer/mid whorls
        float tip_flare = flare[w] * powf(hc, 2.4f);
        // Individual petal Spoon-cupping (centerline bows outward, margins curve inward)
        float cup_arch = 1.0f + 0.11f * cosf(clampf(u_petal / fmaxf(half_width, 0.15f), 0.f, 1.f) * 1.5708f);
        float r_shell = B.radius * r_scale[w] * (base_bulb + tip_flare) * cup_arch;

        float dr = r_cyl - r_shell;
        // Angular distance outside the acuminate petal edge
        float edge_excess = fmaxf(0.0f, u_petal - half_width) * (3.14159265f / n_petals) * fmaxf(r_shell, 0.008f);
        float dy_excess = 0.0f;
        if (hn < 0.0f) dy_excess = -ly;
        else if (hn > 1.0f) dy_excess = ly - Hw;

        // Delicate fleshy lotus petal thickness (thicker at ivory base ~2.2mm, razor-delicate ~0.7mm at rose tip)
        float half_thick = lerpf(0.0014f, 0.00048f, hc);
        float d_petal = sqrtf(dr * dr + edge_excess * edge_excess + dy_excess * dy_excess) - half_thick;

        if (d_petal < res.dist) {
            res.dist = d_petal;
            res.part_id = (w == 0 && hc < 0.35f) ? 1 : 0;
            res.petal_h_norm = hc;
            res.whorl_norm = float(w) * 0.5f;
            res.rib_mask = 0.5f + 0.5f * cosf(u_petal * 28.0f);
        }
    }
    return res;
}
