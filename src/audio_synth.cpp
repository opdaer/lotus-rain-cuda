#include "audio_synth.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace {

// Polynomial approximation to Bessel functions J_m(x) via series (accurate for x in [0, 12])
double bessel_jn_small(int m, double x) {
    double half_x = 0.5 * x;
    double term = 1.0;
    for (int k = 1; k <= m; ++k) {
        term *= half_x / double(k);
    }
    double sum = term;
    double hx2 = -half_x * half_x;
    for (int k = 1; k <= 18; ++k) {
        term *= hx2 / (double(k) * double(k + m));
        sum += term;
        if (std::abs(term) < 1e-12) break;
    }
    return sum;
}

// Cytomic / Simper Linear Trapezoidal State-Variable Filter (2-pole resonant bandpass & lowpass)
struct SvfFilter {
    float ic1eq = 0.0f;
    float ic2eq = 0.0f;

    void process(float v0, float cutoff_hz, float q, float sr, float& lp, float& bp, float& hp) {
        cutoff_hz = std::clamp(cutoff_hz, 20.0f, 0.45f * sr);
        float g = std::tan(3.14159265f * cutoff_hz / sr);
        float k = 1.0f / std::max(q, 0.5f);
        float a1 = 1.0f / (1.0f + g * (g + k));
        float a2 = g * a1;
        float a3 = g * a2;
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a3 * ic1eq + a2 * g * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        lp = v2;
        bp = v1;
        hp = v0 - k * v1 - v2;
    }
};

// 8-Line Householder Feedback Delay Network (FDN) for Open-Air Misty Lotus Pond Reverb
struct PondFdnReverb {
    static constexpr int N = 8;
    // Mutually prime delay lengths in samples @ 48kHz (~13 ms to ~53 ms)
    const std::array<int, N> delays = { 631, 797, 977, 1187, 1439, 1723, 2089, 2539 };
    std::array<std::vector<float>, N> buf;
    std::array<int, N> idx{};
    std::array<float, N> lp_state{};

    PondFdnReverb() {
        for (int i = 0; i < N; ++i) {
            buf[i].assign(delays[i], 0.0f);
            idx[i] = 0;
            lp_state[i] = 0.0f;
        }
    }

    void process(float in_l, float in_r, float& out_l, float& out_r) {
        std::array<float, N> rd;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            rd[i] = buf[i][idx[i]];
            // Warm air & water-surface high-frequency absorption
            lp_state[i] = lp_state[i] * 0.34f + rd[i] * 0.66f;
            rd[i] = lp_state[i];
            sum += rd[i];
        }
        // Householder reflection matrix H = I - (2/N) * 1*1^T
        const float h_scale = 2.0f / float(N);
        const float feedback = 0.78f; // ~0.82s lush natural outdoor pond decay

        out_l = 0.0f;
        out_r = 0.0f;
        for (int i = 0; i < N; ++i) {
            float mixed = rd[i] - h_scale * sum;
            float inject = (i & 1) ? in_r : in_l;
            float sign = (i % 4 < 2) ? 1.0f : -1.0f;
            buf[i][idx[i]] = inject * (0.25f * sign) + mixed * feedback;
            idx[i] = (idx[i] + 1) % delays[i];

            if (i & 1) out_r += rd[i];
            else       out_l += rd[i];
        }
        out_l *= 0.28f;
        out_r *= 0.28f;
    }
};

void add_minnaert_bubble_voice(
    std::vector<float>& mix_l,
    std::vector<float>& mix_r,
    int total_samples,
    int sample_rate,
    double start_time_sec,
    float bubble_radius_mm,
    float chirp_frac,
    float amplitude,
    float pan_angle_01, // [0 = hard left, 0.5 = center, 1 = hard right]
    float dist_z
) {
    int start_idx = int(std::llround(start_time_sec * double(sample_rate)));
    if (start_idx < 0 || start_idx >= total_samples) return;

    // Minnaert resonance formula: f0 = 3.26 / r_b (in meters)
    float rb_m = std::clamp(bubble_radius_mm, 0.30f, 2.35f) * 1e-3f;
    float f0 = std::clamp(3.26f / rb_m, 1380.0f, 9600.0f);

    // Physical bubble damping factor delta = rad + thermal + viscous
    float delta = 0.0135f + 0.00055f * std::sqrt(f0);
    float tau_decay = 1.0f / (3.14159265f * f0 * delta); // ~6 ms to 42 ms
    tau_decay = std::clamp(tau_decay, 0.005f, 0.055f);
    float tau_chirp = tau_decay * 0.58f;

    // Distance attenuation & Haas interaural time difference (ITD up to 0.38 ms)
    float dist_atten = 1.0f / (1.0f + 0.28f * std::max(0.0f, dist_z - 1.2f));
    float gain_l = std::cos(pan_angle_01 * 1.5707963f) * amplitude * dist_atten;
    float gain_r = std::sin(pan_angle_01 * 1.5707963f) * amplitude * dist_atten;

    float pan_signed = (pan_angle_01 - 0.5f) * 2.0f;
    int itd_l = (pan_signed > 0.0f) ? int(pan_signed * 18.0f) : 0;
    int itd_r = (pan_signed < 0.0f) ? int(-pan_signed * 18.0f) : 0;

    int dur_samples = std::min(int(tau_decay * 6.5f * float(sample_rate)), total_samples - start_idx - 20);
    if (dur_samples <= 8) return;

    double phase = 0.0;
    const double inv_sr = 1.0 / double(sample_rate);
    for (int i = 0; i < dur_samples; ++i) {
        float t = float(i) * float(inv_sr);
        // Characteristic 10%-18% upward pitch chirp as bubble nears surface impedance boundary
        float f_inst = f0 * (1.0f + chirp_frac * (1.0f - std::exp(-t / tau_chirp)));
        phase += 6.283185307179586 * double(f_inst) * inv_sr;

        float env_atk = 1.0f - std::exp(-t / 0.00035f);
        float env_dec = std::exp(-t / tau_decay);
        // Pure crystalline sinusoid + subtle 2nd liquid cavity harmonic
        float sig = (float(std::sin(phase)) + 0.08f * float(std::sin(2.0 * phase)) * std::exp(-t / (tau_decay * 0.4f)))
                    * env_atk * env_dec;

        int il = start_idx + i + itd_l;
        int ir = start_idx + i + itd_r;
        if (il < total_samples) mix_l[il] += sig * gain_l;
        if (ir < total_samples) mix_r[ir] += sig * gain_r;
    }
}

} // namespace

AudioDiagnostics synthesize_coupled_pond_audio_wav(
    const std::vector<GpuImpactEvent>& events,
    const LotusLeafState leaves[NUM_LEAVES],
    const std::string& wav_output_path,
    float duration_sec,
    int sample_rate
) {
    AudioDiagnostics diag{};
    const int total_samples = int(std::llround(double(duration_sec) * double(sample_rate)));

    std::vector<float> dry_l(total_samples, 0.0f);
    std::vector<float> dry_r(total_samples, 0.0f);

    // Sort GPU events chronologically for clean deterministic synthesis
    std::vector<GpuImpactEvent> sorted = events;
    std::sort(sorted.begin(), sorted.end(), [](const GpuImpactEvent& a, const GpuImpactEvent& b) {
        float ta = float(a.frame) + a.sub_frac;
        float tb = float(b.frame) + b.sub_frac;
        return ta < tb;
    });

    // Compute per-100ms event density so dense sun-shower 65,536-droplet chorus sums organically
    const int num_bins = int(duration_sec * 10.0f) + 4;
    std::vector<int> bin_counts(num_bins, 0);
    for (const auto& ev : sorted) {
        double t = (double(ev.frame) + double(ev.sub_frac)) / double(FPS);
        int b = std::clamp(int(t * 10.0), 0, num_bins - 1);
        bin_counts[b]++;
    }

    // Circular elastic membrane Bessel roots alpha_mn (m, n):
    // (0,1)=2.4048, (1,1)=3.8317, (2,1)=5.1356, (0,2)=5.5201,
    // (3,1)=6.3802, (1,2)=7.0156, (4,1)=7.5883, (2,2)=8.4172
    constexpr int NUM_MODES = 8;
    const int mode_m[NUM_MODES]      = { 0, 1, 2, 0, 3, 1, 4, 2 };
    const double alpha_mn[NUM_MODES] = { 2.4048, 3.8317, 5.1356, 5.5201, 6.3802, 7.0156, 7.5883, 8.4172 };
    const float mode_ratios[NUM_MODES] = {
        1.0000f, 1.5933f, 2.1356f, 2.2954f, 2.6531f, 2.9173f, 3.1555f, 3.5002f
    };

    uint32_t audio_rng = 0x9E3779B9u;

    for (size_t idx = 0; idx < sorted.size(); ++idx) {
        const auto& ev = sorted[idx];
        double t0 = (double(ev.frame) + double(ev.sub_frac)) / double(FPS);
        if (t0 < 0.0 || t0 >= double(duration_sec)) continue;

        int b = std::clamp(int(t0 * 10.0), 0, num_bins - 1);
        float local_rate = float(bin_counts[b]) * 10.0f; // events/sec
        // Natural acoustic chorus scaling: sparse drops stay crystal-clear & intimate;
        // thousands of storm impacts blend into a shimmering rain curtain without overload
        float crowd_atten = 1.0f / std::pow(1.0f + local_rate / 140.0f, 0.36f);

        // Stereo azimuth from camera perspective at (0, 0.36, 0)
        float az = std::atan2(ev.world_x, std::max(0.4f, ev.world_z));
        float pan01 = std::clamp(0.5f + 0.85f * az, 0.06f, 0.94f);

        if (ev.impact_type == IMPACT_HERO_PEARL) {
            diag.pearl_events++;
            // Heavy merged pearl spilling off lotus leaf into pond:
            // Multi-bubble resonant cascade + soft liquid crater splash!
            float u1 = rand01(audio_rng);
            float u2 = rand01(audio_rng);
            float u3 = rand01(audio_rng);
            float base_amp = 0.20f * std::clamp(ev.radius_mm / 3.0f, 0.7f, 1.45f);

            // 1. Primary deep cavity Minnaert bubble (1.5 kHz - 2.1 kHz, 16% upward chirp)
            add_minnaert_bubble_voice(
                dry_l, dry_r, total_samples, sample_rate,
                t0 + 0.002, 1.65f + 0.52f * u1, 0.14f + 0.04f * u2,
                base_amp, pan01, ev.world_z
            );
            // 2. Secondary Rayleigh crown pinch bubble (+21ms, 2.6 - 4.1 kHz, 13% chirp)
            add_minnaert_bubble_voice(
                dry_l, dry_r, total_samples, sample_rate,
                t0 + 0.019 + 0.008 * u2, 0.82f + 0.42f * u3, 0.12f + 0.04f * u1,
                base_amp * 0.68f, pan01, ev.world_z
            );
            // 3. Satellite crystal droplet plink (+52ms, 4.8 - 7.8 kHz, 11% chirp)
            add_minnaert_bubble_voice(
                dry_l, dry_r, total_samples, sample_rate,
                t0 + 0.048 + 0.015 * u3, 0.42f + 0.26f * u1, 0.10f + 0.04f * u3,
                base_amp * 0.42f, pan01, ev.world_z
            );

            // Soft water-entry splash transient (14ms filtered burst)
            int s0 = int(t0 * sample_rate);
            int splash_len = std::min(int(0.018f * sample_rate), total_samples - s0);
            SvfFilter splash_svf{};
            float gl = std::cos(pan01 * 1.5707963f) * base_amp * 0.42f;
            float gr = std::sin(pan01 * 1.5707963f) * base_amp * 0.42f;
            for (int i = 0; i < splash_len; ++i) {
                float ts = float(i) / float(sample_rate);
                float wn = rand01(audio_rng) * 2.0f - 1.0f;
                float lp, bp, hp;
                splash_svf.process(wn, 1950.0f, 1.4f, float(sample_rate), lp, bp, hp);
                float env = std::exp(-ts / 0.0045f) * (1.0f - std::exp(-ts / 0.0004f));
                dry_l[s0 + i] += bp * env * gl;
                dry_r[s0 + i] += bp * env * gr;
            }
        }
        else if (ev.impact_type == IMPACT_POND_DROP) {
            diag.pond_events++;
            float u_prob  = rand01(audio_rng);
            float u_rad   = rand01(audio_rng);
            float u_chirp = rand01(audio_rng);

            float vel_norm = std::clamp(ev.normal_vel / 6.0f, 0.25f, 1.5f);
            float r_scale  = std::clamp(ev.radius_mm / 1.15f, 0.35f, 2.2f);
            float amp = 0.068f * r_scale * vel_norm * crowd_atten;

            // Delicate water-surface capillary micro-click (1.1 ms high-passed transient)
            int s0 = int(t0 * sample_rate);
            int click_len = std::min(64, total_samples - s0);
            float dist_atten = 1.0f / (1.0f + 0.30f * std::max(0.0f, ev.world_z - 1.2f));
            float gl = std::cos(pan01 * 1.5707963f) * amp * 0.36f * dist_atten;
            float gr = std::sin(pan01 * 1.5707963f) * amp * 0.36f * dist_atten;
            float prev_n = 0.0f;
            for (int i = 0; i < click_len; ++i) {
                float n = rand01(audio_rng) * 2.0f - 1.0f;
                float hp = n - 0.82f * prev_n;
                prev_n = n;
                float env = std::exp(-float(i) / 14.0f);
                dry_l[s0 + i] += hp * env * gl;
                dry_r[s0 + i] += hp * env * gr;
            }

            // Probabilistically trap an underwater Minnaert micro-bubble in [0.3 mm, 2.2 mm]
            if (u_prob < 0.62f) {
                // Larger faster drops bias toward slightly deeper bubbles; fine drops -> bright crystal plinks
                float rb_mm = std::clamp(
                    0.32f + 1.85f * std::pow(u_rad, 1.55f) * (0.55f + 0.45f * std::min(r_scale, 1.4f)),
                    0.32f, 2.18f
                );
                float chirp = 0.10f + 0.08f * u_chirp; // 10% - 18% upward pitch chirp!
                add_minnaert_bubble_voice(
                    dry_l, dry_r, total_samples, sample_rate,
                    t0 + 0.0008, rb_mm, chirp, amp * 1.22f, pan01, ev.world_z
                );
            }
        }
        else if (ev.impact_type == IMPACT_LOTUS_LEAF) {
            diag.leaf_events++;
            int lid = std::clamp(ev.leaf_id, 0, NUM_LEAVES - 1);
            float f_fund = std::clamp(leaves[lid].fundamental_hz, 140.0f, 420.0f);
            double rho = std::clamp(double(ev.radial_norm), 0.02, 0.97);

            // Precompute exact Bessel modal excitation weights |J_m(alpha_mn * rho)|
            float weights[NUM_MODES];
            float w_sum = 1e-6f;
            for (int m = 0; m < NUM_MODES; ++m) {
                float jval = float(std::abs(bessel_jn_small(mode_m[m], alpha_mn[m] * rho)));
                // Higher order modes naturally roll off slightly due to finite droplet contact patch
                weights[m] = jval / (1.0f + 0.22f * float(m));
                w_sum += weights[m];
            }
            for (int m = 0; m < NUM_MODES; ++m) weights[m] /= w_sum;

            float vel_norm = std::clamp(ev.normal_vel / 6.0f, 0.25f, 1.5f);
            float r_scale  = std::clamp(ev.radius_mm / 1.1f, 0.4f, 2.0f);
            float dist_atten = 1.0f / (1.0f + 0.26f * std::max(0.0f, ev.world_z - 1.2f));
            float base_amp = 0.085f * r_scale * vel_norm * crowd_atten * dist_atten;

            float gl = std::cos(pan01 * 1.5707963f) * base_amp;
            float gr = std::sin(pan01 * 1.5707963f) * base_amp;

            int s0 = int(t0 * sample_rate);
            // Damped elastic lotus blade thud lasts ~45-75 ms
            int dur = std::min(int(0.068f * sample_rate), total_samples - s0);
            const float inv_sr = 1.0f / float(sample_rate);
            float prev_n = 0.0f;

            for (int i = 0; i < dur; ++i) {
                float ts = float(i) * inv_sr;
                float env_atk = 1.0f - std::exp(-ts / 0.0005f);

                // Sum 8 inharmonic Bessel circular-membrane modes with slight downward pitch relaxation
                float pitch_bend = 1.0f + 0.035f * std::exp(-ts / 0.006f);
                float modal_sum = 0.0f;
                for (int m = 0; m < NUM_MODES; ++m) {
                    float fm = f_fund * mode_ratios[m] * pitch_bend;
                    // Viscoelastic leaf tissue damps higher modes faster -> warm organic thud
                    float tau_m = 0.022f / (1.0f + 0.42f * float(m));
                    modal_sum += weights[m] * std::sin(6.2831853f * fm * ts) * std::exp(-ts / tau_m);
                }

                // Short high-passed waxy cuticle transient click ("patter") in first 2.2 ms
                float click = 0.0f;
                if (i < 110) {
                    float wn = rand01(audio_rng) * 2.0f - 1.0f;
                    float hp = wn - 0.86f * prev_n;
                    prev_n = wn;
                    click = 0.45f * hp * std::exp(-ts / 0.00075f);
                }

                float sig = (modal_sum * env_atk + click);
                dry_l[s0 + i] += sig * gl;
                dry_r[s0 + i] += sig * gr;
            }
        }
    }

    // ========================================================================
    // Continuous Procedural Summer Wind Gust Whoosh & Leaf Rustle (4.0s - 14.0s)
    // Plus ultra-subtle natural open-air pond air bed (zero silent dropouts!)
    // ========================================================================
    SvfFilter breeze_svf_l{}, breeze_svf_r{};
    SvfFilter rustle_svf_l{}, rustle_svf_r{};
    float brown_l = 0.0f, brown_r = 0.0f;

    for (int i = 0; i < total_samples; ++i) {
        float t = float(i) / float(sample_rate);
        // Smooth summer gust envelope peaking across [4.0s, 14.0s]
        float gust_env = smoothstepf(3.8f, 6.4f, t) * (1.0f - smoothstepf(12.2f, 15.0f, t));
        // Subtle omnipresent warm open-air evening pond air floor (~ -56 dBFS -> zero dead digital silence)
        float air_floor = 0.0018f + 0.0005f * std::sin(t * 0.9f);

        float wl = rand01(audio_rng) * 2.0f - 1.0f;
        float wr = rand01(audio_rng) * 2.0f - 1.0f;
        brown_l = std::clamp(brown_l * 0.985f + wl * 0.12f, -1.0f, 1.0f);
        brown_r = std::clamp(brown_r * 0.985f + wr * 0.12f, -1.0f, 1.0f);

        // Resonant bandpass frequency & stereo pan sweep left-to-right during gust!
        float sweep_prog = std::clamp((t - 4.0f) / 10.0f, 0.0f, 1.0f);
        float mod1 = 0.5f + 0.5f * std::sin(t * 1.45f) * std::cos(t * 0.62f + 0.8f);
        float cutoff_l = 250.0f + 440.0f * gust_env * mod1 + 90.0f * (1.0f - sweep_prog);
        float cutoff_r = 250.0f + 460.0f * gust_env * mod1 + 130.0f * sweep_prog;

        float lp_l, bp_l, hp_l;
        float lp_r, bp_r, hp_r;
        breeze_svf_l.process(brown_l * 0.6f + wl * 0.4f, cutoff_l, 2.6f, float(sample_rate), lp_l, bp_l, hp_l);
        breeze_svf_r.process(brown_r * 0.6f + wr * 0.4f, cutoff_r, 2.6f, float(sample_rate), lp_r, bp_r, hp_r);

        // Delicate high lotus-blade rustle shimmer
        float rlp_l, rbp_l, rhp_l;
        float rlp_r, rbp_r, rhp_r;
        rustle_svf_l.process(wl, 2600.0f + 900.0f * mod1, 1.8f, float(sample_rate), rlp_l, rbp_l, rhp_l);
        rustle_svf_r.process(wr, 2750.0f + 900.0f * mod1, 1.8f, float(sample_rate), rlp_r, rbp_r, rhp_r);

        float pan_gust_l = std::cos(std::clamp(0.25f + 0.50f * sweep_prog, 0.15f, 0.85f) * 1.5707963f);
        float pan_gust_r = std::sin(std::clamp(0.25f + 0.50f * sweep_prog, 0.15f, 0.85f) * 1.5707963f);

        float wind_amp = air_floor + 0.068f * gust_env * (0.72f + 0.28f * mod1);
        dry_l[i] += wind_amp * (bp_l * 1.25f * pan_gust_l + rbp_l * 0.16f * gust_env);
        dry_r[i] += wind_amp * (bp_r * 1.25f * pan_gust_r + rbp_r * 0.16f * gust_env);
    }

    // ========================================================================
    // Early Pond-Surface Specular Reflection (2.3 ms) + 8-Line FDN Air Reverb
    // ========================================================================
    std::vector<float> final_l(total_samples, 0.0f);
    std::vector<float> final_r(total_samples, 0.0f);
    PondFdnReverb reverb;
    const int early_delay = 112; // 2.33 ms @ 48 kHz (low macro mic 40cm above water)

    float raw_peak = 1e-6f;
    for (int i = 0; i < total_samples; ++i) {
        float el = (i >= early_delay) ? dry_l[i - early_delay] * 0.18f : 0.0f;
        float er = (i >= early_delay) ? dry_r[i - early_delay] * 0.18f : 0.0f;
        float rev_l, rev_r;
        reverb.process(dry_l[i] + el, dry_r[i] + er, rev_l, rev_r);

        float sl = dry_l[i] * 0.84f + el + rev_l * 0.22f;
        float sr = dry_r[i] * 0.84f + er + rev_r * 0.22f;

        // Smooth 40ms fade-in & 180ms fade-out to prevent edge clicks
        float edge_env = smoothstepf(0.0f, 0.04f, float(i) / float(sample_rate))
                       * (1.0f - smoothstepf(duration_sec - 0.18f, duration_sec, float(i) / float(sample_rate)));
        // Keep tiny natural floor at very end so no digital zero-dropout occurs
        edge_env = std::max(edge_env, 0.02f);

        final_l[i] = sl * edge_env;
        final_r[i] = sr * edge_env;
        raw_peak = std::max(raw_peak, std::max(std::abs(final_l[i]), std::abs(final_r[i])));
    }

    // ========================================================================
    // Normalize Dynamics & Apply Smooth Lookahead Soft-Limiter (-1.0 dBFS ceiling)
    // -1.0 dBFS = 10^(-1/20) = 0.89125 (< 0.98 zero-clipping gate!)
    // ========================================================================
    const float ceiling = 0.885f; // -1.06 dBFS ceiling
    float target_gain = (raw_peak > 1e-5f) ? std::min(ceiling * 1.18f / raw_peak, 3.2f) : 1.0f;

    float lim_env = 0.0f;
    double sum_sq_0_4 = 0.0, sum_sq_4_14 = 0.0, sum_sq_14_20 = 0.0;
    int cnt_0_4 = 0, cnt_4_14 = 0, cnt_14_20 = 0;

    for (int i = 0; i < total_samples; ++i) {
        float l = final_l[i] * target_gain;
        float r = final_r[i] * target_gain;

        // Soft-knee saturation + envelope limiter strictly bounded below ceiling (0.885)
        auto soft_clip = [&](float x) -> float {
            float kn = x / ceiling;
            return ceiling * std::tanh(kn);
        };
        l = soft_clip(l);
        r = soft_clip(r);

        float pk = std::max(std::abs(l), std::abs(r));
        if (pk > lim_env) lim_env = pk;
        else lim_env = lim_env * 0.9992f + pk * 0.0008f;

        if (lim_env > ceiling) {
            float g = ceiling / lim_env;
            l *= g;
            r *= g;
        }

        if (!std::isfinite(l) || !std::isfinite(r)) {
            diag.has_nan = true;
            l = 0.0f;
            r = 0.0f;
        }

        final_l[i] = l;
        final_r[i] = r;
        diag.max_abs_sample = std::max(diag.max_abs_sample, std::max(std::abs(l), std::abs(r)));

        double e = 0.5 * (double(l) * double(l) + double(r) * double(r));
        float t = float(i) / float(sample_rate);
        if (t < 4.0f) { sum_sq_0_4 += e; cnt_0_4++; }
        else if (t < 14.0f) { sum_sq_4_14 += e; cnt_4_14++; }
        else { sum_sq_14_20 += e; cnt_14_20++; }
    }

    diag.rms_0_4s   = float(std::sqrt(sum_sq_0_4 / std::max(1, cnt_0_4)));
    diag.rms_4_14s  = float(std::sqrt(sum_sq_4_14 / std::max(1, cnt_4_14)));
    diag.rms_14_20s = float(std::sqrt(sum_sq_14_20 / std::max(1, cnt_14_20)));

    // ========================================================================
    // Write Standard 48,000 Hz 24-Bit Stereo PCM WAV (pcm_s24le)
    // ========================================================================
    std::ofstream out(wav_output_path, std::ios::binary);
    if (!out) {
        std::cerr << "[AudioSynth] Failed to open WAV file for writing: " << wav_output_path << "\n";
        return diag;
    }

    const uint16_t num_channels = 2;
    const uint16_t bits_per_sample = 24;
    const uint16_t block_align = num_channels * (bits_per_sample / 8); // 6 bytes
    const uint32_t byte_rate = uint32_t(sample_rate) * block_align;
    const uint32_t data_bytes = uint32_t(total_samples) * block_align;
    const uint32_t riff_size = 36u + data_bytes;

    auto write_u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    auto write_u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };

    out.write("RIFF", 4);
    write_u32(riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(16u);
    write_u16(1u); // PCM integer
    write_u16(num_channels);
    write_u32(uint32_t(sample_rate));
    write_u32(byte_rate);
    write_u16(block_align);
    write_u16(bits_per_sample);
    out.write("data", 4);
    write_u32(data_bytes);

    std::vector<uint8_t> pcm_buf(data_bytes);
    size_t ptr = 0;
    for (int i = 0; i < total_samples; ++i) {
        float chans[2] = { final_l[i], final_r[i] };
        for (int c = 0; c < 2; ++c) {
            double scaled = std::clamp(double(chans[c]), -0.999999, 0.999999) * 8388607.0;
            int32_t iv = int32_t(std::llround(scaled));
            pcm_buf[ptr + 0] = uint8_t(iv & 0xFF);
            pcm_buf[ptr + 1] = uint8_t((iv >> 8) & 0xFF);
            pcm_buf[ptr + 2] = uint8_t((iv >> 16) & 0xFF);
            ptr += 3;
        }
    }
    out.write(reinterpret_cast<const char*>(pcm_buf.data()), pcm_buf.size());
    out.close();

    return diag;
}
