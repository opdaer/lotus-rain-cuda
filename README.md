# 65,536 Raindrops on a Sunlit Lotus Pond (`lotus-rain-cuda`)

[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Zero External Assets](https://img.shields.io/badge/External_Assets-0_(Pure_Math)-D99B52)](#highlights)
[![Built with Gemini 4 Carbon](https://img.shields.io/badge/Built_with-Gemini_4_Carbon-8AB4F8)](https://x.com/jianzhang1987)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![65,536 Raindrops on a Sunlit Lotus Pond — Pure CUDA & C++20 Macro Ray-Traced Frame](docs/hero_sunlit_storm.jpg)

A standalone **3,899-line pure CUDA (`.cu`) & C++20** macro optical ray-tracer, shallow-water wave & superhydrophobic droplet GPU physics solver, and coupled procedural physical audio synthesizer (`1920×1080`, `60 fps`, `20.0s`, `48 kHz` 24-bit stereo WAV muxed into H.264 `.mp4`).

Coded **100% from scratch in [Google Antigravity](https://idx.google.com/) with Gemini 4 Carbon** by **[@jianzhang1987](https://x.com/jianzhang1987)** — **zero game engines, zero 3D libraries (no OpenGL / Vulkan / OptiX / Three.js), zero external meshes or textures, and zero pre-recorded audio files**.

---

## 20-Second Visual & Acoustic Arc

| **Act I · Sunlit Ripples (`0–4s`)** | **Act II · Summer Gust & 65,536 Drops (`4–14s`)** | **Act III · Golden Afterglow (`14–20s`)** |
| :---: | :---: | :---: |
| ![Act I](docs/act1_calm_pond.jpg) | ![Act II](docs/act2_golden_gust.jpg) | ![Act III](docs/act3_afterglow.jpg) |
| Gentle afternoon droplets bead on waxy peltate lotus leaves over dark emerald water. | A warm `18°` backlit summer gust slants **65,536** raindrops across `f/1.4` 9-blade bokeh & Snell caustics. | The shower tapers into tranquil capillary rings, dripping silver pearls, and underwater bubble plinks. |

---

## Highlights

- **Macro `85mm f/1.4` Optical Ray-Tracer ([`src/renderer.cuh`](src/renderer.cuh) — `1,431` lines)**
  - **7 Sculpted Ruffled Lotus Leaves** (*Nelumbo nucifera*) with peltate cupped bowls, asymmetric pouring lips, 21 primary + 42 bifurcating radial veins, backlit golden-emerald subsurface transmission, and superhydrophobic micro-papillae velvet wax sheen.
  - **2 Half-Opened Sacred Lotus Buds** rendered via 3-whorl phyllotaxis SDF ray-marching with pale ivory-to-carmine-rose petal gradients.
  - **Refractive Water Pearls & Solar Caustics**: Smooth-metaball (`smin`) dielectric ray-tracing (`IOR = 1.333`) with magnified leaf-vein refraction and Snell-focused golden caustic hotspots cast onto leaf blades (`sun_elevation = 18°`).
  - **Hybrid Gather + Scatter Shallow Depth-of-Field**: 9-blade polygonal/circular specular bokeh scatter + shutter-integrated (`1/120s`) refractive rain streaks + 64-tap depth-aware Vogel spiral Circle-of-Confusion gather (`z_focus = 1.90 m`) + AgX botanical filmic tone-mapping.

- **Unified `65,536`-Particle CUDA Physics & `1024×1024` Dispersive Pond Grid ([`src/physics.cuh`](src/physics.cuh) — `737` lines)**
  - Simulates **65,536 (`2^16`) active water elements** across slanted wind-driven rain streaks, superhydrophobic leaf beads ($\theta_c \approx 150^\circ$ Cassie-Baxter roll-off), lock-free volume-conserving coalescence ($r_{\text{new}} = (r_1^3 + r_2^3)^{1/3}$), lip-spilling silver pearls, and parabolic Rayleigh-jet crown splashes.
  - Accumulates aerodynamic wind + droplet impact/weight torques ($\mathbf{r} \times \mathbf{F}$) per leaf to drive damped harmonic stem pitch, roll, and bobbing.
  - Solves a `1024×1024` shallow-water capillary-gravity wave PDE ($\omega^2 = gk + \frac{\sigma}{\rho}k^3$) using an isotropic 9-point Oono-Puri Laplacian + biharmonic stencil with sponge boundaries.

- **Coupled `48 kHz` 24-Bit Stereo Physical Audio Synthesis ([`src/audio_synth.cpp`](src/audio_synth.cpp) — `541` lines)**
  - **Built-in Zero-Dependency C++20 Modal & Minnaert Synthesizer**: Exports exact GPU impact events (`frame, sub_frac, world_x, world_z, impact_type, leaf_id, radius_mm, normal_vel`) and synthesizes Minnaert underwater bubble chirps ($f_0 \approx 3.26 / r_b$), 8-mode circular Bessel lotus-leaf membrane thuds, stereo ITD/ILD spatialization, and an 8-line Householder FDN open-air pond reflection tail directly in C++20.
  - **Optional Earshot Procedural Foil & Perceptual QA ([`synthesize_earshot_audio.py`](synthesize_earshot_audio.py))**: Reads exported `gpu_impact_events.bin` to synthesize dual-stage capillary-snap + wet-splash kernels and 24-channel Poisson pond rain fields with offline perceptual verification.

---

## Repository Structure

```text
lotus-rain-cuda/
├── CMakeLists.txt               # CMake 3.24+ CUDA & C++20 build configuration
├── build.bat                    # Windows MSVC + NVCC + Ninja release build helper
├── build_and_render.ps1         # End-to-end build, GPU diag, 1080p60 render & QA gate
├── docs/                        # Rendered 1080p/720p keyframes from the CUDA ray-tracer
├── src/                         # 3,899 lines of pure CUDA (.cu/.cuh) & C++20 source
│   ├── common.cuh               # Botanical SDFs, wind/rain arc fields & shared GPU structs (579 LOC)
│   ├── physics.cuh              # 65,536-droplet physics, coalescence & 1024x1024 wave PDE (737 LOC)
│   ├── renderer.cuh             # 4x RGSS ray-tracer, caustics & 9-blade gather/scatter DoF (1,431 LOC)
│   ├── main.cu                  # Simulation loop, FFmpeg pipe & GPU telemetry exporter (587 LOC)
│   ├── audio_synth.hpp          # Coupled C++20 acoustic synthesizer interface (24 LOC)
│   └── audio_synth.cpp          # Minnaert bubble, Bessel leaf modes & 8-line FDN reverb (541 LOC)
├── synthesize_earshot_audio.py  # Optional Earshot procedural audio synth & perceptual audit
├── render_badged_edition.py     # Optional 4x supersampled cinema title-badge overlay tool
└── verify_qa.py                 # Automated visual luminance & 24-bit WAV dynamics QA gate
```

---

## Prerequisites

- **NVIDIA GPU** with CUDA Toolkit `12.0+` (`nvcc` — runs even on a 4 GB `SM 6.1` GTX 1050 Ti!)
- **CMake** `3.24+` & **Ninja** (or Visual Studio / GCC / Clang C++20 toolchain)
- **FFmpeg** (`ffmpeg` and `ffprobe` in `PATH`, with `libx264` support)
- **Python 3.10+** with `numpy` and `Pillow` (optional, for `verify_qa.py` and `render_badged_edition.py`)

---

## Quick Start

### 1. Cross-Platform CMake Build & Render (Linux / Windows)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 8

# Fast 30-step GPU physics + 1080p render + NaN/bounds sanity check (< 1s)
./build/lotus_rain_sim --diag

# Full 1,200-frame 1920x1080 @ 60fps render + coupled 48kHz 24-bit WAV + H.264 MP4 mux
./build/lotus_rain_sim --out-dir .
```

### 2. One-Command Windows PowerShell Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File .\build_and_render.ps1
```

Useful flags:
- `-DiagOnly`: Compile and run the fast GPU physics/NaN diagnostic check.
- `-PreviewOnly`: Render 6 full-resolution keyframe previews (`0.5s, 3.5s, 7.5s, 11.0s, 15.0s, 18.5s`) + coupled 20s WAV in seconds.

### 3. Optional: Earshot Procedural Rain Audio & Cinema Title Badge

```bash
# Re-synthesize coupled 48kHz 24-bit stereo WAV from gpu_impact_events.bin
python synthesize_earshot_audio.py --events gpu_impact_events.bin --out lotus_rain_65536.wav

# Generate a separate 3.5s fade-in/fade-out top-left glass badge edition without modifying the original MP4
python render_badged_edition.py --input lotus_rain_65536_cuda.mp4 --output lotus_rain_65536_cuda_badged.mp4
```

---

## Author & License

- Created by **[@jianzhang1987](https://x.com/jianzhang1987)** using **Gemini 4 Carbon** in **Google Antigravity**.
- Released under the [MIT License](LICENSE).
