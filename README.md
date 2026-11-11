# 65,536 Raindrops on a Sunlit Lotus Pond (`lotus-rain-cuda`)

A standalone **CUDA 20 (`.cu`) & C++20** macro optical ray-tracer, shallow-water wave & superhydrophobic droplet GPU physics engine, and coupled procedural physical audio synthesizer (`1920x1080`, `60 fps`, `20.0s`, `48 kHz` 24-bit stereo WAV muxed into H.264 `.mp4`).

Built **100% from first principles** — **zero game engines, zero external 3D rendering libraries (no OpenGL/Vulkan/OptiX wrappers), zero external textures/meshes, and zero pre-recorded audio samples**.

---

## Highlights

- **Macro `85mm f/1.4` Optical Ray-Tracer (`src/renderer.cuh`)**
  - **7 Sculpted Ruffled Lotus Leaves** (*Nelumbo nucifera*) with peltate cupped bowls, asymmetric pouring lips, 21 primary + 42 bifurcating radial veins, backlit golden-emerald subsurface transmission, and superhydrophobic micro-papillae velvet wax sheen.
  - **2 Half-Opened Sacred Lotus Buds** rendered via 3-whorl phyllotaxis SDF ray-marching with pale ivory-to-carmine-rose petal gradients.
  - **Refractive Water Pearls & Solar Caustics**: Smooth-metaball (`smin`) dielectric ray-tracing (`IOR = 1.333`) with magnified leaf-vein refraction and Snell-focused golden caustic hotspots cast onto leaf blades (`sun_elevation = 18°`).
  - **Hybrid Gather + Scatter Shallow Depth-of-Field**: 9-blade polygonal/circular specular bokeh scatter + shutter-integrated (`1/120s`) refractive rain streaks + 64-tap depth-aware Vogel spiral Circle-of-Confusion gather (`z_focus = 1.90 m`) + AgX botanical filmic tone-mapping.

- **Unified `65,536`-Particle CUDA Physics & `1024x1024` Dispersive Pond Grid (`src/physics.cuh`)**
  - Simulates **65,536 (`2^16`) active water elements** across slanted wind-driven rain streaks, superhydrophobic leaf beads ($\theta_c \approx 150^\circ$ Cassie-Baxter roll-off), lock-free volume-conserving coalescence ($r_{\text{new}} = (r_1^3 + r_2^3)^{1/3}$), lip-spilling silver pearls, and parabolic Rayleigh-jet crown splashes.
  - Accumulates aerodynamic wind + droplet impact/weight torques ($\mathbf{r} \times \mathbf{F}$) per leaf to drive damped harmonic stem pitch, roll, and bobbing.
  - Solves a `1024x1024` shallow-water capillary-gravity wave PDE ($\omega^2 = gk + \frac{\sigma}{\rho}k^3$) using an isotropic 9-point Oono-Puri Laplacian + biharmonic stencil with sponge boundaries.

- **Coupled `48 kHz` 24-Bit Stereo Physical Audio Synthesis**
  - **Built-in C++20 Modal & Minnaert Synthesizer (`src/audio_synth.cpp`)**: Synthesizes GPU-exported impact events (`frame, world_x, world_z, impact_type, radius_mm, normal_vel`) from first principles with zero external dependencies.
  - **Optional Earshot Procedural Foil & Perceptual QA (`synthesize_earshot_audio.py`)**: Reads exported `gpu_impact_events.bin` to synthesize dual-stage capillary-snap + wet-splash kernels, 24-channel Poisson pond rain fields, and open-air early/late pond reflections via `earshot.forge`, with offline CLAP/AST perceptual verification.

---

## Repository Structure

```text
lotus-rain-cuda/
├── CMakeLists.txt               # CMake 3.24+ CUDA 20 & C++20 build configuration
├── build.bat                    # Windows MSVC + NVCC + Ninja release build helper
├── build_and_render.ps1         # End-to-end build, GPU diag, 1080p60 render & QA gate
├── src/
│   ├── common.cuh               # Botanical SDFs, wind/rain arc fields & shared GPU structs
│   ├── physics.cuh              # 65,536-droplet physics, coalescence & 1024x1024 wave PDE
│   ├── renderer.cuh             # 4x RGSS ray-tracer, caustics & 9-blade gather/scatter DoF
│   ├── main.cu                  # Simulation loop, FFmpeg pipe & GPU telemetry exporter
│   ├── audio_synth.hpp          # Coupled C++20 acoustic synthesizer interface
│   └── audio_synth.cpp          # Minnaert bubble, Bessel leaf modes & 8-line FDN reverb
├── synthesize_earshot_audio.py  # Optional Earshot procedural audio synth & CLAP/AST audit
├── render_badged_edition.py     # Optional 4x supersampled cinema title-badge overlay tool
└── verify_qa.py                 # Automated visual luminance & 24-bit WAV dynamics QA gate
```

---

## Prerequisites

- **NVIDIA GPU** with CUDA Toolkit `12.0+` (`nvcc`)
- **CMake** `3.24+` & **Ninja** (or Visual Studio / GCC / Clang C++20 toolchain)
- **FFmpeg** (`ffmpeg` and `ffprobe` in `PATH`, with `libx264` support)
- **Python 3.10+** with `numpy` and `Pillow` (for `verify_qa.py` and `render_badged_edition.py`)

---

## Quick Start

### 1. Build & Render Full `1920x1080 @ 60fps` Sequence (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\build_and_render.ps1
```

Useful flags:
- `-DiagOnly`: Compile and run a fast 30-frame GPU physics/NaN diagnostic check.
- `-PreviewOnly`: Render 6 full-resolution keyframe previews (`0.5s, 3.5s, 7.5s, 11.0s, 15.0s, 18.5s`) + coupled 20s WAV.

### 2. Cross-Platform CMake Build (Linux / Windows)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 8

# Fast GPU sanity check
./build/lotus_rain_sim --diag

# Full 1,200-frame 1080p60 render + 48kHz 24-bit WAV + H.264 MP4 mux
./build/lotus_rain_sim --out-dir .
```

### 3. Optional: Earshot Procedural Rain Audio & Cinema Title Badge

```bash
# Re-synthesize coupled 48kHz 24-bit stereo WAV from gpu_impact_events.bin using Earshot
python synthesize_earshot_audio.py --events gpu_impact_events.bin --out lotus_rain_65536.wav --audit

# Generate a separate 3.5s fade-in/fade-out top-left glass badge edition without touching the original MP4
python render_badged_edition.py --input lotus_rain_65536_cuda.mp4 --output lotus_rain_65536_cuda_badged.mp4
```

---

## License

Released under the [MIT License](LICENSE).
