import os
import sys
import wave
import subprocess
import numpy as np
from PIL import Image


def read_wav_24bit_stereo(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    assert n_channels == 2, f"Expected stereo (2 ch), got {n_channels}"
    assert sampwidth == 3, f"Expected 24-bit WAV (3 bytes/sample), got {sampwidth}"
    assert framerate == 48000, f"Expected 48000 Hz, got {framerate}"

    # Unpack 24-bit signed little-endian PCM into float64 in [-1, 1]
    u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    i32 = (
        u8[:, 0].astype(np.int32)
        | (u8[:, 1].astype(np.int32) << 8)
        | (u8[:, 2].astype(np.int8).astype(np.int32) << 16)
    )
    samples = (i32.astype(np.float64) / 8388608.0).reshape(-1, 2)
    return samples, framerate


def audit_audio(wav_path: str) -> bool:
    print(f"\n=== AUDIO QA AUDIT: {wav_path} ===")
    samples, sr = read_wav_24bit_stereo(wav_path)
    n_frames = samples.shape[0]
    dur = n_frames / sr
    print(f"Format     : 48,000 Hz | 24-bit PCM | Stereo | Duration: {dur:.3f}s ({n_frames} frames)")

    max_abs = float(np.max(np.abs(samples)))
    peak_dbfs = 20.0 * np.log10(max(max_abs, 1e-12))
    has_nan = bool(not np.all(np.isfinite(samples)))

    s_0_4 = samples[: int(4.0 * sr)]
    s_4_14 = samples[int(4.0 * sr) : int(14.0 * sr)]
    s_14_20 = samples[int(14.0 * sr) :]

    rms_0_4 = float(np.sqrt(np.mean(s_0_4 ** 2)))
    rms_4_14 = float(np.sqrt(np.mean(s_4_14 ** 2)))
    rms_14_20 = float(np.sqrt(np.mean(s_14_20 ** 2)))

    # Check 100ms windows for silent dropouts
    win = int(0.10 * sr)
    n_wins = n_frames // win
    win_rms = [
        float(np.sqrt(np.mean(samples[w * win : (w + 1) * win] ** 2)))
        for w in range(n_wins)
    ]
    min_win_rms = min(win_rms)
    silent_dropouts = sum(1 for r in win_rms if r < 1e-5)

    # Spectral energy bands check (FFT on mono sum)
    mono = 0.5 * (samples[:, 0] + samples[:, 1])
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
    e_leaf = float(np.sum(spec[(freqs >= 140) & (freqs <= 600)] ** 2))
    e_minnaert = float(np.sum(spec[(freqs >= 1500) & (freqs <= 9500)] ** 2))
    stereo_diff = float(np.sqrt(np.mean((samples[:, 0] - samples[:, 1]) ** 2)))

    print(f"Peak Level : {max_abs:.5f} ({peak_dbfs:.2f} dBFS) [Limit < 0.98]")
    print(f"RMS 0-4s   : {rms_0_4:.5f} ({20*np.log10(max(rms_0_4,1e-12)):.2f} dBFS) [Sparse sunlit plinks & hero pearl]")
    print(f"RMS 4-14s  : {rms_4_14:.5f} ({20*np.log10(max(rms_4_14,1e-12)):.2f} dBFS) [Wind gust + 65,536 sun-shower peak]")
    print(f"RMS 14-20s : {rms_14_20:.5f} ({20*np.log10(max(rms_14_20,1e-12)):.2f} dBFS) [Tranquil taper & final pearl]")
    print(f"Min 100ms  : {min_win_rms:.6f} | Silent dropouts: {silent_dropouts}")
    print(f"Stereo W   : L-R RMS = {stereo_diff:.5f} | Leaf-band E={e_leaf:.2e} | Minnaert-band E={e_minnaert:.2e}")

    arc_ok = (rms_4_14 > rms_0_4 * 1.25) and (rms_4_14 > rms_14_20 * 1.25)
    clip_ok = (max_abs < 0.98) and (max_abs > 0.15)
    dropout_ok = (silent_dropouts == 0) and (not has_nan)

    print(f"Audio Gate : arc_ok={arc_ok}, clip_ok={clip_ok}, zero_dropouts={dropout_ok}")
    return arc_ok and clip_ok and dropout_ok


def convert_preview_ppms_to_png(out_dir: str):
    names = ["qa_t00_5s", "qa_t03_5s", "qa_t07_5s", "qa_t11_0s", "qa_t15_0s", "qa_t18_5s"]
    print("\n=== PREVIEW FRAME TELEMETRY ===")
    for nm in names:
        ppm = os.path.join(out_dir, nm + ".ppm")
        png = os.path.join(out_dir, nm + ".png")
        if os.path.exists(ppm):
            img = Image.open(ppm)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img.save(png)
            lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            p5, p50, p95, p999 = np.percentile(lum, [5, 50, 95, 99.9])
            mean_rgb = arr.mean(axis=(0, 1))
            print(
                f"{nm}.png ({img.size[0]}x{img.size[1]}): "
                f"Lum[p5={p5:.3f}, med={p50:.3f}, p95={p95:.3f}, p99.9={p999:.3f}] "
                f"MeanRGB=({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f})"
            )


def extract_mp4_qa_frames(mp4_path: str, out_dir: str):
    times = [0.5, 3.5, 7.5, 11.0, 15.0, 18.5]
    print(f"\n=== EXTRACTING 6 FULL-RES MP4 QA FRAMES FROM {mp4_path} ===")
    for t in times:
        out_png = os.path.join(out_dir, f"mp4_qa_t{t:04.1f}s.png".replace(".", "_"))
        # Fix extension back to .png
        out_png = out_png[:-4] + ".png" if out_png.endswith("_png") else out_png
        out_png = os.path.join(out_dir, f"mp4_qa_t{int(t):02d}_{int(round((t%1)*10))}s.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{t:.3f}", "-i", mp4_path,
                "-frames:v", "1", out_png
            ],
            check=True,
        )
        img = Image.open(out_png)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        p5, p50, p95, p999 = np.percentile(lum, [5, 50, 95, 99.9])
        print(f"t={t:4.1f}s -> {os.path.basename(out_png)} ({img.size[0]}x{img.size[1]}): Lum[p5={p5:.3f}, med={p50:.3f}, p95={p95:.3f}, p99.9={p999:.3f}]")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    convert_preview_ppms_to_png(target_dir)
    mp4_file = os.path.join(target_dir, "lotus_rain_65536_cuda.mp4")
    if os.path.exists(mp4_file):
        extract_mp4_qa_frames(mp4_file, target_dir)
    wav_file = os.path.join(target_dir, "lotus_rain_65536.wav")
    if os.path.exists(wav_file):
        ok = audit_audio(wav_file)
        if not ok:
            sys.exit(2)
