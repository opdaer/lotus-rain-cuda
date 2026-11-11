r"""
Generate the 3.5s Cinema-Grade Top-Left Title Badge Edition of
"65,536 Raindrops on a Sunlit Lotus Pond" without modifying the original MP4.

Specs:
  1. Pillow 4x supersampled anti-aliased dark glass rounded card at (x=36px, y=32px):
     - Backdrop : rgba(12, 16, 14, 196), corner radius 16px
     - Border   : 1px rgba(255, 255, 255, 32) subtle glass highlight rim
     - Line 1   : Bold 36px, #8AB4F8
     - Line 2   : Medium/Semibold 19px, #EAECEF
  2. FFmpeg format=rgba,fade alpha timeline:
     - 0.00s - 0.35s : Smooth alpha fade-in (0% -> 100%)
     - 0.35s - 3.20s : Crisp hold (100%)
     - 3.20s - 4.00s : Smooth alpha fade-out (100% -> 0%), leaving 4.0s - 20.0s 100% unobstructed
  3. Encoding:
     - Video: -c:v libx264 -preset slow -crf 15 -pix_fmt yuv420p -movflags +faststart
     - Audio: -c:a copy (lossless bit-exact passthrough)
"""
import os
import hashlib
import argparse
import tempfile
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_W = 1920
FRAME_H = 1080
SS = 4  # 4x supersampling factor


def pick_font(candidates: list[str], size_px: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size_px)
    return ImageFont.load_default(size=size_px)


def downsample_premultiplied_rgba(img_hi: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Downsample a 4x RGBA image in premultiplied-alpha space using Lanczos resampling
    to eliminate straight-alpha edge halos and preserve exact subpixel border translucency.
    """
    arr = np.asarray(img_hi, dtype=np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    prem = np.concatenate([arr[:, :, :3] * alpha, arr[:, :, 3:4]], axis=2)

    channels = []
    for c in range(4):
        ch_img = Image.fromarray(np.clip(prem[:, :, c], 0.0, 255.0).astype(np.uint8), mode="L")
        ch_lo = ch_img.resize(target_size, resample=Image.Resampling.LANCZOS)
        channels.append(np.asarray(ch_lo, dtype=np.float32))

    prem_lo = np.stack(channels, axis=2)
    a_lo = prem_lo[:, :, 3:4]
    rgb_lo = np.where(a_lo > 0.5, prem_lo[:, :, :3] / np.maximum(a_lo / 255.0, 1e-6), 0.0)
    out = np.concatenate([np.clip(np.round(rgb_lo), 0, 255), np.clip(np.round(a_lo), 0, 255)], axis=2).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def build_4x_cinema_badge_overlay(out_png_path: str, line1: str, line2: str) -> dict:
    font_title = pick_font(
        [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        36 * SS,
    )
    font_sub = pick_font(
        [
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ],
        19 * SS,
    )

    dummy = ImageDraw.Draw(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
    b1 = dummy.textbbox((0, 0), line1, font=font_title)
    b2 = dummy.textbbox((0, 0), line2, font=font_sub)

    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

    pad_x_l = 24 * SS
    pad_x_r = 26 * SS
    pad_y_t = 18 * SS
    pad_y_b = 18 * SS
    line_gap = 10 * SS

    card_w_4x = max(w1, w2) + pad_x_l + pad_x_r
    card_h_4x = pad_y_t + h1 + line_gap + h2 + pad_y_b

    card_w_4x = ((card_w_4x + SS - 1) // SS) * SS
    card_h_4x = ((card_h_4x + SS - 1) // SS) * SS

    margin_4x = 8 * SS
    cw_4x = card_w_4x + 2 * margin_4x
    ch_4x = card_h_4x + 2 * margin_4x

    x0, y0 = margin_4x, margin_4x
    x1, y1 = x0 + card_w_4x - 1, y0 + card_h_4x - 1
    rad_4x = 16 * SS
    border_w_4x = 1 * SS

    layer_bg = Image.new("RGBA", (cw_4x, ch_4x), (0, 0, 0, 0))
    ImageDraw.Draw(layer_bg).rounded_rectangle([x0, y0, x1, y1], radius=rad_4x, fill=(12, 16, 14, 196))

    mask_outer = Image.new("L", (cw_4x, ch_4x), 0)
    ImageDraw.Draw(mask_outer).rounded_rectangle([x0, y0, x1, y1], radius=rad_4x, fill=255)
    mask_inner = Image.new("L", (cw_4x, ch_4x), 0)
    ImageDraw.Draw(mask_inner).rounded_rectangle(
        [x0 + border_w_4x, y0 + border_w_4x, x1 - border_w_4x, y1 - border_w_4x],
        radius=max(0, rad_4x - border_w_4x),
        fill=255,
    )
    ring_alpha = np.clip(
        np.asarray(mask_outer, dtype=np.int16) - np.asarray(mask_inner, dtype=np.int16), 0, 255
    ).astype(np.float32) * (32.0 / 255.0)
    layer_border = Image.new("RGBA", (cw_4x, ch_4x), (255, 255, 255, 0))
    layer_border.putalpha(Image.fromarray(np.round(ring_alpha).astype(np.uint8), mode="L"))

    layer_txt = Image.new("RGBA", (cw_4x, ch_4x), (0, 0, 0, 0))
    d_txt = ImageDraw.Draw(layer_txt)
    d_txt.text((x0 + pad_x_l - b1[0], y0 + pad_y_t - b1[1]), line1, font=font_title, fill=(0x8A, 0xB4, 0xF8, 255))
    d_txt.text((x0 + pad_x_l - b2[0], y0 + pad_y_t + h1 + line_gap - b2[1]), line2, font=font_sub, fill=(0xEA, 0xEC, 0xEF, 255))

    comp_4x = Image.alpha_composite(layer_bg, layer_border)
    comp_4x = Image.alpha_composite(comp_4x, layer_txt)
    comp_1x = downsample_premultiplied_rgba(comp_4x, (cw_4x // SS, ch_4x // SS))

    full_overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    margin_1x = margin_4x // SS
    full_overlay.paste(comp_1x, (36 - margin_1x, 32 - margin_1x))
    full_overlay.save(out_png_path)

    return {
        "overlay_png": out_png_path,
        "card_x": 36,
        "card_y": 32,
        "card_w_px": card_w_4x // SS,
        "card_h_px": card_h_4x // SS,
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_frame_rgb(mp4_path: str, t_sec: float, out_png: str) -> np.ndarray:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t_sec:.3f}", "-i", mp4_path,
            "-frames:v", "1", out_png,
        ],
        check=True,
    )
    return np.asarray(Image.open(out_png).convert("RGB"), dtype=np.float32)


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Render 3.5s Cinema Title-Badge Edition MP4")
    parser.add_argument("--input", default=os.path.join(project_dir, "lotus_rain_65536_cuda.mp4"))
    parser.add_argument("--output", default=os.path.join(project_dir, "lotus_rain_65536_cuda_badged.mp4"))
    parser.add_argument("--preview", default=os.path.join(project_dir, "lotus_rain_65536_preview_1_5s.png"))
    parser.add_argument("--title", default="Gemini 4 Carbon")
    parser.add_argument("--subtitle", default="65,536 Raindrops \u00b7 Pure CUDA/C++ \u00b7 0 Assets")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Source MP4 not found: {args.input}")

    orig_sha_before = sha256_file(args.input)
    orig_mtime_before = os.path.getmtime(args.input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        overlay_png = os.path.join(tmp_dir, "cinema_badge_overlay.png")
        badge_meta = build_4x_cinema_badge_overlay(overlay_png, args.title, args.subtitle)
        print(f"[BadgeGen] Rendered 4x->1x RGBA card at ({badge_meta['card_x']}, {badge_meta['card_y']}), "
              f"size={badge_meta['card_w_px']}x{badge_meta['card_h_px']}px")

        filter_complex = (
            "[1:v]format=rgba,"
            "fade=t=in:st=0.00:d=0.35:alpha=1,"
            "fade=t=out:st=3.20:d=0.80:alpha=1[badge];"
            "[0:v][badge]overlay=0:0:enable='lte(t,4.02)':format=auto[vout]"
        )

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", args.input,
            "-loop", "1", "-framerate", "60", "-t", "20.0", "-i", overlay_png,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "slow", "-crf", "15",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "copy",
            args.output,
        ]
        print("[FFmpeg] Encoding badged cinema MP4 (-c:v libx264 -preset slow -crf 15 -pix_fmt yuv420p -movflags +faststart -c:a copy)...")
        subprocess.run(cmd, check=True)

        extract_frame_rgb(args.output, 1.50, args.preview)
        for t_chk in [0.18, 1.50, 3.60, 4.50, 6.00]:
            f_orig = extract_frame_rgb(args.input, t_chk, os.path.join(tmp_dir, f"chk_orig_{t_chk:.2f}.png"))
            f_badg = extract_frame_rgb(args.output, t_chk, os.path.join(tmp_dir, f"chk_badg_{t_chk:.2f}.png"))
            roi_diff = float(np.mean(np.abs(f_badg[32:150, 36:520] - f_orig[32:150, 36:520])))
            bg_diff = float(np.mean(np.abs(f_badg[300:, 600:] - f_orig[300:, 600:])))
            print(f"[TimelineQA] t={t_chk:4.2f}s | Card ROI mean |diff| = {roi_diff:6.2f} | Unobstructed pond |diff| = {bg_diff:5.2f}")

    orig_sha_after = sha256_file(args.input)
    orig_mtime_after = os.path.getmtime(args.input)
    assert orig_sha_before == orig_sha_after and orig_mtime_before == orig_mtime_after, "Original MP4 was modified!"
    print(f"[IntegrityQA] Original MP4 untouched (SHA-256={orig_sha_after[:16]}... verified identical).")


if __name__ == "__main__":
    main()
