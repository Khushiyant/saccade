"""Demo: surprise-gated streaming over a real video -> annotated GIF + speed summary.

Slides a T-frame window by --stride (overlapping windows = the real streaming regime), and
the gate skips windows whose content barely changed. Run:
  uv run python scripts/demo.py --video clip.mp4 --out docs/demo.gif
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from saccade.config import ModelConfig
from saccade.model import load_encoder
from saccade.streaming.surprise_gate import SurpriseGatedEncoder

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read_frames(path, maxf):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < maxf:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def clip_tensor(frames, res):
    arr = np.stack([cv2.resize(f, (res, res)) for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to("cuda", torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--out", default="docs/demo.gif")
    args = ap.parse_args()

    frames = read_frames(args.video, args.max_frames)
    T, stride = args.frames, args.stride
    starts = list(range(0, len(frames) - T + 1, stride))
    nwin = len(starts)
    enc = load_encoder(ModelConfig(checkpoint="vitl", frames=T, resolution=args.res,
                                   device="cuda", dtype="float16"))
    gate = SurpriseGatedEncoder(enc, tau=args.tau, gate="latent")
    gate.reset()
    depth = len(enc.blocks)

    with torch.no_grad():
        enc.embed(clip_tensor(frames[:T], args.res))
        torch.cuda.synchronize()

    decisions, encode_ms = [], []
    for s in starts:
        clip = clip_tensor(frames[s:s + T], args.res)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, info = gate.step(clip)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        sv = info["surprise"]
        sv = 1.0 if sv == float("inf") else float(sv)
        decisions.append((info["encoded"], sv, ms))
        if info["encoded"]:
            encode_ms.append(ms)
        print(f"win {len(decisions):2d}/{nwin}: {'ENCODE' if info['encoded'] else 'skip  '} "
              f"surprise={sv:.4f}  {ms:6.1f} ms")

    enc_cost = sum(encode_ms) / max(1, len(encode_ms))
    n_enc = sum(1 for d in decisions if d[0])
    gated_total = sum(d[2] for d in decisions)
    full_total = nwin * enc_cost
    summary = {
        "video": os.path.basename(args.video), "windows": nwin,
        "encoded": n_enc, "skipped": nwin - n_enc, "skip_rate": round((nwin - n_enc) / nwin, 3),
        "encode_ms": round(enc_cost, 1),
        "embed_fps_gated": round(nwin / (gated_total / 1000), 1),
        "embed_fps_full": round(nwin / (full_total / 1000), 1),
        "speedup": round(full_total / gated_total, 2), "tau": args.tau, "stride": stride,
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/demo.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    if args.out.lower() == "none":
        return
    fb = ImageFont.truetype(FONT, 20)
    fm = ImageFont.truetype(FONTR, 16)
    fs = ImageFont.truetype(FONTR, 14)
    VW, PANEL = 640, 158
    cum = cum_enc = 0
    gated_so_far = 0.0
    gif = []
    for i, s in enumerate(starts):
        enc_i, surp, ms = decisions[i]
        cum += depth if enc_i else 0
        cum_enc += 1 if enc_i else 0
        gated_so_far += ms
        pct = 100 * cum / ((i + 1) * depth)
        skip_pct = 100 * (i + 1 - cum_enc) / (i + 1)
        speed = ((i + 1) * enc_cost) / max(gated_so_far, 1e-6)

        f = frames[s + T - 1]
        vh = int(VW * f.shape[0] / f.shape[1])
        canvas = Image.new("RGB", (VW, vh + PANEL), (14, 17, 24))
        canvas.paste(Image.fromarray(f).resize((VW, vh), Image.LANCZOS), (0, 0))
        d = ImageDraw.Draw(canvas)
        x0, x1, py = 22, VW - 22, vh + 16
        col = (232, 83, 60) if enc_i else (57, 185, 119)

        d.rounded_rectangle([x0, py, x0 + 150, py + 34], radius=17, fill=col)
        d.text((x0 + 75, py + 17), "ENCODE" if enc_i else "SKIP  reuse",
               font=fb, fill=(255, 255, 255), anchor="mm")
        d.text((x0 + 166, py + 9), f"embedding {i + 1} / {nwin}", font=fm, fill=(214, 220, 232))
        d.text((x1, py + 9), f"{skip_pct:.0f}% skipped    {speed:.1f}x faster",
               font=fm, fill=(120, 190, 255), anchor="ra")

        gy = py + 52
        d.text((x0, gy), "compute vs full", font=fs, fill=(150, 160, 182))
        d.text((x1, gy), f"{pct:.0f}%", font=fs, fill=(214, 220, 232), anchor="ra")
        ty = gy + 20
        d.rounded_rectangle([x0, ty, x1, ty + 14], radius=7, fill=(38, 44, 58))
        d.rounded_rectangle([x0, ty, x0 + (x1 - x0) * pct / 100, ty + 14], radius=7, fill=(86, 166, 255))

        sy = ty + 30
        smax = max(args.tau * 2.0, 0.03)
        d.text((x0, sy), "surprise", font=fs, fill=(150, 160, 182))
        d.text((x1, sy), f"{surp:.3f}   (tau {args.tau:g})", font=fs, fill=(214, 220, 232), anchor="ra")
        sty = sy + 20
        d.rounded_rectangle([x0, sty, x1, sty + 14], radius=7, fill=(38, 44, 58))
        d.rounded_rectangle([x0, sty, x0 + (x1 - x0) * min(surp, smax) / smax, sty + 14],
                            radius=7, fill=col)
        tx = x0 + (x1 - x0) * min(args.tau, smax) / smax
        d.line([tx, sty - 5, tx, sty + 19], fill=(245, 245, 245), width=2)
        gif.append(canvas)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gif[0].save(args.out, save_all=True, append_images=gif[1:], duration=110, loop=0, optimize=True)
    print("wrote", args.out, f"({len(gif)} frames, {gif[0].size})")

    mp4 = os.path.splitext(args.out)[0] + ".mp4"
    w, h = gif[0].size
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h))
    for im in gif:
        vw.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    vw.release()
    print("wrote", mp4, f"({os.path.getsize(mp4) // 1024} KB)")


if __name__ == "__main__":
    main()
