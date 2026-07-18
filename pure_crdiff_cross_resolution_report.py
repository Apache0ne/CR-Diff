#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure CR-Diff evaluation on a complete single-file SDXL checkpoint.

This script does not use Diff-ES or remove any transformer block. It compares:
  A. the complete dense checkpoint
  B. the same full checkpoint after CR-Diff-style down/mid/up magnitude pruning

The primary metric is ImageReward at shifted resolutions. Pixel PSNR/MAE are
reported only as secondary change measurements. All images, metrics, pruning
statistics, errors and configuration are embedded in one HTML file.
"""

from __future__ import annotations

import argparse
import base64
import gc
import html
import json
import math
import random
import shutil
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from lib.prune_unet import find_layers
from single_file_sdxl_utils import load_sdxl_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--output", default="/content/pure_crdiff_cross_resolution_report.html")
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--resolutions", default="400x560,560x400,480x360,360x480")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument("--model-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--scheduler", choices=["checkpoint", "dpmpp-sde-normal", "ddim-trailing", "euler-trailing"], default="dpmpp-sde-normal")
    p.add_argument("--down-ratio", type=float, default=0.03)
    p.add_argument("--mid-ratio", type=float, default=0.03)
    p.add_argument("--up-ratio", type=float, default=0.05)
    p.add_argument("--threshold-sample-per-layer", type=int, default=20000)
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument("--work-dir", default="/content/pure_crdiff_cross_resolution_work")
    p.add_argument("--skip-imagereward", action="store_true")
    return p.parse_args()


def parse_resolutions(text: str) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        w, h = item.split("x", 1)
        width, height = int(w), int(h)
        if width <= 0 or height <= 0 or width % 8 or height % 8:
            raise ValueError(f"Resolution must be positive and divisible by 8: {item}")
        result.append((width, height))
    if not result:
        raise ValueError("No resolutions supplied")
    return result


def load_prompts(path: Path, count: int, seed: int) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: List[str] = []
    seen = set()
    for ann in data.get("annotations", []):
        caption = str(ann.get("caption", "")).strip()
        if caption and caption not in seen:
            prompts.append(caption)
            seen.add(caption)
    if count > len(prompts):
        raise ValueError(f"Requested {count} prompts, found {len(prompts)}")
    return random.Random(seed).sample(prompts, count)


def region_for_name(name: str) -> str | None:
    if name.startswith("down_blocks"):
        return "down"
    if name.startswith("mid_block"):
        return "mid"
    if name.startswith("up_blocks"):
        return "up"
    return None


def selected_layers(model) -> Dict[str, List[Tuple[str, torch.nn.Module]]]:
    groups: Dict[str, List[Tuple[str, torch.nn.Module]]] = {"down": [], "mid": [], "up": []}
    for name, layer in find_layers(model).items():
        region = region_for_name(name)
        if region is None:
            continue
        if "attn2" in name:
            continue
        if not any(token in name for token in ("attn", "ff")):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None or weight.numel() == 0:
            continue
        groups[region].append((name, layer))
    return groups


def sampled_abs(weight: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    flat = weight.detach().abs().reshape(-1)
    if flat.numel() <= count:
        return flat.float().cpu()
    indices = torch.randint(0, flat.numel(), (count,), device=flat.device, generator=generator)
    return flat[indices].float().cpu()


def apply_crdiff_pruning(
    model,
    ratios: Dict[str, float],
    sample_per_layer: int,
    seed: int,
) -> Dict[str, Any]:
    groups = selected_layers(model)
    analysis: Dict[str, Any] = {
        "method": "CR-Diff-style groupwise magnitude pruning with bounded threshold sampling",
        "attn2_cross_attention_preserved": True,
        "groups": {},
    }

    for group_name in ("down", "mid", "up"):
        ratio = float(ratios[group_name])
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"{group_name} ratio must be in [0,1), got {ratio}")
        layers = groups[group_name]
        total = int(sum(layer.weight.numel() for _, layer in layers))
        if ratio == 0.0 or not layers:
            analysis["groups"][group_name] = {
                "requested_ratio": ratio,
                "actual_ratio": 0.0,
                "threshold": 0.0,
                "layers": len(layers),
                "target_parameters": total,
                "zeroed_parameters": 0,
            }
            continue

        device = layers[0][1].weight.device
        rng = torch.Generator(device=device).manual_seed(seed + {"down": 11, "mid": 23, "up": 37}[group_name])
        samples = [sampled_abs(layer.weight, sample_per_layer, rng) for _, layer in layers]
        joined = torch.cat(samples)
        threshold = float(torch.quantile(joined, ratio).item())
        del samples, joined

        zeroed = 0
        with torch.no_grad():
            for _name, layer in layers:
                weight = layer.weight.data
                mask = weight.abs() > threshold
                zeroed += int((~mask).sum().item())
                weight.mul_(mask)

        analysis["groups"][group_name] = {
            "requested_ratio": ratio,
            "actual_ratio": float(zeroed / max(total, 1)),
            "threshold": threshold,
            "layers": len(layers),
            "target_parameters": total,
            "zeroed_parameters": int(zeroed),
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = sum(v["target_parameters"] for v in analysis["groups"].values())
    zeroed = sum(v["zeroed_parameters"] for v in analysis["groups"].values())
    analysis["overall_target_parameters"] = int(total)
    analysis["overall_zeroed_parameters"] = int(zeroed)
    analysis["overall_actual_ratio"] = float(zeroed / max(total, 1))
    return analysis


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(int(seed))


def generate(pipe, prompt: str, width: int, height: int, steps: int, cfg: float, seed: int) -> Tuple[Image.Image, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=make_generator(pipe._execution_device, seed),
            output_type="pil",
        ).images[0].convert("RGB")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return image, time.perf_counter() - start


def pixel_metrics(reference: Image.Image, candidate: Image.Image) -> Dict[str, float]:
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(candidate, dtype=np.float32)
    delta = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    return {
        "psnr_db": float(20.0 * math.log10(255.0 / math.sqrt(max(mse, 1e-12)))),
        "mae_0_1": float(delta.mean() / 255.0),
        "rmse_0_1": float(math.sqrt(mse) / 255.0),
        "pixels_gt_16_pct": float(np.any(delta > 16.0, axis=2).mean() * 100.0),
    }


def encode_jpeg(image: Image.Image, quality: int) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def difference_image(reference: Image.Image, candidate: Image.Image, gain: float = 4.0) -> Image.Image:
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(candidate, dtype=np.float32)
    diff = np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def score_imagereward(rows: List[Dict[str, Any]]) -> Tuple[bool, str | None]:
    try:
        import ImageReward as RM

        print("Loading ImageReward-v1.0 after releasing the SDXL pipeline...", flush=True)
        model = RM.load("ImageReward-v1.0")
        for index, row in enumerate(rows):
            dense_score = model.score(row["prompt"], [str(row["dense_path"])])
            cr_score = model.score(row["prompt"], [str(row["cr_path"])])
            if isinstance(dense_score, list):
                dense_score = dense_score[0]
            if isinstance(cr_score, list):
                cr_score = cr_score[0]
            row["dense_imagereward"] = float(dense_score)
            row["cr_imagereward"] = float(cr_score)
            row["imagereward_delta"] = float(cr_score - dense_score)
            print(
                f"[IR {index+1:02d}/{len(rows)}] dense={dense_score:.6f} CR={cr_score:.6f} delta={cr_score-dense_score:+.6f}",
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, None
    except Exception:
        error = traceback.format_exc()
        print("ImageReward scoring failed:\n" + error, flush=True)
        for row in rows:
            row["dense_imagereward"] = None
            row["cr_imagereward"] = None
            row["imagereward_delta"] = None
        return False, error


def mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    return float(np.mean(items)) if items else float("nan")


def build_html(args, rows, resolutions, pruning, ir_ok: bool, ir_error: str | None) -> str:
    deltas = [row["imagereward_delta"] for row in rows if row["imagereward_delta"] is not None]
    dense_ir = [row["dense_imagereward"] for row in rows if row["dense_imagereward"] is not None]
    cr_ir = [row["cr_imagereward"] for row in rows if row["cr_imagereward"] is not None]
    win_count = sum(delta > 0 for delta in deltas)
    summary = {
        "mode": "pure CR-Diff on the complete full model; no Diff-ES and no block removal",
        "requested_ratios": {"down": args.down_ratio, "mid": args.mid_ratio, "up": args.up_ratio},
        "actual_pruning": pruning,
        "imagereward_available": ir_ok,
        "mean_dense_imagereward": mean(dense_ir) if dense_ir else None,
        "mean_cr_imagereward": mean(cr_ir) if cr_ir else None,
        "mean_imagereward_delta": mean(deltas) if deltas else None,
        "cr_wins": int(win_count),
        "cases": len(rows),
        "cr_win_rate": float(win_count / len(deltas)) if deltas else None,
        "mean_dense_time_sec": mean(row["dense_time"] for row in rows),
        "mean_cr_time_sec": mean(row["cr_time"] for row in rows),
    }

    cards = []
    table = []
    for row in sorted(rows, key=lambda r: (r["imagereward_delta"] is None, r["imagereward_delta"] or 0.0)):
        delta_text = "N/A" if row["imagereward_delta"] is None else f"{row['imagereward_delta']:+.6f}"
        dense_text = "N/A" if row["dense_imagereward"] is None else f"{row['dense_imagereward']:.6f}"
        cr_text = "N/A" if row["cr_imagereward"] is None else f"{row['cr_imagereward']:.6f}"
        pm = row["pixel_metrics"]
        table.append(
            "<tr>"
            f"<td>{row['index']}</td><td>{row['seed']}</td><td>{row['width']}×{row['height']}</td>"
            f"<td>{dense_text}</td><td>{cr_text}</td><td>{delta_text}</td>"
            f"<td>{pm['psnr_db']:.3f}</td><td>{pm['mae_0_1']:.6f}</td>"
            f"<td>{row['dense_time']:.3f}</td><td>{row['cr_time']:.3f}</td>"
            f"<td>{html.escape(row['prompt'])}</td></tr>"
        )
        cards.append(
            f"""
<section class="card"><h3>Case {row['index']} — seed {row['seed']} — {row['width']}×{row['height']}</h3>
<p class="prompt">{html.escape(row['prompt'])}</p>
<p><b>ImageReward:</b> dense {dense_text}, CR {cr_text}, delta {delta_text} &nbsp;
<b>PSNR:</b> {pm['psnr_db']:.3f} dB &nbsp; <b>MAE:</b> {pm['mae_0_1']:.6f}</p>
<div class="images">
<figure><figcaption>Dense full model</figcaption><img src="data:image/jpeg;base64,{encode_jpeg(row['dense_image'], args.jpeg_quality)}"></figure>
<figure><figcaption>Pure CR-Diff full model</figcaption><img src="data:image/jpeg;base64,{encode_jpeg(row['cr_image'], args.jpeg_quality)}"></figure>
<figure><figcaption>Absolute difference ×4</figcaption><img src="data:image/jpeg;base64,{encode_jpeg(difference_image(row['dense_image'], row['cr_image']), args.jpeg_quality)}"></figure>
</div></section>
"""
        )

    machine = html.escape(json.dumps({
        "arguments": vars(args),
        "resolutions": resolutions,
        "summary": summary,
        "pruning": pruning,
        "imagereward_error": ir_error,
        "per_case": [{k: v for k, v in row.items() if not k.endswith("_image") and not k.endswith("_path")} for row in rows],
    }, indent=2, default=str))

    if ir_ok:
        delta_value = f"{summary['mean_imagereward_delta']:+.6f}"
        win_value = f"{summary['cr_wins']}/{summary['cases']} ({summary['cr_win_rate']*100:.1f}%)"
        ir_note = "Positive ImageReward delta means CR-Diff scored better for prompt-image alignment and perceptual preference."
    else:
        delta_value = "N/A"
        win_value = "N/A"
        ir_note = "ImageReward could not be loaded. The full traceback is embedded below; the visual report and pixel-change metrics were still completed."

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Pure CR-Diff cross-resolution report</title><style>
body{{font-family:Arial,sans-serif;background:#f4f5f7;color:#171717;margin:24px;line-height:1.4}}
.card{{background:white;border:1px solid #ddd;border-radius:10px;padding:18px;margin:16px 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.metric{{background:#f7f7f8;border-radius:8px;padding:14px}} .value{{font-size:23px;font-weight:700}}
.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}} img{{width:100%;height:auto;border:1px solid #bbb}}
figure{{margin:0}} figcaption{{font-weight:700;margin-bottom:6px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #ccc;padding:7px;vertical-align:top}} th{{background:#eee}} .prompt{{font-style:italic}}
pre{{white-space:pre-wrap;word-break:break-word;background:#111;color:#eee;padding:14px;overflow:auto}}
@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><h1>Pure CR-Diff on the complete full model</h1>
<section class="card"><h2>Question tested</h2><p>Does CR-Diff improve perceptual quality and semantic alignment at shifted resolutions? No transformer block is removed and Diff-ES is not imported. {html.escape(ir_note)}</p></section>
<section class="card"><h2>Aggregate result</h2><div class="metrics">
<div class="metric"><div class="value">{summary['mean_dense_imagereward'] if summary['mean_dense_imagereward'] is not None else 'N/A'}</div>Dense mean ImageReward</div>
<div class="metric"><div class="value">{summary['mean_cr_imagereward'] if summary['mean_cr_imagereward'] is not None else 'N/A'}</div>CR mean ImageReward</div>
<div class="metric"><div class="value">{delta_value}</div>Mean CR − dense ImageReward</div>
<div class="metric"><div class="value">{win_value}</div>CR wins</div>
<div class="metric"><div class="value">{pruning['overall_actual_ratio']*100:.3f}%</div>Actual target sparsity</div>
<div class="metric"><div class="value">{summary['mean_dense_time_sec']:.3f}s / {summary['mean_cr_time_sec']:.3f}s</div>Dense / CR mean generation time</div>
</div></section>
<section class="card"><h2>Configuration</h2><p><b>Checkpoint:</b> {html.escape(args.model_path)}<br>
<b>Mode:</b> pure CR-Diff only; complete UNet retained<br>
<b>Ratios:</b> down={args.down_ratio:.4f}, mid={args.mid_ratio:.4f}, up={args.up_ratio:.4f}<br>
<b>Cross-attention:</b> preserved &nbsp; <b>Scheduler:</b> {html.escape(args.scheduler)} &nbsp; <b>Steps:</b> {args.steps} &nbsp; <b>CFG:</b> {args.cfg_scale}<br>
<b>Resolutions:</b> {', '.join(f'{w}×{h}' for w, h in resolutions)} &nbsp; <b>Prompts:</b> {args.num_prompts}</p></section>
<section class="card"><h2>Pruning analysis</h2><pre>{html.escape(json.dumps(pruning, indent=2))}</pre></section>
<section class="card"><h2>Per-case results</h2><table><thead><tr><th>#</th><th>Seed</th><th>Resolution</th><th>Dense IR</th><th>CR IR</th><th>Delta</th><th>PSNR</th><th>MAE</th><th>Dense s</th><th>CR s</th><th>Prompt</th></tr></thead><tbody>{''.join(table)}</tbody></table></section>
{''.join(cards)}
<section class="card"><h2>Embedded machine-readable report and errors</h2><pre>{machine}</pre></section>
</body></html>"""


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    ann_path = Path(args.ann_file).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    resolutions = parse_resolutions(args.resolutions)
    prompts = load_prompts(ann_path, args.num_prompts, args.base_seed)
    shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / "dense").mkdir(parents=True, exist_ok=True)
    (work_dir / "cr").mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading complete full SDXL checkpoint: {model_path}", flush=True)
    pipe = load_sdxl_pipeline(
        device=str(device),
        model_path=str(model_path),
        model_dtype=args.model_dtype,
        scheduler=args.scheduler,
        local_files_only=False,
    )

    rows: List[Dict[str, Any]] = []
    total_cases = len(prompts) * len(resolutions)
    print(f"Generating {total_cases} dense full-model images", flush=True)
    index = 0
    for prompt_index, prompt in enumerate(prompts):
        for resolution_index, (width, height) in enumerate(resolutions):
            seed = args.base_seed + prompt_index * 100 + resolution_index
            image, elapsed = generate(pipe, prompt, width, height, args.steps, args.cfg_scale, seed)
            path = work_dir / "dense" / f"{index:03d}.png"
            image.save(path)
            rows.append({
                "index": index,
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "dense_time": elapsed,
                "dense_image": image,
                "dense_path": path,
            })
            print(f"[Dense {index+1:02d}/{total_cases}] {width}x{height} {elapsed:.3f}s", flush=True)
            index += 1

    ratios = {"down": args.down_ratio, "mid": args.mid_ratio, "up": args.up_ratio}
    print(f"Applying pure CR-Diff pruning to the full UNet: {ratios}", flush=True)
    pruning = apply_crdiff_pruning(pipe.unet, ratios, args.threshold_sample_per_layer, args.base_seed)
    print(json.dumps(pruning, indent=2), flush=True)

    print(f"Generating {total_cases} pure CR-Diff images", flush=True)
    for row in rows:
        image, elapsed = generate(pipe, row["prompt"], row["width"], row["height"], args.steps, args.cfg_scale, row["seed"])
        path = work_dir / "cr" / f"{row['index']:03d}.png"
        image.save(path)
        row["cr_time"] = elapsed
        row["cr_image"] = image
        row["cr_path"] = path
        row["pixel_metrics"] = pixel_metrics(row["dense_image"], image)
        print(f"[CR {row['index']+1:02d}/{total_cases}] {row['width']}x{row['height']} {elapsed:.3f}s", flush=True)

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_imagereward:
        ir_ok, ir_error = False, "ImageReward explicitly skipped by --skip-imagereward"
        for row in rows:
            row["dense_imagereward"] = None
            row["cr_imagereward"] = None
            row["imagereward_delta"] = None
    else:
        ir_ok, ir_error = score_imagereward(rows)

    report = build_html(args, rows, resolutions, pruning, ir_ok, ir_error)
    output.write_text(report, encoding="utf-8")
    print("=" * 80, flush=True)
    print(f"One-file report: {output}", flush=True)
    if ir_ok:
        deltas = [row["imagereward_delta"] for row in rows]
        print(f"Mean ImageReward delta (CR-dense): {np.mean(deltas):+.6f}", flush=True)
        print(f"CR wins: {sum(v > 0 for v in deltas)}/{len(deltas)}", flush=True)
    print("=" * 80, flush=True)
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
