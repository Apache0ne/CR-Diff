#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-file HTML report for Diff-ES block removal plus CR-Diff-style pruning.

Variants:
  A. Dense complete single-file SDXL checkpoint.
  B. The same checkpoint with one BasicTransformerBlock skipped at every step.
  C. Variant B plus mild group-wise magnitude pruning over the remaining UNet.

The selected block is excluded from the CR-Diff magnitude budget. Cross-attention
(attn2) remains unpruned, matching CR-Diff's SDXL implementation.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from lib.prune_unet import find_layers
from single_file_sdxl_utils import load_sdxl_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--diff-es-dir", default="/content/Diff-ES")
    p.add_argument("--ann-file", default="/content/coco/annotations/captions_val2017.json")
    p.add_argument("--block-id", type=int, default=11)
    p.add_argument("--down-ratio", type=float, default=0.03)
    p.add_argument("--mid-ratio", type=float, default=0.03)
    p.add_argument("--up-ratio", type=float, default=0.05)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument("--model-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument(
        "--scheduler",
        choices=["checkpoint", "dpmpp-sde-normal", "ddim-trailing", "euler-trailing"],
        default="dpmpp-sde-normal",
    )
    p.add_argument("--model-config", default=None)
    p.add_argument("--original-config-file", default=None)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--threshold-sample-per-layer", type=int, default=100000)
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument("--output", default="/content/block11_crdiff_hybrid_report.html")
    return p.parse_args()


def load_diff_es(path: str):
    sdxl_dir = Path(path).expanduser().resolve() / "sdxl"
    if not sdxl_dir.is_dir():
        raise FileNotFoundError(f"Diff-ES SDXL directory not found: {sdxl_dir}")
    sys.path.insert(0, str(sdxl_dir))
    import evo_pruning_sdxl as diff_es

    return diff_es


def load_prompts(path: Path, count: int, seed: int) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: List[str] = []
    seen = set()
    for ann in data.get("annotations", []):
        caption = str(ann.get("caption", "")).strip()
        if caption and caption not in seen:
            seen.add(caption)
            prompts.append(caption)
    if count > len(prompts):
        raise ValueError(f"Requested {count} prompts, found only {len(prompts)}")
    return random.Random(seed).sample(prompts, count)


def clear_blockdrop(unet) -> None:
    if hasattr(unet, "clear_all_accel"):
        unet.clear_all_accel()
    elif hasattr(unet, "set_layerdrop"):
        unet.set_layerdrop([])
    else:
        unet.drop_block_ids = set()


def set_global_blockdrop(unet, block_id: int) -> None:
    clear_blockdrop(unet)
    if hasattr(unet, "set_layerdrop"):
        unet.set_layerdrop([int(block_id)])
    else:
        unet.drop_block_ids = {int(block_id)}


def block_prefixes(unet, block_id: int) -> List[str]:
    prefixes = [
        name
        for name, module in unet.named_modules()
        if getattr(module, "_blk_id", None) == int(block_id)
    ]
    if not prefixes:
        raise RuntimeError(f"Could not find module prefix for Diff-ES block {block_id}")
    return sorted(set(prefixes))


def under_any_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def sample_abs(weight: torch.Tensor, max_items: int, rng: torch.Generator) -> torch.Tensor:
    flat = weight.detach().abs().reshape(-1)
    if flat.numel() <= max_items:
        return flat.float().cpu()
    indices = torch.randint(
        0,
        flat.numel(),
        (max_items,),
        generator=rng,
        device=flat.device,
    )
    return flat[indices].float().cpu()


def prune_groupwise_magnitude(
    model,
    ratios: Dict[str, float],
    *,
    excluded_prefixes: Sequence[str],
    sample_per_layer: int,
    seed: int,
) -> Dict[str, Any]:
    """CR-Diff-style down/mid/up pruning without a giant concatenated tensor."""
    all_layers = find_layers(model)
    group_prefix = {"down": "down_blocks", "mid": "mid_block", "up": "up_blocks"}
    analysis: Dict[str, Any] = {
        "method": "groupwise magnitude ratio with sampled threshold",
        "attn2_preserved": True,
        "excluded_module_prefixes": list(excluded_prefixes),
        "groups": {},
    }

    for group, ratio in ratios.items():
        ratio = float(ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"{group} ratio must be in [0,1), got {ratio}")
        prefix = group_prefix[group]
        selected = []
        for name, layer in all_layers.items():
            if not name.startswith(prefix):
                continue
            if "attn2" in name:
                continue
            if not any(token in name for token in ("attn", "ff")):
                continue
            if under_any_prefix(name, excluded_prefixes):
                continue
            if getattr(layer, "weight", None) is None or layer.weight.numel() == 0:
                continue
            selected.append((name, layer))

        if ratio == 0.0 or not selected:
            analysis["groups"][group] = {
                "requested_ratio": ratio,
                "actual_ratio": 0.0,
                "threshold": 0.0,
                "layers": len(selected),
                "target_parameters": int(sum(layer.weight.numel() for _, layer in selected)),
                "zeroed_parameters": 0,
            }
            continue

        sample_rng = torch.Generator(device=selected[0][1].weight.device).manual_seed(seed + len(group))
        samples = [sample_abs(layer.weight, sample_per_layer, sample_rng) for _, layer in selected]
        sample = torch.cat(samples)
        threshold = float(torch.quantile(sample, ratio).item())

        total = 0
        zeroed = 0
        with torch.no_grad():
            for _name, layer in selected:
                weight = layer.weight.data
                mask = weight.abs() > threshold
                total += weight.numel()
                zeroed += int((~mask).sum().item())
                weight.mul_(mask)

        analysis["groups"][group] = {
            "requested_ratio": ratio,
            "actual_ratio": float(zeroed / max(total, 1)),
            "threshold": threshold,
            "layers": len(selected),
            "threshold_sample_values": int(sample.numel()),
            "target_parameters": int(total),
            "zeroed_parameters": int(zeroed),
        }
        del samples, sample

    total_params = sum(v["target_parameters"] for v in analysis["groups"].values())
    total_zeroed = sum(v["zeroed_parameters"] for v in analysis["groups"].values())
    analysis["overall_actual_ratio"] = float(total_zeroed / max(total_params, 1))
    analysis["overall_target_parameters"] = int(total_params)
    analysis["overall_zeroed_parameters"] = int(total_zeroed)
    return analysis


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(int(seed))


def generate(pipe, prompt: str, seed: int, args: argparse.Namespace) -> Tuple[Image.Image, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            generator=make_generator(pipe._execution_device, seed),
            output_type="pil",
        ).images[0].convert("RGB")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return image, time.perf_counter() - start


def metrics(reference: Image.Image, candidate: Image.Image) -> Dict[str, float]:
    a = np.asarray(reference, dtype=np.uint8)
    b = np.asarray(candidate, dtype=np.uint8)
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
        "psnr_db": float(peak_signal_noise_ratio(a, b, data_range=255)),
        "mae_0_1": float(delta.mean() / 255.0),
        "rmse_0_1": float(np.sqrt(np.mean(delta * delta)) / 255.0),
        "pixels_gt_16_pct": float(np.any(delta > 16, axis=2).mean() * 100.0),
    }


def difference_image(reference: Image.Image, candidate: Image.Image, gain: float = 4.0) -> Image.Image:
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(candidate, dtype=np.float32)
    diff = np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def encode_jpeg(image: Image.Image, quality: int) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    ms = [row[key] for row in rows]
    return {
        "mean_ssim": finite_mean(m["ssim"] for m in ms),
        "median_ssim": float(np.median([m["ssim"] for m in ms])),
        "min_ssim": float(min(m["ssim"] for m in ms)),
        "mean_psnr_db": finite_mean(m["psnr_db"] for m in ms),
        "mean_mae_0_1": finite_mean(m["mae_0_1"] for m in ms),
        "mean_pixels_gt_16_pct": finite_mean(m["pixels_gt_16_pct"] for m in ms),
    }


def build_html(args, rows, cr_analysis, scheduler, component_counts) -> str:
    block_summary = aggregate(rows, "block_vs_dense")
    hybrid_summary = aggregate(rows, "hybrid_vs_dense")
    hybrid_vs_block = aggregate(rows, "hybrid_vs_block")
    dense_time = finite_mean(row["dense_time"] for row in rows)
    block_time = finite_mean(row["block_time"] for row in rows)
    hybrid_time = finite_mean(row["hybrid_time"] for row in rows)

    summary = {
        "block_id_removed_at_all_steps": args.block_id,
        "cr_diff_requested_ratios": {
            "down": args.down_ratio,
            "mid": args.mid_ratio,
            "up": args.up_ratio,
        },
        "block_vs_dense": block_summary,
        "hybrid_vs_dense": hybrid_summary,
        "hybrid_vs_block": hybrid_vs_block,
        "mean_times_sec": {
            "dense": dense_time,
            "block_only": block_time,
            "hybrid": hybrid_time,
        },
        "cr_diff_analysis": cr_analysis,
    }

    table_rows = []
    cards = []
    for row in sorted(rows, key=lambda x: x["hybrid_vs_dense"]["ssim"]):
        b = row["block_vs_dense"]
        h = row["hybrid_vs_dense"]
        table_rows.append(
            "<tr>"
            f"<td>{row['index']}</td><td>{row['seed']}</td>"
            f"<td>{b['ssim']:.6f}</td><td>{h['ssim']:.6f}</td>"
            f"<td>{h['psnr_db']:.3f}</td><td>{h['mae_0_1']:.6f}</td>"
            f"<td>{row['dense_time']:.3f}</td><td>{row['block_time']:.3f}</td><td>{row['hybrid_time']:.3f}</td>"
            f"<td>{html.escape(row['prompt'])}</td></tr>"
        )
        cards.append(
            f"""
<section class="card"><h3>Example {row['index']} — seed {row['seed']}</h3>
<p class="prompt">{html.escape(row['prompt'])}</p>
<p><b>Block-only SSIM:</b> {b['ssim']:.6f} &nbsp; <b>Hybrid SSIM:</b> {h['ssim']:.6f}
&nbsp; <b>Hybrid PSNR:</b> {h['psnr_db']:.3f} dB &nbsp; <b>Hybrid MAE:</b> {h['mae_0_1']:.6f}</p>
<div class="images">
<figure><figcaption>Dense</figcaption><img src="data:image/jpeg;base64,{row['dense_b64']}"></figure>
<figure><figcaption>Block {args.block_id} removed</figcaption><img src="data:image/jpeg;base64,{row['block_b64']}"></figure>
<figure><figcaption>Block-only difference ×4</figcaption><img src="data:image/jpeg;base64,{row['block_diff_b64']}"></figure>
<figure><figcaption>Block {args.block_id} + CR-Diff</figcaption><img src="data:image/jpeg;base64,{row['hybrid_b64']}"></figure>
<figure><figcaption>Hybrid difference ×4</figcaption><img src="data:image/jpeg;base64,{row['hybrid_diff_b64']}"></figure>
</div></section>
"""
        )

    machine = html.escape(
        json.dumps(
            {
                "arguments": vars(args),
                "scheduler_class": scheduler.__class__.__name__,
                "scheduler_config": dict(scheduler.config),
                "component_parameter_counts": component_counts,
                "summary": summary,
                "per_image": [
                    {
                        k: v
                        for k, v in row.items()
                        if not k.endswith("_b64") and not k.endswith("_image")
                    }
                    for row in rows
                ],
            },
            indent=2,
            default=str,
        )
    )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Diff-ES + CR-Diff hybrid report</title><style>
body{{font-family:Arial,sans-serif;background:#f4f5f7;color:#171717;margin:24px;line-height:1.4}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:18px;margin:16px 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.metric{{background:#f7f7f8;border-radius:8px;padding:14px}} .value{{font-size:23px;font-weight:700}}
.images{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}} img{{width:100%;height:auto;border:1px solid #bbb}}
figure{{margin:0}} figcaption{{font-weight:700;margin-bottom:6px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #ccc;padding:7px;vertical-align:top}} th{{background:#eee}} .prompt{{font-style:italic}}
pre{{white-space:pre-wrap;word-break:break-word;background:#111;color:#eee;padding:14px;overflow:auto}}
@media(max-width:1100px){{.images{{grid-template-columns:1fr 1fr}}}} @media(max-width:700px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><h1>Diff-ES block removal + CR-Diff pruning</h1>
<section class="card"><h2>Aggregate quality</h2><div class="metrics">
<div class="metric"><div class="value">{block_summary['mean_ssim']:.6f}</div>Block-only mean SSIM</div>
<div class="metric"><div class="value">{block_summary['min_ssim']:.6f}</div>Block-only worst SSIM</div>
<div class="metric"><div class="value">{hybrid_summary['mean_ssim']:.6f}</div>Hybrid mean SSIM</div>
<div class="metric"><div class="value">{hybrid_summary['min_ssim']:.6f}</div>Hybrid worst SSIM</div>
<div class="metric"><div class="value">{cr_analysis['overall_actual_ratio']*100:.3f}%</div>Actual CR-Diff target sparsity</div>
<div class="metric"><div class="value">{dense_time:.3f}s / {block_time:.3f}s / {hybrid_time:.3f}s</div>Dense / block / hybrid mean time</div>
</div></section>
<section class="card"><h2>Configuration</h2>
<p><b>Checkpoint:</b> {html.escape(args.model_path)}<br><b>Block removed at all steps:</b> {args.block_id}<br>
<b>CR-Diff requested ratios:</b> down={args.down_ratio:.4f}, mid={args.mid_ratio:.4f}, up={args.up_ratio:.4f}<br>
<b>Cross-attention:</b> preserved; <b>removed block excluded from CR budget:</b> yes<br>
<b>Scheduler:</b> {scheduler.__class__.__name__} ({html.escape(args.scheduler)}) &nbsp; <b>Steps:</b> {args.steps} &nbsp; <b>CFG:</b> {args.cfg_scale}<br>
<b>Resolution:</b> {args.width}×{args.height} &nbsp; <b>Prompts:</b> {args.num_prompts}</p></section>
<section class="card"><h2>Per-image results</h2><table><thead><tr><th>#</th><th>Seed</th><th>Block SSIM</th><th>Hybrid SSIM</th><th>Hybrid PSNR</th><th>Hybrid MAE</th><th>Dense s</th><th>Block s</th><th>Hybrid s</th><th>Prompt</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
{''.join(cards)}
<section class="card"><h2>Embedded machine-readable report</h2><pre>{machine}</pre></section>
</body></html>"""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SDXL report")

    model_path = Path(args.model_path).expanduser().resolve()
    ann_path = Path(args.ann_file).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    diff_es = load_diff_es(args.diff_es_dir)
    prompts = load_prompts(ann_path, args.num_prompts, args.base_seed)
    device = torch.device("cuda")

    print(f"Loading complete SDXL checkpoint: {model_path}")
    pipe = load_sdxl_pipeline(
        device=str(device),
        model_path=str(model_path),
        model_dtype=args.model_dtype,
        scheduler=args.scheduler,
        local_files_only=args.local_files_only,
        model_config=args.model_config,
        original_config_file=args.original_config_file,
    )
    pipe.unet = diff_es.attach_pruned_unet(pipe, device)
    clear_blockdrop(pipe.unet)
    block_count, wrapper_count = diff_es.count_pruned_blocks(pipe.unet)
    if not 0 <= args.block_id < block_count:
        raise ValueError(f"block-id must be in [0,{block_count - 1}]")

    component_counts = {
        "unet": sum(p.numel() for p in pipe.unet.parameters()),
        "text_encoder": sum(p.numel() for p in pipe.text_encoder.parameters()),
        "text_encoder_2": sum(p.numel() for p in pipe.text_encoder_2.parameters()),
        "vae": sum(p.numel() for p in pipe.vae.parameters()),
        "prunable_basic_transformer_blocks": block_count,
        "pruned_transformer_wrappers": wrapper_count,
    }

    rows: List[Dict[str, Any]] = []
    print(f"Generating dense and block-{args.block_id} variants for {args.num_prompts} prompts")
    for index, prompt in enumerate(prompts):
        seed = args.base_seed + index
        clear_blockdrop(pipe.unet)
        dense, dense_time = generate(pipe, prompt, seed, args)
        set_global_blockdrop(pipe.unet, args.block_id)
        block, block_time = generate(pipe, prompt, seed, args)
        rows.append(
            {
                "index": index,
                "seed": seed,
                "prompt": prompt,
                "dense_time": dense_time,
                "block_time": block_time,
                "dense_image": dense,
                "block_image": block,
                "block_vs_dense": metrics(dense, block),
            }
        )
        print(f"[{index+1:02d}/{args.num_prompts}] block SSIM={rows[-1]['block_vs_dense']['ssim']:.6f}")

    excluded = block_prefixes(pipe.unet, args.block_id)
    print(f"Applying CR-Diff-style pruning; excluding block prefixes: {excluded}")
    cr_analysis = prune_groupwise_magnitude(
        pipe.unet,
        {"down": args.down_ratio, "mid": args.mid_ratio, "up": args.up_ratio},
        excluded_prefixes=excluded,
        sample_per_layer=args.threshold_sample_per_layer,
        seed=args.base_seed,
    )
    print(json.dumps(cr_analysis, indent=2))

    print("Generating hybrid block-drop + CR-Diff variants")
    for row in rows:
        set_global_blockdrop(pipe.unet, args.block_id)
        hybrid, hybrid_time = generate(pipe, row["prompt"], row["seed"], args)
        dense = row["dense_image"]
        block = row["block_image"]
        row["hybrid_time"] = hybrid_time
        row["hybrid_vs_dense"] = metrics(dense, hybrid)
        row["hybrid_vs_block"] = metrics(block, hybrid)
        row["dense_b64"] = encode_jpeg(dense, args.jpeg_quality)
        row["block_b64"] = encode_jpeg(block, args.jpeg_quality)
        row["block_diff_b64"] = encode_jpeg(difference_image(dense, block), args.jpeg_quality)
        row["hybrid_b64"] = encode_jpeg(hybrid, args.jpeg_quality)
        row["hybrid_diff_b64"] = encode_jpeg(difference_image(dense, hybrid), args.jpeg_quality)
        print(
            f"[{row['index']+1:02d}/{args.num_prompts}] hybrid SSIM={row['hybrid_vs_dense']['ssim']:.6f}"
        )

    clear_blockdrop(pipe.unet)
    report = build_html(args, rows, cr_analysis, pipe.scheduler, component_counts)
    output.write_text(report, encoding="utf-8")
    block_summary = aggregate(rows, "block_vs_dense")
    hybrid_summary = aggregate(rows, "hybrid_vs_dense")
    print("=" * 80)
    print(f"Block-only mean SSIM: {block_summary['mean_ssim']:.6f}")
    print(f"Hybrid mean SSIM:     {hybrid_summary['mean_ssim']:.6f}")
    print(f"One-file report:      {output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
