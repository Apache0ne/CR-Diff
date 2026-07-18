#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rescore an existing pure CR-Diff report without regenerating SDXL images.

The script reads metadata from the one-file HTML report, loads the PNGs retained
under its work directory, applies compatibility aliases required by older
ImageReward builds on modern Transformers, scores dense and CR images, and
writes a new self-contained HTML report.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-report", default="/content/pure_crdiff_cross_resolution_report.html")
    p.add_argument("--output", default="/content/pure_crdiff_cross_resolution_report_scored.html")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--jpeg-quality", type=int, default=88)
    return p.parse_args()


def extract_machine_report(report_path: Path) -> Dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    marker = '<section class="card"><h2>Embedded machine-readable report and errors</h2><pre>'
    start = text.find(marker)
    if start < 0:
        raise ValueError("Embedded machine-readable report section was not found")
    start += len(marker)
    end = text.find("</pre></section>", start)
    if end < 0:
        raise ValueError("Embedded machine-readable report closing tag was not found")
    payload = html.unescape(text[start:end])
    return json.loads(payload)


def patch_transformers_for_imagereward() -> Dict[str, str]:
    """Expose utilities at their legacy import path before ImageReward imports."""
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    patched: Dict[str, str] = {}
    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name):
            value = getattr(pytorch_utils, name)
            setattr(modeling_utils, name, value)
            patched[name] = "transformers.pytorch_utils"
    return patched


def normalize_score(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("ImageReward returned an empty score list")
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().reshape(-1)[0].item()
    return float(value)


def encode_jpeg(image: Image.Image, quality: int) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def difference_image(reference: Image.Image, candidate: Image.Image, gain: float = 4.0) -> Image.Image:
    a = np.asarray(reference.convert("RGB"), dtype=np.float32)
    b = np.asarray(candidate.convert("RGB"), dtype=np.float32)
    diff = np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def fmt(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def build_html(
    *,
    source: Dict[str, Any],
    rows: List[Dict[str, Any]],
    quality: int,
    aliases: Dict[str, str],
    error: str | None,
) -> str:
    valid = [row for row in rows if row.get("imagereward_delta") is not None]
    dense_scores = [row["dense_imagereward"] for row in valid]
    cr_scores = [row["cr_imagereward"] for row in valid]
    deltas = [row["imagereward_delta"] for row in valid]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(abs(delta) <= 1e-8 for delta in deltas)

    mean_dense = float(np.mean(dense_scores)) if dense_scores else None
    mean_cr = float(np.mean(cr_scores)) if cr_scores else None
    mean_delta = float(np.mean(deltas)) if deltas else None
    median_delta = float(np.median(deltas)) if deltas else None

    by_resolution: Dict[str, Dict[str, Any]] = {}
    for row in valid:
        key = f"{row['width']}x{row['height']}"
        by_resolution.setdefault(key, {"deltas": [], "wins": 0, "cases": 0})
        by_resolution[key]["deltas"].append(row["imagereward_delta"])
        by_resolution[key]["wins"] += int(row["imagereward_delta"] > 0)
        by_resolution[key]["cases"] += 1
    for value in by_resolution.values():
        value["mean_delta"] = float(np.mean(value.pop("deltas")))

    summary = {
        "mean_dense_imagereward": mean_dense,
        "mean_cr_imagereward": mean_cr,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "cr_wins": wins,
        "ties": ties,
        "cases": len(valid),
        "win_rate": float(wins / len(valid)) if valid else None,
        "by_resolution": by_resolution,
        "compatibility_aliases": aliases,
        "scoring_error": error,
    }

    table_rows: List[str] = []
    cards: List[str] = []
    ordered = sorted(rows, key=lambda row: row.get("imagereward_delta") if row.get("imagereward_delta") is not None else -math.inf)
    for row in ordered:
        pm = row.get("pixel_metrics", {})
        table_rows.append(
            "<tr>"
            f"<td>{row['index']}</td><td>{row['seed']}</td><td>{row['width']}×{row['height']}</td>"
            f"<td>{fmt(row.get('dense_imagereward'))}</td><td>{fmt(row.get('cr_imagereward'))}</td>"
            f"<td>{fmt(row.get('imagereward_delta'))}</td><td>{fmt(pm.get('psnr_db'), 3)}</td>"
            f"<td>{fmt(pm.get('mae_0_1'))}</td><td>{html.escape(row['prompt'])}</td></tr>"
        )
        dense = Image.open(row["dense_path"]).convert("RGB")
        cr = Image.open(row["cr_path"]).convert("RGB")
        cards.append(
            f'''<section class="card"><h3>Case {row['index']} — seed {row['seed']} — {row['width']}×{row['height']}</h3>
<p class="prompt">{html.escape(row['prompt'])}</p>
<p><b>ImageReward:</b> dense {fmt(row.get('dense_imagereward'))}, CR {fmt(row.get('cr_imagereward'))}, delta {fmt(row.get('imagereward_delta'))}</p>
<div class="images">
<figure><figcaption>Dense full model</figcaption><img src="data:image/jpeg;base64,{encode_jpeg(dense, quality)}"></figure>
<figure><figcaption>Pure CR-Diff full model</figcaption><img src="data:image/jpeg;base64,{encode_jpeg(cr, quality)}"></figure>
<figure><figcaption>Absolute difference ×4</figcaption><img src="data:image/jpeg;base64,{encode_jpeg(difference_image(dense, cr), quality)}"></figure>
</div></section>'''
        )

    machine = html.escape(json.dumps({"source_report": source, "rescored_summary": summary, "per_case": [
        {k: v for k, v in row.items() if k not in {"dense_path", "cr_path"}} for row in rows
    ]}, indent=2, default=str))

    status = "ImageReward scoring succeeded." if error is None else "ImageReward scoring failed; traceback is embedded below."
    win_text = "N/A" if not valid else f"{wins}/{len(valid)} ({wins/len(valid)*100:.1f}%)"
    resolution_rows = "".join(
        f"<tr><td>{key}</td><td>{value['mean_delta']:+.6f}</td><td>{value['wins']}/{value['cases']}</td></tr>"
        for key, value in by_resolution.items()
    )

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Pure CR-Diff rescored report</title>
<style>body{{font-family:Arial,sans-serif;background:#f4f5f7;color:#171717;margin:24px;line-height:1.4}}.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:18px;margin:16px 0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.metric{{background:#f7f7f8;border-radius:8px;padding:14px}}.value{{font-size:23px;font-weight:700}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}img{{width:100%;border:1px solid #bbb}}figure{{margin:0}}figcaption{{font-weight:700;margin-bottom:6px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccc;padding:7px;vertical-align:top}}th{{background:#eee}}.prompt{{font-style:italic}}pre{{white-space:pre-wrap;word-break:break-word;background:#111;color:#eee;padding:14px;overflow:auto}}@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Pure CR-Diff cross-resolution ImageReward report</h1>
<section class="card"><p>{status} No SDXL images were regenerated.</p></section>
<section class="card"><h2>Aggregate preference result</h2><div class="metrics">
<div class="metric"><div class="value">{fmt(mean_dense)}</div>Dense mean ImageReward</div>
<div class="metric"><div class="value">{fmt(mean_cr)}</div>CR mean ImageReward</div>
<div class="metric"><div class="value">{fmt(mean_delta)}</div>Mean CR − dense</div>
<div class="metric"><div class="value">{fmt(median_delta)}</div>Median CR − dense</div>
<div class="metric"><div class="value">{win_text}</div>CR wins</div>
</div></section>
<section class="card"><h2>Result by resolution</h2><table><thead><tr><th>Resolution</th><th>Mean CR − dense</th><th>CR wins</th></tr></thead><tbody>{resolution_rows}</tbody></table></section>
<section class="card"><h2>Per-case scores</h2><table><thead><tr><th>#</th><th>Seed</th><th>Resolution</th><th>Dense IR</th><th>CR IR</th><th>Delta</th><th>PSNR</th><th>MAE</th><th>Prompt</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
{''.join(cards)}
<section class="card"><h2>Embedded machine-readable report and errors</h2><pre>{machine}</pre></section>
</body></html>'''


def main() -> None:
    args = parse_args()
    input_report = Path(args.input_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    source = extract_machine_report(input_report)
    work_dir = Path(args.work_dir or source["arguments"]["work_dir"]).expanduser().resolve()

    rows: List[Dict[str, Any]] = []
    for item in source["per_case"]:
        row = dict(item)
        index = int(row["index"])
        row["dense_path"] = work_dir / "dense" / f"{index:03d}.png"
        row["cr_path"] = work_dir / "cr" / f"{index:03d}.png"
        if not row["dense_path"].is_file() or not row["cr_path"].is_file():
            raise FileNotFoundError(f"Missing retained images for case {index} under {work_dir}")
        rows.append(row)

    aliases: Dict[str, str] = {}
    error: str | None = None
    try:
        aliases = patch_transformers_for_imagereward()
        import ImageReward as RM

        print(f"Compatibility aliases: {aliases}", flush=True)
        print("Loading ImageReward-v1.0...", flush=True)
        model = RM.load("ImageReward-v1.0")
        with torch.inference_mode():
            for number, row in enumerate(rows, start=1):
                dense = normalize_score(model.score(row["prompt"], [str(row["dense_path"])]))
                cr = normalize_score(model.score(row["prompt"], [str(row["cr_path"])]))
                row["dense_imagereward"] = dense
                row["cr_imagereward"] = cr
                row["imagereward_delta"] = cr - dense
                print(f"[{number:02d}/{len(rows)}] dense={dense:.6f} CR={cr:.6f} delta={cr-dense:+.6f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        for row in rows:
            row["dense_imagereward"] = None
            row["cr_imagereward"] = None
            row["imagereward_delta"] = None

    report = build_html(source=source, rows=rows, quality=args.jpeg_quality, aliases=aliases, error=error)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"One-file rescored report: {output}", flush=True)
    if error is not None:
        raise RuntimeError("ImageReward rescoring failed; inspect the embedded traceback")


if __name__ == "__main__":
    main()
