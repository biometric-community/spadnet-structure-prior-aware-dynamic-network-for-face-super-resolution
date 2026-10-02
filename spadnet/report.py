"""Generate REPORT.md + SVG figures from real train/eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spadnet.plot_style import (
    BAR_COLOR,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FIGSIZE,
    LABEL_SIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    TITLE_SIZE,
    apply_rcparams,
    apply_style,
)

PAPER_TABLE_REF = {
    # Prose cites SPADNet 31.95 dB vs WSRNet 30.92 (×4 context). Exact Table I cells are image-only.
    "celeba_x4_psnr": 31.95,
    "note": "Table I numeric grid not OCR'd; use paper PDF for full baseline grid.",
}


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def plot_train_curves(history: dict, figdir: Path):
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    epochs = history.get("epochs", [])
    if not epochs:
        plt.close(fig)
        return None
    ax.plot(epochs, history["val_psnr"], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN, label="Val PSNR")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["val_ssim"], color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="Val SSIM")
    apply_style(ax)
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_ylabel("SSIM", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title("SPADNet training (Ours)", fontsize=TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    out = figdir / "train_val_curves.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_psnr_ssim_bars(eval_res: dict, figdir: Path, dataset: str, scale: int):
    apply_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, metric, ov, ylab in [
        (axes[0], "PSNR", eval_res["psnr"], "PSNR (dB)"),
        (axes[1], "SSIM", eval_res["ssim"], "SSIM"),
    ]:
        ax.bar([0], [ov], color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        ax.set_xticks([0])
        ax.set_xticklabels(["Ours"])
        ax.set_ylabel(ylab, fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_title(metric, fontsize=TITLE_SIZE, fontweight="bold")
        apply_style(ax, grid_axis="y")
    fig.suptitle(f"×{scale} FSR on {dataset} (Ours — real eval)", fontsize=TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    out = figdir / f"metrics_bar_{dataset}_x{scale}.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def write_report(cfg: dict, train_sum, eval_res, figures: list, report_path: Path, history=None):
    title = cfg.get("report", {}).get("title") or cfg["paper"]["title"]
    lines = [f"# {title}", "", "## Summary", ""]
    lines += [
        "| Run | PSNR (dB) | SSIM | N | Protocol |",
        "|-----|-----------|------|---|----------|",
    ]
    if eval_res:
        lines.append(
            f"| Eval ({eval_res.get('protocol', 'run')}) | {eval_res['psnr']:.3f} | "
            f"{eval_res['ssim']:.4f} | {eval_res.get('n', '—')} | "
            f"`{eval_res.get('protocol', cfg['train'].get('protocol', 'subset'))}` |"
        )
    if train_sum:
        lines.append(
            f"| Full train val (epoch {train_sum.get('epoch', '?')}) | "
            f"{train_sum['val_psnr']:.3f} | {train_sum['val_ssim']:.4f} | "
            f"— | `{train_sum.get('protocol', 'full')}` in progress |"
        )
    lines.append(
        f"| Paper Table I (×{cfg['model']['scale']}, prose) | see PDF "
        f"(×4≈{PAPER_TABLE_REF['celeba_x4_psnr']} dB) | see PDF | 1000 test | author protocol |"
    )
    lines += [
        "",
        f"_Scale ×{cfg['model']['scale']}; dataset `{cfg['data'].get('dataset', 'celeba')}`. "
        f"{PAPER_TABLE_REF['note']} Full protocol: {cfg['train'].get('epochs', '?')} epochs "
        f"(D3); refresh after completion._",
        "",
    ]
    if not eval_res and not train_sum:
        lines = [
            f"# {title}",
            "",
            "## Summary",
            "",
            "**BLOCKED:** Real dataset training required. No fabricated metrics.",
            "",
        ]

    size_note = ""
    size_path = Path("outputs/logs/dataset_size.json")
    if size_path.is_file():
        try:
            size_note = (
                f"- Size gate: **{json.loads(size_path.read_text()).get('total_gib', '?')} GiB** "
                f"total → `{json.loads(size_path.read_text()).get('decision', '?')}` "
                f"(`outputs/logs/dataset_size.json`)"
            )
        except Exception:
            size_note = ""

    lines += ["## Setup", ""]
    lines += [
        f"- Dataset: CelebA (`{cfg['data'].get('root', '')}`) + Helen "
        f"(`{cfg['data'].get('helen_root', '../../datasets/helen')}`)",
        f"- Scale: ×{cfg['model']['scale']} (LR → HR {cfg['model'].get('img_size', 128)}×"
        f"{cfg['model'].get('img_size', 128)})",
    ]
    if size_note:
        lines.append(size_note)
    lines += [
        f"- Protocol: `{cfg['train'].get('protocol', 'subset')}`",
        f"- Upstream refactor: {cfg.get('source_code', {}).get('url', 'N/A')} → package `spadnet`",
        "",
        "## Figures",
        "",
    ]
    if history and isinstance(history.get("epochs"), list) and history["epochs"]:
        last = history["epochs"][-1]
        lines.append(
            f"_Train curves through epoch {last}: "
            f"val PSNR={history['val_psnr'][-1]:.3f} dB, "
            f"SSIM={history['val_ssim'][-1]:.4f}._"
        )
        lines.append("")
    for fig in figures:
        if fig:
            rel = fig.relative_to(report_path.parent)
            lines.append(f"![{fig.stem}]({rel})")
            lines.append("")
    lines += [
        "## Regenerate",
        "",
        "```bash",
        "bash scripts/report.sh",
        "# After full train finishes:",
        "bash scripts/eval.sh --config configs/full.yaml --checkpoint outputs/checkpoints/full/latest.pt",
        "bash scripts/report.sh --config configs/full.yaml",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="SPADNet report")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--logs", default="outputs/logs")
    parser.add_argument("--figdir", default="outputs/figures")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_dir = Path(args.logs)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    train_sum = _load_json(log_dir / "train_summary.json")
    eval_res = _load_json(log_dir / "eval_results.json")
    history = _load_json(log_dir / "train_history.json") or {}

    figures = []
    if history.get("epochs"):
        figures.append(plot_train_curves(history, figdir))
    if eval_res:
        figures.append(
            plot_psnr_ssim_bars(
                eval_res,
                figdir,
                eval_res.get("dataset", "celeba"),
                int(eval_res.get("scale", cfg["model"]["scale"])),
            )
        )

    write_report(cfg, train_sum, eval_res, figures, Path("REPORT.md"), history=history)
    print("Wrote REPORT.md")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
