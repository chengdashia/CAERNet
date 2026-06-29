from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

_ROOT_FOR_CACHE = Path(__file__).resolve().parents[2]
_MPLCONFIGDIR = _ROOT_FOR_CACHE / "tmp" / "mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ==========================================
# 路径配置
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "outputs" / "runs" / "A100" / "results"
FIG_DIR = ROOT / "paper" / "img"
SOURCE_DIR = ROOT / "paper" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# SCI 顶刊级别的 Matplotlib 全局配置
# ==========================================
mpl.rcParams.update(
    {
        # 字体规范 (Nature/Science 偏好 Arial 或 Helvetica)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,  # 保证 PDF 导出时字体可编辑
        "ps.fonttype": 42,

        # 字号规范 (通常单栏图的字号在 7pt - 9pt 之间)
        "font.size": 8,
        "axes.titlesize": 9,  # 子图标题稍大
        "axes.labelsize": 8.5,  # 轴标签字号
        "xtick.labelsize": 7.5,  # 刻度字号
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 8,

        # 轴线与刻度规范 (标准线宽通常为 0.8pt)
        "axes.linewidth": 0.8,
        "axes.spines.top": False,  # 去除顶部边框，提升现代感
        "axes.spines.right": False,  # 去除右侧边框
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",  # 刻度朝外
        "ytick.direction": "out",

        # 网格与背景
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "grid.color": "#B0B0B0",
        "grid.linestyle": "--",
        "axes.axisbelow": True,  # 网格线在图层最下方

        # 图像分辨率
        "figure.dpi": 300,
        "savefig.dpi": 600,  # 导出高分辨率用于出版
        "savefig.bbox": "tight",  # 自动裁剪白边
        "savefig.pad_inches": 0.05,
    }
)

# ==========================================
# 学术感配色方案 (色盲友好、高对比度)
# ==========================================
PALETTE = {
    "neutral": "#7F8C8D",  # 高级灰
    "blue": "#2980B9",  # 学术蓝
    "cyan": "#1ABC9C",  # 青色
    "teal": "#16A085",  # 深青色
    "orange": "#D35400",  # 强调橙
    "red": "#C0392B",  # 砖红
    "magenta": "#8E44AD",  # 稳重紫
    "grey": "#BDC3C7",  # 浅灰 (用于 Baseline)
    "black": "#2C3E50",  # 深色代替纯黑，更柔和
    "pale_blue": "#D4E6F1",
    "pale_teal": "#D1F2EB",
    "pale_orange": "#FADBD8",
}

# 文本映射配置保持不变
DISPLAY = {
    "revision_adaptation_linear_probe": "CLIP linear probe",
    "revision_adaptation_full_visual": "Full visual FT",
    "revision_clip_adapter_baseline": "CLIP-Adapter",
    "revision_adaptation_adapter_only": "Residual adapter",
    "revision_adaptation_adapter_partial_b2": "Adapter + partial FT",
    "revision_adaptation_partial_only_b2": "Partial FT only",
    "clip_mlssa_frozen": "MLSSA frozen",
    "clip_mlssa_single_final": "Single final layer",
    "clip_mlssa_single_final_mean_only": "Final mean only",
    "clip_mlssa_no_gate": "MLSSA no gate",
    "clip_mlssa_uniform_fusion": "MLSSA uniform fusion",
    "clip_mlssa_full": "MLSSA + orth.",
    "clip_mlssa_no_orth": "MLSSA",
    "clip_mlssa_layers_8_12": "Layers 8+12",
    "clip_mlssa_layers_even": "Even layers",
}


# 数据加载与处理逻辑保持不变
def _base_from_run(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(RESULT_ROOT.rglob("test_metrics.json")):
        cfg_path = metrics_path.parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        metrics = json.loads(metrics_path.read_text())
        run_name = cfg.get("output", {}).get("run_name") or metrics_path.parent.name
        base = _base_from_run(run_name)
        seed = cfg.get("seed", cfg.get("train", {}).get("seed"))
        if seed is None:
            match = re.search(r"_seed(\d+)$", run_name) or re.search(
                r"_seed(\d+)$", metrics_path.parent.name
            )
            seed = int(match.group(1)) if match else None

        rows.append(
            {
                "base": base,
                "run_name": run_name,
                "seed": seed,
                "path": metrics_path.parent,
                "accuracy": metrics["accuracy"] * 100.0,
                "macro_f1": metrics["macro_f1"] * 100.0,
                "ece": metrics["ece"] * 100.0,
                "loss": metrics["loss"],
                "style_weights": {
                    key: value
                    for key, value in metrics.items()
                    if key.startswith("style_layer_weight_")
                },
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["base"]].append(row)

    summary = {}
    for base, items in groups.items():
        summary[base] = {
            "label": DISPLAY.get(base, base),
            "n": len(items),
            "seeds": sorted([x["seed"] for x in items if x["seed"] is not None]),
            "acc_mean": float(np.mean([x["accuracy"] for x in items])),
            "acc_std": float(np.std([x["accuracy"] for x in items], ddof=1)) if len(items) > 1 else 0.0,
            "f1_mean": float(np.mean([x["macro_f1"] for x in items])),
            "f1_std": float(np.std([x["macro_f1"] for x in items], ddof=1)) if len(items) > 1 else 0.0,
            "ece_mean": float(np.mean([x["ece"] for x in items])),
            "ece_std": float(np.std([x["ece"] for x in items], ddof=1)) if len(items) > 1 else 0.0,
            "items": items,
        }
    return summary


def write_source_tables(rows: list[dict], summary: dict[str, dict]) -> None:
    with (SOURCE_DIR / "mlssa_run_level_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["base", "label", "seed", "accuracy", "macro_f1", "ece", "path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "base": row["base"], "label": DISPLAY.get(row["base"], row["base"]),
                "seed": row["seed"], "accuracy": f"{row['accuracy']:.4f}",
                "macro_f1": f"{row['macro_f1']:.4f}", "ece": f"{row['ece']:.4f}", "path": row["path"].relative_to(ROOT)
            })

    with (SOURCE_DIR / "mlssa_summary_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["base", "label", "n", "seeds", "acc_mean", "acc_std", "f1_mean", "f1_std",
                                            "ece_mean", "ece_std"])
        writer.writeheader()
        for base, item in sorted(summary.items()):
            writer.writerow({
                "base": base, "label": item["label"], "n": item["n"],
                "seeds": " ".join(map(str, item["seeds"])), "acc_mean": f"{item['acc_mean']:.4f}",
                "acc_std": f"{item['acc_std']:.4f}", "f1_mean": f"{item['f1_mean']:.4f}",
                "f1_std": f"{item['f1_std']:.4f}", "ece_mean": f"{item['ece_mean']:.4f}",
                "ece_std": f"{item['ece_std']:.4f}"
            })


# ==========================================
# 绘图辅助函数
# ==========================================
def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save the manuscript figure in editable and preview formats."""
    for suffix, kwargs in {
        "png": {"dpi": 600, "transparent": False},
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", **kwargs)


def panel_label(ax, label: str) -> None:
    """标准化的子图标签 (a, b, c...)，加粗显示"""
    ax.text(
        -0.15, 1.05, label, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="bottom", ha="left"
    )


def add_grid(ax, axis="x") -> None:
    ax.grid(axis=axis)


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pct(value: str | float) -> float:
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def method_family(base: str) -> str:
    if base.startswith("clip_mlssa"):
        return "mlssa"
    if base.startswith("revision_"):
        return "clip"
    return "other"


def pretty_class_name(name: str) -> str:
    return name.replace("_", " ").replace("post impressionism", "post-imp.").title()


# ==========================================
# 核心绘图逻辑 (重构布局与视觉映射)
# ==========================================
def plot_main_comparison_single(summary: dict[str, dict]) -> None:
    records = [
        ("clip_zeroshot", "Zero-shot CLIP", 49.25, 0.0, 48.13, 0.0, "zero"),
        ("resnet50", "ResNet50", 68.11, 0.0, 67.84, 0.0, "cnn"),
        ("ca_resnet50", "CA-ResNet50", 67.81, 0.0, 67.53, 0.0, "cnn"),
        ("ms_caernet", "MS-CAERNet", 69.13, 0.0, 68.89, 0.0, "cnn"),
        ("efficientnetv2_s", "EffNetV2-S", 69.92, 0.0, 69.67, 0.0, "cnn"),
        ("convnext_tiny", "ConvNeXt-T", 71.47, 0.0, 71.36, 0.0, "cnn"),
    ]
    for base in [
        "revision_adaptation_linear_probe",
        "revision_clip_adapter_baseline",
        "revision_adaptation_adapter_only",
        "revision_adaptation_partial_only_b2",
        "clip_mlssa_no_orth",
    ]:
        if base not in summary:
            continue
        item = summary[base]
        family = "ours" if base == "clip_mlssa_no_orth" else "clip"
        records.append(
            (
                base, item["label"].replace("Partial FT only", "Partial FT"),
                item["acc_mean"], item["acc_std"], item["f1_mean"], item["f1_std"], family
            )
        )

    records.sort(key=lambda item: item[2])
    labels = [x[1] for x in records]
    y = np.arange(len(records))
    families = [x[6] for x in records]

    family_color = {
        "zero": "#8A8F98",
        "cnn": "#6E7781",
        "clip": "#2F6FAE",
        "ours": "#B6423C",
    }
    family_soft = {
        "zero": "#E9ECEF",
        "cnn": "#DFE3E8",
        "clip": "#DCEAF7",
        "ours": "#F5D9D6",
    }
    gain_green = "#2E8B57"
    neutral_text = "#2F3437"
    x_min = 47.0
    x_max = max(x[2] for x in records) + 1.7

    fig = plt.figure(figsize=(3.55, 5.15), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.7, 1.25])
    ax_rank = fig.add_subplot(gs[0])
    ax_gain = fig.add_subplot(gs[1])

    mlssa_idx = labels.index("MLSSA") if "MLSSA" in labels else None
    if mlssa_idx is not None:
        ax_rank.axhspan(
            mlssa_idx - 0.46,
            mlssa_idx + 0.46,
            color=family_soft["ours"],
            alpha=0.75,
            zorder=0,
        )

    metric_offset = 0.11
    for i, (_, label, acc_value, acc_sd, f1_value, f1_sd, family) in enumerate(records):
        color = family_color[family]
        ax_rank.plot(
            [x_min, acc_value],
            [i, i],
            color=family_soft[family],
            lw=4.0 if family != "ours" else 5.5,
            solid_capstyle="round",
            zorder=1,
        )
        if acc_sd > 0:
            ax_rank.errorbar(
                acc_value,
                i + metric_offset,
                xerr=acc_sd,
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.4,
                capthick=0.9,
                zorder=3,
            )
        if f1_sd > 0:
            ax_rank.errorbar(
                f1_value,
                i - metric_offset,
                xerr=f1_sd,
                fmt="none",
                ecolor="#9AA3AD",
                elinewidth=0.75,
                capsize=2.0,
                capthick=0.75,
                zorder=2,
            )
        ax_rank.scatter(
            f1_value,
            i - metric_offset,
            marker="D",
            s=20,
            facecolor="white",
            edgecolor="#4D5963",
            linewidth=0.75,
            zorder=4,
        )
        ax_rank.scatter(
            acc_value,
            i + metric_offset,
            marker="o",
            s=48 if family == "ours" else 30,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
        )

        should_label = family == "ours" or acc_value >= 72.0 or family == "zero"
        if should_label:
            ax_rank.text(
                acc_value + 0.38,
                i + metric_offset,
                f"{acc_value:.2f}" if family == "ours" else f"{acc_value:.1f}",
                va="center",
                ha="left",
                fontsize=7.2,
                fontweight="bold" if family == "ours" else "normal",
                color=color if family == "ours" else neutral_text,
            )

    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels)
    for tick, family in zip(ax_rank.get_yticklabels(), families):
        tick.set_fontsize(7.2)
        if family == "ours":
            tick.set_fontweight("bold")
            tick.set_color(family_color["ours"])
        elif family == "clip":
            tick.set_color("#1F5D99")

    ax_rank.set_xlim(x_min, x_max)
    ax_rank.set_xlabel("Held-out test performance (%)")
    ax_rank.set_title("Held-out performance ranking", loc="left", pad=6, fontweight="bold")
    ax_rank.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax_rank.spines["left"].set_visible(False)
    ax_rank.tick_params(axis="y", length=0, pad=2)
    ax_rank.scatter([], [], marker="o", s=30, facecolor="#2F6FAE", edgecolor="white", label="Accuracy")
    ax_rank.scatter([], [], marker="D", s=20, facecolor="white", edgecolor="#4D5963", label="Macro-F1")
    ax_rank.legend(
        loc="lower right",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.2,
        fontsize=7,
    )

    by_base = {base: record for base, *record in records}
    mlssa_acc = by_base.get("clip_mlssa_no_orth", [None, None, None])[1]
    gain_targets = [
        ("revision_adaptation_linear_probe", "Linear probe"),
        ("revision_clip_adapter_baseline", "CLIP-Adapter"),
        ("revision_adaptation_adapter_only", "Residual adapter"),
        ("revision_adaptation_partial_only_b2", "Partial FT"),
    ]
    gain_rows = []
    if mlssa_acc is not None:
        for base, short_label in gain_targets:
            if base in by_base:
                acc_value = by_base[base][1]
                gain_rows.append((short_label, mlssa_acc - acc_value))

    gain_rows = gain_rows[::-1]
    gain_y = np.arange(len(gain_rows))
    gains = [row[1] for row in gain_rows]
    gain_colors = [gain_green if value >= 0.20 else "#9AA3AD" for value in gains]
    ax_gain.axvline(0, color="#5D666F", linewidth=0.8, zorder=1)
    ax_gain.barh(
        gain_y,
        gains,
        height=0.55,
        color=gain_colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    for yi, value in zip(gain_y, gains):
        label_x = value + (0.14 if value >= 0 else -0.14)
        ax_gain.text(
            label_x,
            yi,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7.2,
            color=neutral_text,
            fontweight="bold" if value >= 1.0 else "normal",
        )
    ax_gain.set_yticks(gain_y)
    ax_gain.set_yticklabels([row[0] for row in gain_rows], fontsize=7.2)
    ax_gain.set_xlabel("MLSSA accuracy gain (percentage points)")
    ax_gain.set_title("MLSSA gain over key CLIP baselines", loc="left", pad=5, fontweight="bold")
    ax_gain.set_xlim(-0.35, max(gains + [0.5]) + 0.85)
    ax_gain.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax_gain.spines["left"].set_visible(False)
    ax_gain.tick_params(axis="y", length=0, pad=2)

    panel_label(ax_rank, "a")
    panel_label(ax_gain, "b")

    save_figure(fig, "fig_results_main_comparison_single")
    plt.close(fig)


def plot_ablation_single(summary: dict[str, dict]) -> None:
    bases = [
        "clip_mlssa_frozen", "clip_mlssa_single_final", "clip_mlssa_single_final_mean_only",
        "clip_mlssa_no_gate", "clip_mlssa_uniform_fusion", "clip_mlssa_full", "clip_mlssa_no_orth",
    ]
    labels = ["Frozen", "Final", "Mean\nonly", "No gate", "Uniform", "+ orth.", "MLSSA"]
    colors = [PALETTE["grey"], PALETTE["blue"], PALETTE["pale_blue"], PALETTE["magenta"], PALETTE["teal"],
              PALETTE["neutral"], PALETTE["red"]]

    x = np.arange(len(bases))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.8), sharex=True, layout="constrained")

    # 配置条形图样式: 添加边框色和适当宽度
    bar_kwargs = {"edgecolor": "white", "linewidth": 0.8, "width": 0.65, "capsize": 3,
                  "error_kw": {"elinewidth": 1.0, "capthick": 1.0}}

    acc = [summary[b]["acc_mean"] for b in bases]
    acc_err = [summary[b]["acc_std"] for b in bases]
    bars = axes[0].bar(x, acc, yerr=acc_err, color=colors, **bar_kwargs)
    axes[0].set_ylim(72.0, 74.8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Recognition effect", loc="left", pad=8)
    add_grid(axes[0], "y")

    # 在最重要的柱子上加高亮数值
    for i, bar in enumerate(bars):
        if i == len(bars) - 1:  # MLSSA
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}",
                         ha='center', va='bottom', fontsize=7, fontweight='bold', color=colors[-1])

    ece = [summary[b]["ece_mean"] for b in bases]
    ece_err = [summary[b]["ece_std"] for b in bases]
    axes[1].bar(x, ece, yerr=ece_err, color=colors, **bar_kwargs)
    axes[1].set_ylabel("ECE (%)")
    axes[1].set_title("Calibration side effect", loc="left", pad=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    add_grid(axes[1], "y")

    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    save_figure(fig, "fig_results_ablation_single")
    plt.close(fig)


def aggregate_history(base: str, summary: dict[str, dict], column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for item in summary[base]["items"]:
        history_path = item["path"] / "history.csv"
        if not history_path.exists():
            continue
        with history_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get(column) in (None, ""):
                    continue
                by_epoch[int(float(row["epoch"]))].append(float(row[column]) * 100.0)
    epochs = sorted(by_epoch)
    means = [float(np.mean(by_epoch[e])) for e in epochs]
    stds = [float(np.std(by_epoch[e], ddof=1)) if len(by_epoch[e]) > 1 else 0.0 for e in epochs]
    return np.asarray(epochs), np.asarray(means), np.asarray(stds)


def plot_training_curves_single(summary: dict[str, dict]) -> None:
    bases = [
        "revision_adaptation_linear_probe", "revision_clip_adapter_baseline",
        "revision_adaptation_partial_only_b2", "clip_mlssa_no_orth",
    ]
    colors = [PALETTE["grey"], PALETTE["blue"], PALETTE["teal"], PALETTE["red"]]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True, layout="constrained")
    for base, color in zip(bases, colors):
        epochs, mean, std = aggregate_history(base, summary, "val_accuracy")
        if len(epochs) == 0: continue

        # 增加线的粗细与填充的平滑度
        axes[0].plot(epochs, mean, color=color, lw=1.8, label=summary[base]["label"], zorder=3)
        axes[0].fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0, zorder=2)

        epochs_loss, loss_mean, loss_std = aggregate_history(base, summary, "val_loss")
        if len(epochs_loss) > 0:
            axes[1].plot(epochs_loss, loss_mean / 100.0, color=color, lw=1.8, label=summary[base]["label"], zorder=3)

    axes[0].set_ylabel("Validation\naccuracy (%)")
    axes[0].set_ylim(62, 76)
    axes[0].set_title("Validation accuracy", loc="left", pad=8)
    add_grid(axes[0], "y")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation\nloss")
    axes[1].set_title("Validation loss", loc="left", pad=8)
    add_grid(axes[1], "y")

    # 优化图例：去边框，增加透明度
    axes[1].legend(loc="upper right", frameon=False, fontsize=7)

    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    save_figure(fig, "fig_results_training_curves_single")
    plt.close(fig)


def plot_layer_weights_single(summary: dict[str, dict]) -> None:
    bases = ["clip_mlssa_frozen", "clip_mlssa_no_gate", "clip_mlssa_full", "clip_mlssa_no_orth"]
    labels = ["Frozen", "No gate", "+ orth.", "MLSSA"]
    layer_labels = ["Layer 4", "Layer 8", "Layer 12"]
    data = []
    for base in bases:
        values = []
        for idx in range(3):
            key = f"style_layer_weight_{idx}"
            per_seed = [item["style_weights"][key] for item in summary[base]["items"] if key in item["style_weights"]]
            values.append(float(np.mean(per_seed)) if per_seed else math.nan)
        data.append(values)
    data = np.asarray(data)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    left = np.zeros(len(bases))

    # 堆叠图颜色选取同一色系或对比鲜明的颜色
    colors = ["#A9CCE3", "#2980B9", "#1A5276"]
    y = np.arange(len(bases))

    for idx, layer in enumerate(layer_labels):
        ax.barh(y, data[:, idx], left=left, color=colors[idx], edgecolor="white", linewidth=0.8, label=layer,
                height=0.6)
        # 为深色背景上的文字使用白色提高对比度
        text_color = "white" if idx > 0 else "#2C3E50"
        for i, value in enumerate(data[:, idx]):
            ax.text(left[i] + value / 2, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color,
                    fontweight="medium")
        left += data[:, idx]

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average fusion weight")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    # 优化图例并放置在上方
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7)

    add_grid(ax, "x")
    save_figure(fig, "fig_results_layer_weights_single")
    plt.close(fig)


def plot_seed_stability(summary: dict[str, dict]) -> None:
    bases = [
        "revision_adaptation_linear_probe",
        "revision_clip_adapter_baseline",
        "revision_adaptation_adapter_only",
        "revision_adaptation_adapter_partial_b2",
        "revision_adaptation_partial_only_b2",
        "clip_mlssa_frozen",
        "clip_mlssa_no_gate",
        "clip_mlssa_uniform_fusion",
        "clip_mlssa_no_orth",
    ]
    bases = [base for base in bases if base in summary and summary[base]["n"] >= 3]
    labels = [
        summary[base]["label"].replace("Adapter + partial FT", "Adapter + partial")
        .replace("CLIP linear probe", "Linear probe")
        .replace("MLSSA uniform fusion", "Uniform fusion")
        for base in bases
    ]
    y = np.arange(len(bases))
    colors = [
        "#B6BDC7" if method_family(base) == "clip" else "#CF6B63"
        for base in bases
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.35), sharey=True, layout="constrained")
    metrics = [
        ("accuracy", "Accuracy across seeds (%)", "acc_mean", "acc_std", (65.5, 75.0)),
        ("ece", "ECE across seeds (%)", "ece_mean", "ece_std", (0.0, 14.0)),
    ]
    for ax, (metric, title, mean_key, std_key, xlim) in zip(axes, metrics):
        for i, base in enumerate(bases):
            item = summary[base]
            values = [row[metric] for row in item["items"]]
            seeds = [row["seed"] for row in item["items"]]
            offsets = np.linspace(-0.16, 0.16, len(values))
            ax.errorbar(
                item[mean_key],
                i,
                xerr=item[std_key],
                fmt="none",
                ecolor="#4E5965",
                elinewidth=0.9,
                capsize=2.5,
                zorder=2,
            )
            ax.scatter(
                values,
                i + offsets,
                s=36,
                color=colors[i],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            if metric == "accuracy":
                for value, seed, dy in zip(values, seeds, offsets):
                    ax.text(
                        value + 0.08,
                        i + dy,
                        str(seed),
                        va="center",
                        ha="left",
                        fontsize=6.1,
                        color="#555D66",
                    )
        ax.set_xlim(*xlim)
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        ax.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Held-out test accuracy (%)")
    axes[1].set_xlabel("Expected calibration error (%)")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save_figure(fig, "fig_results_seed_stability")
    plt.close(fig)

    with (SOURCE_DIR / "table_seed_stability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "n", "seeds", "acc_mean", "acc_std", "ece_mean", "ece_std"],
        )
        writer.writeheader()
        for base in bases:
            item = summary[base]
            writer.writerow(
                {
                    "method": item["label"],
                    "n": item["n"],
                    "seeds": " ".join(map(str, item["seeds"])),
                    "acc_mean": f"{item['acc_mean']:.4f}",
                    "acc_std": f"{item['acc_std']:.4f}",
                    "ece_mean": f"{item['ece_mean']:.4f}",
                    "ece_std": f"{item['ece_std']:.4f}",
                }
            )


def plot_accuracy_calibration_tradeoff(summary: dict[str, dict]) -> None:
    bases = [
        "revision_adaptation_linear_probe",
        "revision_adaptation_full_visual",
        "revision_clip_adapter_baseline",
        "revision_adaptation_adapter_only",
        "revision_adaptation_adapter_partial_b2",
        "revision_adaptation_partial_only_b2",
        "clip_mlssa_frozen",
        "clip_mlssa_single_final",
        "clip_mlssa_single_final_mean_only",
        "clip_mlssa_no_gate",
        "clip_mlssa_uniform_fusion",
        "clip_mlssa_full",
        "clip_mlssa_no_orth",
    ]
    bases = [base for base in bases if base in summary]
    colors = {
        "clip": "#3775BA",
        "mlssa": "#B64342",
        "other": "#A8A8A8",
    }
    fig, ax = plt.subplots(figsize=(3.55, 3.05), layout="constrained")
    for base in bases:
        item = summary[base]
        family = method_family(base)
        marker = "*" if base == "clip_mlssa_no_orth" else "o"
        size = 115 if base == "clip_mlssa_no_orth" else 45
        ax.errorbar(
            item["ece_mean"],
            item["acc_mean"],
            xerr=item["ece_std"],
            yerr=item["acc_std"],
            fmt="none",
            ecolor="#A5ADB7",
            elinewidth=0.75,
            capsize=2.0,
            zorder=1,
        )
        ax.scatter(
            item["ece_mean"],
            item["acc_mean"],
            marker=marker,
            s=size,
            color=colors[family],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
    label_bases = {
        "revision_adaptation_full_visual": (0.25, 0.10, "left"),
        "revision_clip_adapter_baseline": (0.22, 0.10, "left"),
        "clip_mlssa_frozen": (0.18, -0.22, "left"),
        "clip_mlssa_no_orth": (0.22, 0.22, "left"),
    }
    for base, (dx, dy, ha) in label_bases.items():
        if base not in summary:
            continue
        item = summary[base]
        ax.text(
            item["ece_mean"] + dx,
            item["acc_mean"] + dy,
            item["label"].replace("Partial FT only", "Partial FT").replace("MLSSA + orth.", "+ orth."),
            fontsize=6.8,
            color="#30363D",
            ha=ha,
        )
    ax.set_xlabel("Expected calibration error (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy-calibration trade-off", loc="left", fontweight="bold", pad=6)
    ax.grid(color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax.set_xlim(1.0, 39.0)
    ax.set_ylim(65.5, 75.0)
    save_figure(fig, "fig_results_accuracy_calibration")
    plt.close(fig)


def plot_per_class_profile() -> None:
    per_class_path = SOURCE_DIR / "table_per_class_metrics.csv"
    pred_path = SOURCE_DIR / "artvla_test_predictions.csv"
    class_path = SOURCE_DIR / "artvla_test_predictions.classes.txt"
    metadata_path = SOURCE_DIR / "artvla_test_predictions.metadata.json"
    if not (per_class_path.exists() and pred_path.exists() and class_path.exists()):
        return

    rows = read_csv_dicts(per_class_path)
    class_names = [line.strip() for line in class_path.read_text().splitlines() if line.strip()]
    label_names = [pretty_class_name(name) for name in class_names]
    f1 = [pct(row["f1"]) for row in rows]
    precision = [pct(row["precision"]) for row in rows]
    recall = [pct(row["recall"]) for row in rows]

    n = len(class_names)
    confusion = np.zeros((n, n), dtype=float)
    for row in read_csv_dicts(pred_path):
        confusion[int(row["y_true"]), int(row["y_pred"])] += 1
    normalized = confusion / confusion.sum(axis=1, keepdims=True) * 100.0

    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    fig = plt.figure(figsize=(7.15, 3.9), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12])
    ax_bar = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    order = np.argsort(f1)
    y = np.arange(n)
    sorted_labels = [label_names[i] for i in order]
    sorted_f1 = [f1[i] for i in order]
    sorted_precision = [precision[i] for i in order]
    sorted_recall = [recall[i] for i in order]
    bar_colors = ["#B64342" if value < 65 else "#3775BA" for value in sorted_f1]
    ax_bar.barh(y, sorted_f1, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.62, label="F1")
    ax_bar.scatter(sorted_precision, y, marker="|", s=110, color="#272727", linewidth=1.1, label="Precision")
    ax_bar.scatter(sorted_recall, y, marker="o", s=20, facecolor="white", edgecolor="#272727", linewidth=0.8, label="Recall")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(sorted_labels, fontsize=7.0)
    ax_bar.set_xlim(50, 100.5)
    ax_bar.set_xlabel("Class-level metric (%)")
    ax_bar.set_title("Per-class recognition profile", loc="left", fontweight="bold", pad=6)
    ax_bar.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax_bar.legend(loc="lower right", fontsize=6.6, handletextpad=0.35)

    im = ax_heat.imshow(normalized, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_xticklabels(label_names, rotation=45, ha="right", fontsize=6.4)
    ax_heat.set_yticklabels(label_names, fontsize=6.4)
    ax_heat.set_xlabel("Predicted style")
    ax_heat.set_ylabel("True style")
    acc_note = f"checkpoint acc. {float(metadata.get('accuracy', 0))*100:.2f}%"
    ax_heat.set_title(
        f"Row-normalized confusion matrix\nRepresentative {acc_note}",
        loc="left",
        fontweight="bold",
        pad=6,
        fontsize=8.3,
    )
    for i in range(n):
        for j in range(n):
            val = normalized[i, j]
            if i == j or val >= 8.0:
                ax_heat.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.7,
                    color="white" if val >= 50 else "#1F2933",
                )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.set_label("Row percentage")
    panel_label(ax_bar, "a")
    panel_label(ax_heat, "b")
    save_figure(fig, "fig_results_per_class_confusion")
    plt.close(fig)

    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j and normalized[i, j] > 0:
                off_diag.append((normalized[i, j], label_names[i], label_names[j]))
    off_diag.sort(reverse=True)
    with (SOURCE_DIR / "table_top_confusions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true_class", "predicted_class", "row_percent"])
        writer.writeheader()
        for value, true_label, pred_label in off_diag[:12]:
            writer.writerow(
                {
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "row_percent": f"{value:.2f}",
                }
            )


def plot_compute_profile() -> None:
    path = SOURCE_DIR / "table_compute_profile.csv"
    if not path.exists():
        return
    rows = read_csv_dicts(path)
    labels = [row["method"].replace("EfficientNetV2-S", "EffNetV2-S").replace("ArtVLA-Adapter", "Adapter profile") for row in rows]
    params = np.asarray([float(row["params_m"]) for row in rows])
    trainable = np.asarray([float(row["trainable_params_m"]) for row in rows])
    flops = np.asarray([float(row["flops_g"]) for row in rows])
    epochs = np.asarray([float(row["completed_epochs"]) for row in rows])
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharey=True, layout="constrained")
    axes[0].barh(y, params, color="#D8DDE6", edgecolor="white", height=0.62, label="Total")
    axes[0].barh(y, trainable, color="#3775BA", edgecolor="white", height=0.36, label="Trainable")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parameters (M, log scale)")
    axes[0].set_title("Parameter footprint", loc="left", fontweight="bold", pad=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.0)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=6.8)

    ax2 = axes[1]
    ax2.barh(y - 0.16, flops, color="#CF6B63", edgecolor="white", height=0.28, label="FLOPs")
    ax2_epoch = ax2.twiny()
    ax2_epoch.plot(epochs, y + 0.16, marker="o", color="#272727", lw=1.2, ms=3.5, label="Completed epochs")
    ax2.set_xlabel("FLOPs (G)")
    ax2_epoch.set_xlabel("Completed epochs")
    ax2.set_title("Compute and training length", loc="left", fontweight="bold", pad=6)
    ax2.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax2_epoch.set_xlim(0, max(epochs) * 1.12)
    lines, names = ax2.get_legend_handles_labels()
    lines2, names2 = ax2_epoch.get_legend_handles_labels()
    ax2.legend(lines + lines2, names + names2, loc="lower right", fontsize=6.8)
    for ax in axes:
        ax.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save_figure(fig, "fig_results_compute_profile")
    plt.close(fig)


def plot_main_comparison_simple(summary: dict[str, dict]) -> None:
    records = [
        ("CLIP zero-shot", 49.25, 0.0, "zero"),
        ("ResNet50", 68.11, 0.0, "cnn"),
        ("CA-ResNet50", 67.81, 0.0, "cnn"),
        ("MS-CAERNet", 69.13, 0.0, "cnn"),
        ("EffNetV2-S", 69.92, 0.0, "cnn"),
        ("ConvNeXt-Tiny", 71.47, 0.0, "cnn"),
    ]
    for base in [
        "revision_adaptation_linear_probe",
        "revision_clip_adapter_baseline",
        "revision_adaptation_adapter_only",
        "revision_adaptation_adapter_partial_b2",
        "revision_adaptation_partial_only_b2",
        "clip_mlssa_no_orth",
    ]:
        if base not in summary:
            continue
        item = summary[base]
        label = item["label"].replace("Partial FT only", "Partial FT")
        family = "ours" if base == "clip_mlssa_no_orth" else "clip"
        records.append((label, item["acc_mean"], item["acc_std"], family))

    records.sort(key=lambda row: row[1])
    labels = [row[0] for row in records]
    acc = np.asarray([row[1] for row in records])
    err = np.asarray([row[2] for row in records])
    family = [row[3] for row in records]
    colors = {
        "zero": "#C9CED6",
        "cnn": "#D8DDE6",
        "clip": "#7EAAD3",
        "ours": "#C93F36",
    }

    fig, ax = plt.subplots(figsize=(3.55, 4.25), layout="constrained")
    y = np.arange(len(records))
    bars = ax.barh(
        y,
        acc,
        xerr=err,
        color=[colors[f] for f in family],
        edgecolor="white",
        linewidth=0.6,
        height=0.62,
        capsize=2.4,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlim(47.0, 75.5)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Main ArtBench-10 comparison", loc="left", fontweight="bold", pad=6)
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value, fam in zip(bars, acc, family):
        ax.text(
            value + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}" if fam == "ours" else f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=6.9,
            fontweight="bold" if fam == "ours" else "normal",
            color="#30363D",
        )
    save_figure(fig, "fig_results_main_comparison_single")
    plt.close(fig)


def plot_ablation_simple(summary: dict[str, dict]) -> None:
    bases = [
        "clip_mlssa_frozen",
        "clip_mlssa_single_final",
        "clip_mlssa_single_final_mean_only",
        "clip_mlssa_no_gate",
        "clip_mlssa_uniform_fusion",
        "clip_mlssa_full",
        "clip_mlssa_no_orth",
    ]
    labels = ["Frozen", "Final layer", "Mean only", "No gate", "Uniform fusion", "+ orth.", "MLSSA"]
    bases = [base for base in bases if base in summary]
    labels = labels[: len(bases)]
    acc = np.asarray([summary[base]["acc_mean"] for base in bases])
    acc_err = np.asarray([summary[base]["acc_std"] for base in bases])
    ece = np.asarray([summary[base]["ece_mean"] for base in bases])
    ece_err = np.asarray([summary[base]["ece_std"] for base in bases])
    y = np.arange(len(bases))
    colors = ["#D8DDE6"] * len(bases)
    if colors:
        colors[-1] = "#C93F36"

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True, layout="constrained")
    axes[0].barh(y, acc, xerr=acc_err, color=colors, edgecolor="white", height=0.62, capsize=2.4)
    axes[0].set_xlim(72.0, 74.8)
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_title("Ablation accuracy", loc="left", fontweight="bold", pad=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.0)
    axes[0].invert_yaxis()

    axes[1].barh(y, ece, xerr=ece_err, color=colors, edgecolor="white", height=0.62, capsize=2.4)
    axes[1].set_xlim(0, 12.5)
    axes[1].set_xlabel("ECE (%)")
    axes[1].set_title("Calibration", loc="left", fontweight="bold", pad=6)
    for ax in axes:
        ax.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save_figure(fig, "fig_results_ablation_single")
    plt.close(fig)


def plot_per_class_f1_simple() -> None:
    path = SOURCE_DIR / "table_per_class_metrics.csv"
    if not path.exists():
        return
    rows = read_csv_dicts(path)
    values = [(pretty_class_name(row["class"]), pct(row["f1"])) for row in rows]
    values.sort(key=lambda row: row[1])
    labels = [row[0] for row in values]
    f1 = np.asarray([row[1] for row in values])
    y = np.arange(len(values))
    colors = ["#C93F36" if value < 65 else "#3775BA" for value in f1]

    fig, ax = plt.subplots(figsize=(3.55, 3.15), layout="constrained")
    bars = ax.barh(y, f1, color=colors, edgecolor="white", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlim(50, 100)
    ax.set_xlabel("F1 (%)")
    ax.set_title("Per-class F1 of MLSSA", loc="left", fontweight="bold", pad=6)
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, f1):
        ax.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=6.7)
    save_figure(fig, "fig_results_per_class_f1")
    plt.close(fig)


def plot_training_curves_simple(summary: dict[str, dict]) -> None:
    bases = [
        "revision_adaptation_linear_probe",
        "revision_clip_adapter_baseline",
        "revision_adaptation_partial_only_b2",
        "clip_mlssa_no_orth",
    ]
    colors = ["#B6BDC7", "#3775BA", "#1E9D84", "#C93F36"]
    fig, ax = plt.subplots(figsize=(3.55, 2.8), layout="constrained")
    for base, color in zip(bases, colors):
        if base not in summary:
            continue
        epochs, mean, std = aggregate_history(base, summary, "val_accuracy")
        if len(epochs) == 0:
            continue
        label = summary[base]["label"].replace("Partial FT only", "Partial FT")
        ax.plot(epochs, mean, color=color, lw=1.5, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(62, 76)
    ax.set_title("Validation accuracy trajectory", loc="left", fontweight="bold", pad=6)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.45, linestyle="-", alpha=0.9)
    ax.legend(loc="lower right", fontsize=6.6, frameon=False)
    save_figure(fig, "fig_results_training_curves_single")
    plt.close(fig)


# 其余的冗余（大宽版）双栏图表函数，考虑到论文排版，这里对其逻辑进行了同样的 rcParams 全局优化。
# 为了保持代码简洁，若论文要求主要是单栏图表，以上的 Single 函数已经被重构为最高质量的标准。
# 现将主流程保留：

def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_source_tables(rows, summary)

    # Clear manuscript figures: one message per plot.
    plot_main_comparison_simple(summary)
    plot_ablation_simple(summary)
    plot_per_class_f1_simple()
    plot_training_curves_simple(summary)
    plot_layer_weights_single(summary)

    # 原始的双栏图表代码未列出，它们也会自动继承 mpl.rcParams 的高级全局配置
    # 如果还需要调用双图函数，如 plot_main_comparison(summary)，其外观也会大幅提升

    print(f"Wrote PNG figures to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR}")


if __name__ == "__main__":
    main()
