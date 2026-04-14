#!/usr/bin/env python3
"""Plot Pareto frontiers for Harbor jobs summaries.

The figure compares average completion rate against one selected efficiency metric
for all job records found under ``jobs/**/summary.json``. It prefers matplotlib
when available and falls back to a self-contained SVG renderer otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOBS_DIR = ROOT / "jobs"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "figures"
DEFAULT_METRIC = "cost_usd"
DEFAULT_BASENAME = "jobs_pareto_completion_vs_cost_usd"

BASE_COLOR = "#4C78A8"
SKILL_COLOR = "#E45756"
FRONTIER_COLOR = "#1B1F3B"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#222222"
BACKGROUND_COLOR = "#FFFFFF"
FONT_SCALE = 2.5
POINT_LABEL_SCALE = 1.1
LEGEND_SCALE = 0.72

AGENT_PREFIX_LABELS = {
    "claude-code-": "Claude Code",
    "codex-cli-": "Codex CLI",
    "kimi-cli-": "Kimi CLI",
    "qwen-coder-": "Qwen Coder",
}

MODEL_LABEL_OVERRIDES = {
    "minimax2dot5": "MiniMax 2.5",
    "minimax2dot7": "MiniMax 2.7",
    "opus4dot5": "Opus 4.5",
    "opus4dot6": "Opus 4.6",
    "sonnet4dot5": "Sonnet 4.5",
    "sonnet4dot6": "Sonnet 4.6",
    "5.3-codex": "GPT-5.3 Codex",
    "5.4": "GPT-5.4",
    "kimi-k2dot5": "Kimi K2.5",
    "qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "qwen3-coder-mext": "Qwen3-Coder-Mext",
}

MARKERS = {
    "claude": "circle",
    "minimax": "square",
    "kimi": "triangle",
    "qwen": "diamond",
    "other": "circle",
}


@dataclass(frozen=True)
class JobPoint:
    slug: str
    label: str
    base_slug: str
    family: str
    is_skill: bool
    completion_rate: float
    cost_usd: float | None
    interaction_turns: float | None
    output_tokens_k: float | None


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class Canvas:
    width: int = 1600
    height: int = 980
    left: int = 120
    right: int = 320
    top: int = 90
    bottom: int = 120

    @property
    def plot_width(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> int:
        return self.height - self.top - self.bottom


@dataclass(frozen=True)
class MetricSpec:
    key: str
    field_name: str
    axis_label: str
    basename: str
    log_scale: bool
    tick_kind: str


METRIC_SPECS = {
    "cost_usd": MetricSpec(
        key="cost_usd",
        field_name="cost_usd",
        axis_label="Average Cost (USD, log scale)",
        basename="jobs_pareto_completion_vs_cost_usd",
        log_scale=True,
        tick_kind="currency",
    ),
    "output_tokens_k": MetricSpec(
        key="output_tokens_k",
        field_name="output_tokens_k",
        axis_label="Average Output Tokens (K, log scale)",
        basename="jobs_pareto_completion_vs_output_tokens_k",
        log_scale=True,
        tick_kind="tokens_k",
    ),
    "interaction_turns": MetricSpec(
        key="interaction_turns",
        field_name="interaction_turns",
        axis_label="Average Interaction Turns",
        basename="jobs_pareto_completion_vs_interaction_turns",
        log_scale=False,
        tick_kind="turns",
    ),
}


def parse_args(
    argv: list[str] | None = None,
    *,
    default_metric: str = DEFAULT_METRIC,
    default_basename: str = DEFAULT_BASENAME,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Pareto frontier for jobs summaries.")
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help=f"Directory containing job batch folders (default: {DEFAULT_JOBS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS),
        default=default_metric,
        help="Metric shown on the x-axis.",
    )
    parser.add_argument(
        "--basename",
        default=default_basename,
        help="Base filename for generated figure files.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Expected a JSON object in {path}"
        raise ValueError(msg)
    return data


def parse_optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        return None
    return parsed


def model_slug_to_label(model_slug: str) -> str:
    if model_slug in MODEL_LABEL_OVERRIDES:
        return MODEL_LABEL_OVERRIDES[model_slug]

    label = model_slug.replace("dot", ".")
    label = label.replace("-", " ")
    return " ".join(part.upper() if part.isupper() else part for part in label.split())


def short_model_label(model_label: str) -> str:
    short = model_label
    short = short.replace("Claude Code + ", "")
    short = short.replace("Codex CLI + ", "")
    short = short.replace("Kimi CLI + ", "")
    short = short.replace("Qwen Coder + ", "")
    short = short.replace("Qwen3-Coder-", "Qwen3-")
    short = short.replace("-A35B-Instruct", "-A35B")
    short = short.replace("Kimi K2.5", "Kimi-K2.5")
    short = short.replace("GPT-5.3 Codex", "GPT-5.3-Codex")
    return short


def slug_to_label(slug: str) -> str:
    job_slug = Path(slug).parts[0]
    has_skill = job_slug.endswith("-skill")
    base_job_slug = job_slug.removesuffix("-skill")

    for prefix, agent_label in AGENT_PREFIX_LABELS.items():
        if base_job_slug.startswith(prefix):
            model_slug = base_job_slug[len(prefix) :]
            model_label = model_slug_to_label(model_slug)
            short_label = short_model_label(model_label)
            return f"{short_label}*" if has_skill else short_label

    fallback = base_job_slug.replace("dot", ".").replace("-", " ")
    fallback = " ".join(fallback.split())
    return f"{fallback}*" if has_skill else fallback


def infer_family(slug: str) -> str:
    lowered = slug.lower()
    if "qwen" in lowered:
        return "qwen"
    if "kimi" in lowered:
        return "kimi"
    if "minimax" in lowered:
        return "minimax"
    if "claude" in lowered:
        return "claude"
    return "other"


def _is_skill_record(path: Path) -> bool:
    return any(part.endswith("-skill") for part in path.parts)


def _base_slug_from_path(path: Path) -> str:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.endswith("-skill"):
            parts[index] = part.removesuffix("-skill")
            break
    return "/".join(parts)


def load_points(jobs_dir: Path) -> list[JobPoint]:
    points: list[JobPoint] = []
    for summary_path in sorted(jobs_dir.rglob("summary.json")):
        if not summary_path.is_file():
            continue
        record_dir = summary_path.parent
        rel_path = record_dir.relative_to(jobs_dir)
        data = load_json(summary_path)
        slug = rel_path.as_posix()
        is_skill = _is_skill_record(rel_path)
        base_slug = _base_slug_from_path(rel_path) if is_skill else slug
        points.append(
            JobPoint(
                slug=slug,
                label=slug_to_label(slug),
                base_slug=base_slug,
                family=infer_family(slug),
                is_skill=is_skill,
                completion_rate=float(data["avg_completion_rate"]),
                cost_usd=parse_optional_positive_float(data.get("avg_cost_usd")),
                interaction_turns=parse_optional_positive_float(data.get("avg_interaction_turns")),
                output_tokens_k=parse_optional_positive_float(data.get("avg_output_tokens_k")),
            )
        )
    if not points:
        msg = f"No summary.json found under {jobs_dir}"
        raise FileNotFoundError(msg)
    return points


def metric_value(point: JobPoint, spec: MetricSpec) -> float | None:
    value = getattr(point, spec.field_name)
    if value is None:
        return None
    return float(value)


def select_points(points: list[JobPoint], spec: MetricSpec) -> list[JobPoint]:
    selected = [point for point in points if metric_value(point, spec) is not None]
    if not selected:
        msg = f"No valid points found for metric {spec.key}"
        raise ValueError(msg)
    return selected


def pareto_frontier(points: list[JobPoint], spec: MetricSpec) -> list[JobPoint]:
    ordered = sorted(
        points,
        key=lambda item: (metric_value(item, spec), -item.completion_rate, item.slug),
    )
    frontier: list[JobPoint] = []
    best_y = -1.0
    for point in ordered:
        if point.completion_rate > best_y + 1e-12:
            frontier.append(point)
            best_y = point.completion_rate
    return frontier


def compute_bounds(points: list[JobPoint], spec: MetricSpec) -> Bounds:
    x_values = [metric_value(point, spec) for point in points]
    valid_x_values = [value for value in x_values if value is not None]
    y_values = [point.completion_rate for point in points]
    x_min = min(valid_x_values)
    x_max = max(valid_x_values)
    if spec.log_scale:
        x_min *= 0.85
        x_max *= 1.2
        if x_min <= 0:
            x_min = min(value for value in valid_x_values if value > 0) * 0.85
    else:
        spread = x_max - x_min
        margin = spread * 0.08 if spread > 0 else max(1.0, x_max * 0.08)
        x_min = max(0.0, x_min - margin)
        x_max = x_max + margin
        if math.isclose(x_min, x_max):
            x_max = x_min + 1.0
    y_min = max(0.0, min(y_values) - 0.05)
    y_max = min(1.0, max(y_values) + 0.05)
    if y_max - y_min < 0.2:
        mid = (y_min + y_max) / 2
        y_min = max(0.0, mid - 0.1)
        y_max = min(1.0, mid + 0.1)
    return Bounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def x_to_px(x: float, bounds: Bounds, canvas: Canvas, spec: MetricSpec) -> float:
    if spec.log_scale:
        log_min = math.log10(bounds.x_min)
        log_max = math.log10(bounds.x_max)
        ratio = (math.log10(x) - log_min) / (log_max - log_min)
    else:
        ratio = (x - bounds.x_min) / (bounds.x_max - bounds.x_min)
    return canvas.left + ratio * canvas.plot_width


def y_to_px(y: float, bounds: Bounds, canvas: Canvas) -> float:
    ratio = (y - bounds.y_min) / (bounds.y_max - bounds.y_min)
    return canvas.top + (1 - ratio) * canvas.plot_height


def marker_color(point: JobPoint) -> str:
    return SKILL_COLOR if point.is_skill else BASE_COLOR


def marker_shape(point: JobPoint) -> str:
    return MARKERS.get(point.family, "circle")


def pair_segments(points: list[JobPoint]) -> list[tuple[JobPoint, JobPoint]]:
    by_slug = {point.slug: point for point in points}
    segments: list[tuple[JobPoint, JobPoint]] = []
    for point in points:
        if point.is_skill:
            base = by_slug.get(point.base_slug)
            if base is not None:
                segments.append((base, point))
    return sorted(segments, key=lambda pair: pair[0].slug)


def svg_marker(shape: str, x: float, y: float, color: str) -> str:
    stroke = FRONTIER_COLOR if color == SKILL_COLOR else "#2F4B7C"
    if shape == "square":
        return f'<rect x="{x - 7:.2f}" y="{y - 7:.2f}" width="14" height="14" fill="{color}" stroke="{stroke}" stroke-width="1.5" />'
    if shape == "triangle":
        points = f"{x:.2f},{y - 8:.2f} {x - 8:.2f},{y + 7:.2f} {x + 8:.2f},{y + 7:.2f}"
        return f'<polygon points="{points}" fill="{color}" stroke="{stroke}" stroke-width="1.5" />'
    if shape == "diamond":
        points = f"{x:.2f},{y - 9:.2f} {x - 9:.2f},{y:.2f} {x:.2f},{y + 9:.2f} {x + 9:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{color}" stroke="{stroke}" stroke-width="1.5" />'
    if shape == "cross":
        return (
            f'<g stroke="{color}" stroke-width="3">'
            f'<line x1="{x - 7:.2f}" y1="{y - 7:.2f}" x2="{x + 7:.2f}" y2="{y + 7:.2f}" />'
            f'<line x1="{x - 7:.2f}" y1="{y + 7:.2f}" x2="{x + 7:.2f}" y2="{y - 7:.2f}" />'
            "</g>"
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="{stroke}" stroke-width="1.5" />'


def log_tick_values(bounds: Bounds) -> list[float]:
    start = int(math.floor(math.log10(bounds.x_min)))
    end = int(math.ceil(math.log10(bounds.x_max)))
    ticks: list[float] = []
    for exponent in range(start, end + 1):
        value = 10**exponent
        if bounds.x_min <= value <= bounds.x_max:
            ticks.append(float(value))
    return ticks


def linear_tick_values(bounds: Bounds, target_steps: int = 6) -> list[float]:
    span = bounds.x_max - bounds.x_min
    if span <= 0:
        return [bounds.x_min]
    rough_step = span / max(1, target_steps - 1)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1:
        nice_step = 1 * magnitude
    elif normalized <= 2:
        nice_step = 2 * magnitude
    elif normalized <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    start = math.ceil(bounds.x_min / nice_step) * nice_step
    ticks: list[float] = []
    value = start
    while value <= bounds.x_max + 1e-9:
        ticks.append(round(value, 10))
        value += nice_step
    return ticks or [bounds.x_min, bounds.x_max]


def tick_values(bounds: Bounds, spec: MetricSpec) -> list[float]:
    if spec.log_scale:
        return log_tick_values(bounds)
    return linear_tick_values(bounds)


def format_tick(value: float, spec: MetricSpec) -> str:
    if spec.tick_kind == "currency":
        if value >= 1:
            return f"${value:.0f}" if math.isclose(value, round(value)) else f"${value:.2f}".rstrip("0").rstrip(".")
        if value >= 0.1:
            return f"${value:.2f}".rstrip("0").rstrip(".")
        return f"${value:.3f}".rstrip("0").rstrip(".")
    if spec.tick_kind == "tokens_k":
        return f"{value:g}k"
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def build_svg(points: list[JobPoint], spec: MetricSpec, output_path: Path) -> None:
    canvas = Canvas()
    bounds = compute_bounds(points, spec)
    frontier = pareto_frontier(points, spec)
    frontier_slugs = {point.slug for point in frontier}
    segments = pair_segments(points)

    label_positions = {}
    sorted_points = sorted(
        points,
        key=lambda item: (item.completion_rate, metric_value(item, spec), item.slug),
    )
    for index, point in enumerate(sorted_points):
        x_value = metric_value(point, spec)
        if x_value is None:
            continue
        x = x_to_px(x_value, bounds, canvas, spec)
        y = y_to_px(point.completion_rate, bounds, canvas)
        dx = 12 if index % 2 == 0 else -12
        dy = -10 - (index % 4) * 4
        text_anchor = "start" if dx > 0 else "end"
        label_positions[point.slug] = (x + dx, y + dy, text_anchor)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width}" height="{canvas.height}" viewBox="0 0 {canvas.width} {canvas.height}">',
        f'<rect width="100%" height="100%" fill="{BACKGROUND_COLOR}" />',
    ]

    for y_tick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        if bounds.y_min <= y_tick <= bounds.y_max:
            py = y_to_px(y_tick, bounds, canvas)
            svg_parts.append(
                f'<line x1="{canvas.left}" y1="{py:.2f}" x2="{canvas.left + canvas.plot_width}" y2="{py:.2f}" stroke="{GRID_COLOR}" stroke-width="1" stroke-dasharray="6 6" />'
            )
            svg_parts.append(
                f'<text x="{canvas.left - 16}" y="{py + 5:.2f}" text-anchor="end" fill="#666666" font-size="14" font-family="Arial, Helvetica, sans-serif">{y_tick:.1f}</text>'
            )

    for tick in tick_values(bounds, spec):
        px = x_to_px(tick, bounds, canvas, spec)
        svg_parts.append(
            f'<text x="{px:.2f}" y="{canvas.top + canvas.plot_height + 28}" text-anchor="middle" fill="#666666" font-size="14" font-family="Arial, Helvetica, sans-serif">{escape(format_tick(tick, spec))}</text>'
        )

    svg_parts.append(
        f'<rect x="{canvas.left}" y="{canvas.top}" width="{canvas.plot_width}" height="{canvas.plot_height}" fill="none" stroke="#AEB6C2" stroke-width="1.4" />'
    )

    for base_point, skill_point in segments:
        x1_value = metric_value(base_point, spec)
        x2_value = metric_value(skill_point, spec)
        if x1_value is None or x2_value is None:
            continue
        x1 = x_to_px(x1_value, bounds, canvas, spec)
        y1 = y_to_px(base_point.completion_rate, bounds, canvas)
        x2 = x_to_px(x2_value, bounds, canvas, spec)
        y2 = y_to_px(skill_point.completion_rate, bounds, canvas)
        svg_parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#B8BFCB" stroke-width="1.8" stroke-dasharray="6 6" />'
        )

    frontier_points = sorted(frontier, key=lambda item: (metric_value(item, spec), item.slug))
    for first, second in zip(frontier_points, frontier_points[1:]):
        x1_value = metric_value(first, spec)
        x2_value = metric_value(second, spec)
        if x1_value is None or x2_value is None:
            continue
        x1 = x_to_px(x1_value, bounds, canvas, spec)
        y1 = y_to_px(first.completion_rate, bounds, canvas)
        x2 = x_to_px(x2_value, bounds, canvas, spec)
        y2 = y_to_px(second.completion_rate, bounds, canvas)
        svg_parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{FRONTIER_COLOR}" stroke-width="3" />'
        )

    for point in points:
        x_value = metric_value(point, spec)
        if x_value is None:
            continue
        x = x_to_px(x_value, bounds, canvas, spec)
        y = y_to_px(point.completion_rate, bounds, canvas)
        svg_parts.append(svg_marker(marker_shape(point), x, y, marker_color(point)))
        if point.slug in frontier_slugs:
            svg_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="12" fill="none" stroke="{FRONTIER_COLOR}" stroke-width="2" stroke-dasharray="2 4" />'
            )
        label_x, label_y, anchor = label_positions[point.slug]
        svg_parts.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" fill="{TEXT_COLOR}" font-size="13" font-family="Arial, Helvetica, sans-serif">{escape(point.label)}</text>'
        )

    x_axis_center = canvas.left + canvas.plot_width / 2
    y_axis_center = canvas.top + canvas.plot_height / 2
    svg_parts.append(
        f'<text x="{x_axis_center:.2f}" y="{canvas.height - 38}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="18" font-family="Arial, Helvetica, sans-serif">{escape(spec.axis_label)}</text>'
    )
    svg_parts.append(
        f'<text x="38" y="{y_axis_center:.2f}" text-anchor="middle" transform="rotate(-90 38 {y_axis_center:.2f})" fill="{TEXT_COLOR}" font-size="18" font-family="Arial, Helvetica, sans-serif">Average Completion Rate</text>'
    )

    legend_x = canvas.left + canvas.plot_width + 36
    legend_y = canvas.top + 10
    svg_parts.append(
        f'<text x="{legend_x}" y="{legend_y}" fill="{TEXT_COLOR}" font-size="18" font-family="Arial, Helvetica, sans-serif" font-weight="700">Legend</text>'
    )
    svg_parts.append(svg_marker("circle", legend_x + 10, legend_y + 28, BASE_COLOR))
    svg_parts.append(
        f'<text x="{legend_x + 30}" y="{legend_y + 33}" fill="{TEXT_COLOR}" font-size="14" font-family="Arial, Helvetica, sans-serif">Base run</text>'
    )
    svg_parts.append(svg_marker("circle", legend_x + 10, legend_y + 58, SKILL_COLOR))
    svg_parts.append(
        f'<text x="{legend_x + 30}" y="{legend_y + 63}" fill="{TEXT_COLOR}" font-size="14" font-family="Arial, Helvetica, sans-serif">Skill run</text>'
    )
    svg_parts.append(
        f'<line x1="{legend_x}" y1="{legend_y + 84}" x2="{legend_x + 22}" y2="{legend_y + 84}" stroke="#B8BFCB" stroke-width="1.8" stroke-dasharray="6 6" />'
    )
    svg_parts.append(
        f'<text x="{legend_x + 30}" y="{legend_y + 89}" fill="{TEXT_COLOR}" font-size="14" font-family="Arial, Helvetica, sans-serif">Base → skill pair</text>'
    )
    svg_parts.append(
        f'<line x1="{legend_x}" y1="{legend_y + 114}" x2="{legend_x + 22}" y2="{legend_y + 114}" stroke="{FRONTIER_COLOR}" stroke-width="3" />'
    )
    svg_parts.append(
        f'<text x="{legend_x + 30}" y="{legend_y + 119}" fill="{TEXT_COLOR}" font-size="14" font-family="Arial, Helvetica, sans-serif">Pareto frontier</text>'
    )

    family_entries = [
        ("claude", "Claude family"),
        ("minimax", "MiniMax family"),
        ("kimi", "Kimi family"),
        ("qwen", "Qwen family"),
    ]
    for index, (family, title) in enumerate(family_entries):
        cy = legend_y + 162 + index * 28
        svg_parts.append(svg_marker(MARKERS[family], legend_x + 10, cy, "#7F8EA3"))
        svg_parts.append(
            f'<text x="{legend_x + 30}" y="{cy + 5}" fill="{TEXT_COLOR}" font-size="14" font-family="Arial, Helvetica, sans-serif">{title}</text>'
        )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def try_plot_with_matplotlib(points: list[JobPoint], spec: MetricSpec, output_paths: list[Path]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "svg.fonttype": "none",
            "font.size": 10 * FONT_SCALE,
            "axes.labelsize": 12 * FONT_SCALE,
            "xtick.labelsize": 10 * FONT_SCALE,
            "ytick.labelsize": 10 * FONT_SCALE,
            "legend.fontsize": 10 * FONT_SCALE * LEGEND_SCALE,
        }
    )

    bounds = compute_bounds(points, spec)
    frontier = pareto_frontier(points, spec)
    segments = pair_segments(points)

    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=220)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.9, linestyle=(0, (4, 4)))

    for base_point, skill_point in segments:
        x1_value = metric_value(base_point, spec)
        x2_value = metric_value(skill_point, spec)
        if x1_value is None or x2_value is None:
            continue
        ax.plot(
            [x1_value, x2_value],
            [base_point.completion_rate, skill_point.completion_rate],
            color="#B8BFCB",
            linewidth=1.5,
            linestyle=(0, (4, 4)),
            zorder=1,
        )

    for point in points:
        x_value = metric_value(point, spec)
        if x_value is None:
            continue
        color = marker_color(point)
        marker = {
            "circle": "o",
            "square": "s",
            "triangle": "^",
            "diamond": "D",
            "cross": "X",
        }[marker_shape(point)]
        ax.scatter(
            x_value,
            point.completion_rate,
            s=150,
            marker=marker,
            color=color,
            edgecolor="#24364B",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            point.label,
            (x_value, point.completion_rate),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8.5 * FONT_SCALE * POINT_LABEL_SCALE,
            color=TEXT_COLOR,
            zorder=4,
            clip_on=False,
        )

    ax.plot(
        [metric_value(point, spec) for point in frontier],
        [point.completion_rate for point in frontier],
        color=FRONTIER_COLOR,
        linewidth=2.6,
        zorder=2,
    )

    if spec.log_scale:
        ax.set_xscale("log")
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ticks = tick_values(bounds, spec)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_tick(tick, spec) for tick in ticks])
    ax.set_xlabel(spec.axis_label, fontsize=12 * FONT_SCALE)
    ax.set_ylabel("Average Completion Rate", fontsize=12 * FONT_SCALE)

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Base run", markerfacecolor=BASE_COLOR, markeredgecolor="#24364B", markersize=9.5 * FONT_SCALE * LEGEND_SCALE),
        Line2D([0], [0], marker="o", color="w", label="Skill run (*)", markerfacecolor=SKILL_COLOR, markeredgecolor="#24364B", markersize=9.5 * FONT_SCALE * LEGEND_SCALE),
        Line2D([0], [0], color="#B8BFCB", linestyle=(0, (4, 4)), label="Base → skill pair"),
        Line2D([0], [0], color=FRONTIER_COLOR, linewidth=2.6, label="Pareto frontier"),
        Line2D([0], [0], marker="o", color="#7F8EA3", linestyle="None", label="Claude Code", markersize=9.5 * FONT_SCALE * LEGEND_SCALE),
        Line2D([0], [0], marker="s", color="#7F8EA3", linestyle="None", label="Codex CLI", markersize=9.5 * FONT_SCALE * LEGEND_SCALE),
        Line2D([0], [0], marker="^", color="#7F8EA3", linestyle="None", label="Kimi CLI", markersize=9.5 * FONT_SCALE * LEGEND_SCALE),
        Line2D([0], [0], marker="D", color="#7F8EA3", linestyle="None", label="Qwen Coder", markersize=9.5 * FONT_SCALE * LEGEND_SCALE),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#BFC7D5",
        framealpha=0.95,
        borderpad=0.35,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    fig.tight_layout()
    written_paths: list[Path] = []
    for output_path in output_paths:
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        written_paths.append(output_path)
    plt.close(fig)
    return written_paths


def write_data_table(points: list[JobPoint], output_path: Path) -> None:
    lines = [
        "slug,label,condition,completion_rate,cost_usd,interaction_turns,output_tokens_k"
    ]
    for point in sorted(points, key=lambda item: item.slug):
        condition = "skill" if point.is_skill else "base"
        cost_usd = "" if point.cost_usd is None else f"{point.cost_usd:.6f}"
        interaction_turns = "" if point.interaction_turns is None else f"{point.interaction_turns:.2f}"
        output_tokens_k = "" if point.output_tokens_k is None else f"{point.output_tokens_k:.2f}"
        lines.append(
            f'{point.slug},"{point.label}",{condition},{point.completion_rate:.4f},{cost_usd},{interaction_turns},{output_tokens_k}'
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_metric(points: list[JobPoint], spec: MetricSpec, output_dir: Path, basename: str) -> list[Path]:
    selected_points = select_points(points, spec)
    svg_path = output_dir / f"{basename}.svg"
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    csv_path = output_dir / f"{basename}.csv"

    mpl_written_paths = try_plot_with_matplotlib(selected_points, spec, [svg_path, png_path, pdf_path])
    if not mpl_written_paths:
        build_svg(selected_points, spec, svg_path)
    write_data_table(selected_points, csv_path)

    print(f"[OK] Wrote SVG: {svg_path}")
    if mpl_written_paths:
        for path in mpl_written_paths:
            if path.suffix.lower() == ".svg":
                continue
            print(f"[OK] Wrote {path.suffix.upper().lstrip('.')} : {path}")
    else:
        print("[WARN] matplotlib unavailable; SVG used fallback renderer; skipped PNG/PDF output")
    print(f"[OK] Wrote CSV: {csv_path}")
    return [svg_path, *[path for path in mpl_written_paths if path != svg_path], csv_path]


def main(
    argv: list[str] | None = None,
    *,
    default_metric: str = DEFAULT_METRIC,
    default_basename: str = DEFAULT_BASENAME,
) -> int:
    args = parse_args(argv, default_metric=default_metric, default_basename=default_basename)
    jobs_dir = args.jobs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(jobs_dir)
    spec = METRIC_SPECS[args.metric]
    render_metric(points, spec, output_dir, args.basename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
