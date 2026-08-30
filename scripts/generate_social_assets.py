"""Generate publication figures and a short LinkedIn video from verified results."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "benchmark.json"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

NAVY = "#071426"
PANEL = "#10233D"
BLUE = "#3B82F6"
TEAL = "#14B8A6"
AMBER = "#F59E0B"
RED = "#F87171"
WHITE = "#F8FAFC"
MUTED = "#A8B7CC"
GRID = "#29415F"
WIDTH = 1080
HEIGHT = 1080


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = [run["seed"] for run in payload["runs"]]
    if seeds != [11, 22, 33]:
        raise ValueError(f"expected verified seeds [11, 22, 33], got {seeds}")
    if payload["protocol"]["primary_metric"] != (
        "mean per-pixel MSE on the canonical MNIST test set"
    ):
        raise ValueError("unexpected primary metric")
    return payload


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=10)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="y", color=GRID, alpha=0.55, linewidth=0.8)
    axis.set_axisbelow(True)


def generate_dashboard(payload: dict[str, Any]) -> Path:
    aggregate = payload["aggregate"]
    baselines = payload["baselines"]
    capacity = payload["capacity_control"]

    labels = ["PCA", "Unified", "Mean image", "Full specialists", "Budget specialists"]
    values = [
        baselines["pca"]["test_mse"],
        aggregate["unified"]["test_mse_mean"],
        baselines["global_mean"]["test_mse"],
        aggregate["full_specialists"]["test_mse_mean"],
        aggregate["budget_specialists"]["test_mse_mean"],
    ]
    errors = [
        0.0,
        aggregate["unified"]["test_mse_sd"],
        0.0,
        aggregate["full_specialists"]["test_mse_sd"],
        aggregate["budget_specialists"]["test_mse_sd"],
    ]
    colors = [TEAL, BLUE, MUTED, AMBER, RED]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure = plt.figure(figsize=(16, 10), facecolor=NAVY)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.96,
        top=0.84,
        bottom=0.09,
        hspace=0.34,
        wspace=0.23,
    )
    figure.text(
        0.06,
        0.94,
        "Unified vs. class-specific autoencoders",
        color=WHITE,
        fontsize=27,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.885,
        "Canonical MNIST test set | 3 seeds | 63 neural fits | lower MSE is better",
        color=MUTED,
        fontsize=14,
    )

    axis = figure.add_subplot(grid[0, 0])
    _style_axis(axis)
    positions = np.arange(len(labels))
    axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="none",
        alpha=0.95,
    )
    axis.set_xticks(positions, labels, rotation=17, ha="right")
    axis.set_ylabel("Test reconstruction MSE", color=MUTED, fontsize=11)
    axis.set_title("Primary benchmark", color=WHITE, fontsize=16, loc="left", pad=14)
    axis.set_ylim(0, max(values) * 1.22)
    for position, value in zip(positions, values, strict=True):
        axis.text(
            position,
            value + 0.006,
            f"{value:.4f}",
            ha="center",
            color=WHITE,
            fontsize=10,
            fontweight="bold",
        )

    axis = figure.add_subplot(grid[0, 1])
    _style_axis(axis)
    names = ["Full specialists", "Budget specialists"]
    deltas = [
        aggregate["full_specialists"]["paired_delta_mean"],
        aggregate["budget_specialists"]["paired_delta_mean"],
    ]
    intervals = [
        aggregate["full_specialists"]["seed_bootstrap_95_ci"],
        aggregate["budget_specialists"]["seed_bootstrap_95_ci"],
    ]
    lower = [value - interval[0] for value, interval in zip(deltas, intervals, strict=True)]
    upper = [interval[1] - value for value, interval in zip(deltas, intervals, strict=True)]
    y_positions = np.arange(2)
    axis.errorbar(
        deltas,
        y_positions,
        xerr=np.asarray([lower, upper]),
        fmt="o",
        markersize=11,
        linewidth=2.2,
        capsize=6,
        color=AMBER,
        ecolor=MUTED,
    )
    axis.axvline(0, color=TEAL, linewidth=2)
    axis.set_yticks(y_positions, names)
    axis.set_xlabel("Specialist minus unified MSE (positive = worse)", color=MUTED)
    axis.set_title("Paired effect across seeds", color=WHITE, fontsize=16, loc="left", pad=14)
    axis.set_xlim(-0.01, 0.165)
    for y_position, value in zip(y_positions, deltas, strict=True):
        axis.text(value + 0.006, y_position, f"+{value:.4f}", va="center", color=WHITE)
    axis.invert_yaxis()

    axis = figure.add_subplot(grid[1, 0])
    axis.set_facecolor(PANEL)
    axis.axis("off")
    axis.set_title("Capacity control", color=WHITE, fontsize=16, loc="left", pad=14)
    capacity_rows = [
        ("Unified model", capacity["unified_parameters"], BLUE),
        ("10 full specialists", capacity["full_specialists_total_parameters"], AMBER),
        ("10 budget specialists", capacity["budget_specialists_total_parameters"], TEAL),
    ]
    maximum = max(item[1] for item in capacity_rows)
    for index, (name, count, color) in enumerate(capacity_rows):
        y = 0.76 - index * 0.26
        axis.text(0.02, y + 0.08, name, transform=axis.transAxes, color=MUTED, fontsize=12)
        axis.add_patch(
            plt.Rectangle(
                (0.02, y),
                0.70 * count / maximum,
                0.08,
                transform=axis.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(
            0.76,
            y + 0.04,
            f"{count:,}",
            transform=axis.transAxes,
            va="center",
            color=WHITE,
            fontsize=13,
            fontweight="bold",
        )
    axis.text(
        0.02,
        0.03,
        "Budget specialists are within 3% of unified total parameters.",
        transform=axis.transAxes,
        color=TEAL,
        fontsize=12,
        fontweight="bold",
    )

    axis = figure.add_subplot(grid[1, 1])
    axis.set_facecolor(PANEL)
    axis.axis("off")
    probe = 100 * aggregate["linear_probe"]["accuracy_mean"]
    probe_sd = 100 * aggregate["linear_probe"]["accuracy_sd"]
    axis.text(
        0.05,
        0.78,
        f"{probe:.2f}%",
        transform=axis.transAxes,
        color=TEAL,
        fontsize=48,
        fontweight="bold",
    )
    axis.text(
        0.05,
        0.65,
        f"+/- {probe_sd:.2f}% frozen-latent linear probe",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=13,
    )
    axis.text(
        0.05,
        0.43,
        "Main finding",
        transform=axis.transAxes,
        color=AMBER,
        fontsize=14,
        fontweight="bold",
    )
    finding = (
        "Neither specialist condition improved in any seed. "
        "The 64-component PCA baseline achieved the lowest reconstruction error."
    )
    axis.text(
        0.05,
        0.31,
        "\n".join(textwrap.wrap(finding, width=48)),
        transform=axis.transAxes,
        color=WHITE,
        fontsize=14,
        linespacing=1.45,
        va="top",
    )

    figure.text(
        0.06,
        0.025,
        "Measured results | source: results/benchmark.json | fixed 12-pass training budget",
        color=MUTED,
        fontsize=10,
    )
    output = RESULTS_DIR / "results_dashboard.png"
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def generate_digit_effects(payload: dict[str, Any]) -> Path:
    runs = payload["runs"]
    conditions = ("full_specialists", "budget_specialists")
    arrays = {
        condition: np.asarray(
            [
                [
                    run[condition]["by_digit"][str(digit)]["paired_delta_vs_unified"]
                    for digit in range(10)
                ]
                for run in runs
            ]
        )
        for condition in conditions
    }
    digits = np.arange(10)
    figure, axis = plt.subplots(figsize=(12, 6.8), facecolor=NAVY)
    _style_axis(axis)
    axis.errorbar(
        digits - 0.09,
        arrays["full_specialists"].mean(axis=0),
        yerr=arrays["full_specialists"].std(axis=0, ddof=1),
        fmt="o",
        markersize=8,
        capsize=4,
        linewidth=2,
        color=AMBER,
        label="Full specialists",
    )
    axis.errorbar(
        digits + 0.09,
        arrays["budget_specialists"].mean(axis=0),
        yerr=arrays["budget_specialists"].std(axis=0, ddof=1),
        fmt="o",
        markersize=8,
        capsize=4,
        linewidth=2,
        color=RED,
        label="Budget specialists",
    )
    axis.axhline(0, color=TEAL, linewidth=2, label="No difference")
    axis.set_xticks(digits)
    axis.set_xlabel("MNIST digit", color=MUTED, fontsize=12)
    axis.set_ylabel("Specialist minus unified MSE (positive = worse)", color=MUTED, fontsize=12)
    axis.set_title(
        "Specialists increased reconstruction error for every digit",
        color=WHITE,
        fontsize=20,
        loc="left",
        pad=17,
        fontweight="bold",
    )
    axis.text(
        0,
        1.02,
        "Mean +/- sample SD across seeds 11, 22, and 33",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=11,
    )
    legend = axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    for text in legend.get_texts():
        text.set_color(WHITE)
    axis.set_ylim(-0.01, 0.19)
    figure.tight_layout(rect=(0.04, 0.08, 0.98, 0.96))
    output = RESULTS_DIR / "paired_effects_by_digit.png"
    figure.savefig(output, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ["seguisb.ttf", "segoeuib.ttf"] if bold else ["segoeui.ttf"]
    names += ["arialbd.ttf" if bold else "arial.ttf"]
    for name in names:
        path = windows / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)


def _rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str) -> None:
    draw.rounded_rectangle(box, radius=28, fill=fill)


def _wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    spacing: int = 12,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = xy[1]
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3]
    for line in lines:
        draw.text((xy[0], y), line, font=font, fill=fill)
        y += line_height + spacing
    return y


def _base_frame(
    section: str, frame_number: int, total_frames: int
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(image)
    draw.text((70, 54), section.upper(), font=_font(23, True), fill=TEAL)
    draw.rounded_rectangle((70, 1015, 1010, 1027), radius=6, fill=GRID)
    progress = max(5, int(940 * frame_number / max(total_frames - 1, 1)))
    draw.rounded_rectangle((70, 1015, 70 + progress, 1027), radius=6, fill=BLUE)
    return image, draw


def _slide_question(progress: float, frame_number: int, total_frames: int) -> Image.Image:
    image, draw = _base_frame("Research question", frame_number, total_frames)
    _wrapped(
        draw,
        "Can 10 specialist autoencoders beat one unified model?",
        (70, 180),
        _font(67, True),
        WHITE,
        920,
        spacing=18,
    )
    draw.text((72, 490), "A controlled MNIST reconstruction study", font=_font(34), fill=MUTED)
    cards = [("3", "seeds"), ("63", "neural fits"), ("10,000", "test images")]
    for index, (value, label) in enumerate(cards):
        x = 70 + index * 315
        _rounded(draw, (x, 620, x + 285, 830), PANEL)
        draw.text((x + 28, 660), value, font=_font(55, True), fill=TEAL)
        draw.text((x + 28, 744), label, font=_font(27), fill=WHITE)
    draw.text(
        (70, 914), "Official split | paired errors | fixed protocol", font=_font(26), fill=AMBER
    )
    return image


def _slide_design(progress: float, frame_number: int, total_frames: int) -> Image.Image:
    image, draw = _base_frame("Experimental design", frame_number, total_frames)
    draw.text((70, 145), "Three conditions. One fair question.", font=_font(48, True), fill=WHITE)
    cards = [
        ("UNIFIED", "1 model", "113,017 parameters", BLUE),
        ("FULL SPECIALISTS", "10 models", "10x total capacity", AMBER),
        ("BUDGET SPECIALISTS", "10 smaller models", "within 3% total", TEAL),
    ]
    for index, (title, line1, line2, color) in enumerate(cards):
        y = 280 + index * 205
        _rounded(draw, (70, y, 1010, y + 165), PANEL)
        draw.rounded_rectangle((70, y, 86, y + 165), radius=8, fill=color)
        draw.text((115, y + 27), title, font=_font(27, True), fill=color)
        draw.text((115, y + 78), line1, font=_font(31, True), fill=WHITE)
        draw.text((610, y + 83), line2, font=_font(25), fill=MUTED)
    draw.text(
        (70, 930),
        "Primary metric: canonical test MSE | lower is better",
        font=_font(25),
        fill=MUTED,
    )
    return image


def _slide_bars(progress: float, frame_number: int, total_frames: int) -> Image.Image:
    image, draw = _base_frame("Measured result", frame_number, total_frames)
    draw.text((70, 130), "The simple baseline won.", font=_font(58, True), fill=WHITE)
    draw.text((70, 210), "Canonical MNIST test MSE", font=_font(29), fill=MUTED)
    rows = [
        ("PCA (64 components)", 0.009089, TEAL),
        ("Unified autoencoder", 0.016295, BLUE),
        ("Full specialists", 0.091855, AMBER),
        ("Budget specialists", 0.157198, RED),
    ]
    maximum = 0.17
    for index, (label, value, color) in enumerate(rows):
        y = 335 + index * 145
        draw.text((70, y), label, font=_font(27, True), fill=WHITE)
        draw.rounded_rectangle((70, y + 48, 850, y + 88), radius=18, fill=GRID)
        width = int(780 * min(value / maximum, 1.0) * progress)
        if width > 0:
            draw.rounded_rectangle((70, y + 48, 70 + width, y + 88), radius=18, fill=color)
        draw.text((875, y + 45), f"{value:.4f}", font=_font(27, True), fill=color)
    draw.text(
        (70, 930),
        "Neural values averaged across seeds | error bars reported in the repository",
        font=_font(22),
        fill=MUTED,
    )
    return image


def _slide_negative(progress: float, frame_number: int, total_frames: int) -> Image.Image:
    image, draw = _base_frame("Primary conclusion", frame_number, total_frames)
    draw.text((70, 135), "Specialization did not win.", font=_font(61, True), fill=WHITE)
    _rounded(draw, (70, 285, 500, 665), PANEL)
    draw.text((120, 345), "0 / 3", font=_font(96, True), fill=RED)
    draw.text((120, 490), "seeds improved", font=_font(33, True), fill=WHITE)
    draw.text((120, 548), "in either specialist", font=_font(25), fill=MUTED)
    draw.text((120, 590), "condition", font=_font(25), fill=MUTED)
    _rounded(draw, (540, 285, 1010, 665), PANEL)
    draw.text((590, 342), "PCA", font=_font(76, True), fill=TEAL)
    draw.text((590, 462), "best reconstruction", font=_font(28, True), fill=WHITE)
    draw.text((590, 520), "MSE 0.009089", font=_font(27), fill=MUTED)
    draw.text((70, 760), "A negative result is still a result", font=_font(41, True), fill=AMBER)
    _wrapped(
        draw,
        "when the baseline, protocol, and limitations are visible.",
        (70, 835),
        _font(31),
        WHITE,
        900,
    )
    return image


def _slide_probe(progress: float, frame_number: int, total_frames: int) -> Image.Image:
    image, draw = _base_frame("Representation check", frame_number, total_frames)
    _wrapped(
        draw,
        "The bottleneck still retained label information.",
        (70, 140),
        _font(45, True),
        WHITE,
        max_width=930,
        spacing=8,
    )
    center = (540, 520)
    radius = 230
    box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    draw.arc(box, 135, 405, fill=GRID, width=42)
    end = 135 + 270 * 0.9117 * progress
    draw.arc(box, 135, end, fill=TEAL, width=42)
    draw.text((365, 420), "91.17%", font=_font(76, True), fill=TEAL)
    draw.text((413, 535), "+/- 0.32%", font=_font(31), fill=MUTED)
    draw.text((345, 600), "linear-probe accuracy", font=_font(29, True), fill=WHITE)
    draw.text(
        (70, 880), "Frozen 64-D unified latents | official test set", font=_font(27), fill=MUTED
    )
    return image


def _slide_close(progress: float, frame_number: int, total_frames: int) -> Image.Image:
    image, draw = _base_frame("Research lesson", frame_number, total_frames)
    _wrapped(
        draw,
        "Validate the comparison, not just the visualization.",
        (70, 155),
        _font(61, True),
        WHITE,
        920,
        spacing=18,
    )
    checklist = [
        ("\u2713", "Canonical split", TEAL),
        ("\u2713", "Capacity control", TEAL),
        ("\u2713", "Raw evidence + CI", TEAL),
        ("\u2713", "Negative result retained", TEAL),
    ]
    for index, (symbol, label, color) in enumerate(checklist):
        y = 505 + index * 80
        draw.text((82, y), symbol, font=_font(32, True), fill=color)
        draw.text((135, y), label, font=_font(30), fill=WHITE)
    draw.rounded_rectangle((70, 865, 1010, 960), radius=24, fill=BLUE)
    draw.text(
        (105, 887),
        "github.com/Keval-7503/mnist-class-conditional-autoencoders",
        font=_font(23, True),
        fill=WHITE,
    )
    return image


def generate_video(payload: dict[str, Any], fps: int = 30) -> tuple[Path, Path]:
    del payload
    slides = [
        (4.0, "Research question", _slide_question),
        (4.5, "Experimental design", _slide_design),
        (6.0, "Measured result", _slide_bars),
        (4.5, "Primary conclusion", _slide_negative),
        (4.0, "Representation check", _slide_probe),
        (4.5, "Research lesson", _slide_close),
    ]
    total_seconds = sum(item[0] for item in slides)
    total_frames = int(total_seconds * fps)
    transition = 0.45
    output = ASSETS_DIR / "linkedin-project-video.mp4"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(output),
        (WIDTH, HEIGHT),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart", "-an"],
    )
    writer.send(None)
    try:
        for frame_number in range(total_frames):
            time_value = frame_number / fps
            elapsed = 0.0
            for index, (duration, _, renderer) in enumerate(slides):
                if time_value < elapsed + duration or index == len(slides) - 1:
                    local = max(0.0, time_value - elapsed)
                    progress = min(1.0, local / 1.15)
                    progress = progress * progress * (3.0 - 2.0 * progress)
                    frame = renderer(progress, frame_number, total_frames)
                    remaining = duration - local
                    if remaining < transition and index + 1 < len(slides):
                        next_renderer = slides[index + 1][2]
                        next_frame = next_renderer(0.0, frame_number, total_frames)
                        alpha = 1.0 - max(0.0, remaining) / transition
                        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                        frame = Image.blend(frame, next_frame, alpha)
                    writer.send(np.asarray(frame))
                    break
                elapsed += duration
    finally:
        writer.close()

    captions = """1
00:00:00,000 --> 00:00:04,000
Can ten specialist autoencoders beat one unified model?

2
00:00:04,000 --> 00:00:08,500
Three seeds, 63 neural fits, paired errors, and a total-parameter control.

3
00:00:08,500 --> 00:00:14,500
PCA achieved the lowest reconstruction MSE: 0.009089.

4
00:00:14,500 --> 00:00:19,000
Neither specialist condition improved in any of the three seeds.

5
00:00:19,000 --> 00:00:23,000
Unified 64-dimensional latents supported 91.17 percent linear-probe accuracy.

6
00:00:23,000 --> 00:00:27,500
Validate the comparison, not just the visualization.
"""
    caption_path = ASSETS_DIR / "linkedin-project-video.srt"
    caption_path.write_text(captions, encoding="utf-8")
    return output, caption_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charts-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.charts_only and args.video_only:
        raise ValueError("--charts-only and --video-only are mutually exclusive")
    if not 15 <= args.fps <= 60:
        raise ValueError("--fps must be between 15 and 60")
    payload = load_results()
    outputs: list[Path] = []
    if not args.video_only:
        outputs.extend([generate_dashboard(payload), generate_digit_effects(payload)])
    if not args.charts_only:
        outputs.extend(generate_video(payload, fps=args.fps))
    for output in outputs:
        print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
