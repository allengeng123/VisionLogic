"""Create the Section 3.3 grounding-flow figure from a verified example."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "figures" / "grounding_flow_assets"
OUT = ROOT / "figures" / "VisionLogic_Grounding_Flow.pdf"
PREVIEW = ROOT / "figures" / "VisionLogic_Grounding_Flow.png"


def draw_boxes(ax, boxes, color="#00ff00", linewidth=1.7):
    for box in boxes:
        ax.add_patch(Rectangle(
            (box["x0"], box["y0"]), box["x1"] - box["x0"], box["y1"] - box["y0"],
            fill=False, edgecolor=color, linewidth=linewidth, joinstyle="round"
        ))


def image_axis(fig, rect, image, title, subtitle=None, boxes=None):
    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_xlim(0, 224); ax.set_ylim(224, 0); ax.axis("off")
    if boxes:
        draw_boxes(ax, boxes)
    ax.text(.5, 1.055, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.3, fontweight="semibold", color="#182433")
    if subtitle:
        ax.text(.5, -.075, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=11.4, color="#263548")
    return ax


def arrow(fig, start, end, color="#253244", style="-|>", lw=2.0, rad=0.0, alpha=1.0):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                  arrowstyle=style, mutation_scale=15,
                                  linewidth=lw, color=color,
                                  alpha=alpha,
                                  connectionstyle=f"arc3,rad={rad}"))


def main():
    metadata = json.loads((ASSETS / "metadata.json").read_text(encoding="utf-8"))
    boxes = metadata["boxes"]
    rgb = Image.open(ASSETS / "input.png").convert("RGB")
    heat_overlay = Image.open(ASSETS / "heatmap.png").convert("RGB")
    ablated = Image.open(ASSETS / "blurred.png").convert("RGB")
    overlay = Image.open(ASSETS / "foreground.png").convert("RGB")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.2, 4.65), facecolor="white")

    # Two large panels echo the original Figure 1 silhouette.
    left = FancyBboxPatch((.012, .090), .680, .860,
                          boxstyle="round,pad=0.007,rounding_size=.025",
                          transform=fig.transFigure, facecolor="#fff4e8",
                          edgecolor="#1f2933", linewidth=2.1,
                          linestyle=(0, (5, 4)), zorder=-10)
    right = FancyBboxPatch((.712, .090), .276, .860,
                           boxstyle="round,pad=0.007,rounding_size=.025",
                           transform=fig.transFigure, facecolor="#edf4ff",
                           edgecolor="#1f2933", linewidth=2.1,
                           linestyle=(0, (5, 4)), zorder=-10)
    fig.add_artist(left); fig.add_artist(right)

    # Left panel: proposal and validation.
    w, h, y = .128, .43, .30
    image_axis(fig, [.035, y, w, h], rgb, "Input image",
               r"$p_{(1349,-)}(x)=1$")
    image_axis(fig, [.198, y, w, h], heat_overlay, "Neuron-targeted heatmap",
               "gradient-based; perturbation-based\nfallback")
    image_axis(fig, [.366, y, w, h], rgb, "Candidate box union $B$",
               r"connected regions", boxes=boxes)
    image_axis(fig, [.535, y, w, h], ablated, "Targeted blur removal")
    arrow(fig, (.164, .555), (.195, .555))
    arrow(fig, (.328, .555), (.363, .555))
    arrow(fig, (.496, .555), (.532, .555))

    search = FancyBboxPatch((.095, .832), .518, .067,
                            boxstyle="round,pad=.008,rounding_size=.014",
                            transform=fig.transFigure, facecolor="#ffead4",
                            edgecolor="#c56e24", linewidth=1.3)
    fig.add_artist(search)
    fig.text(.354, .866,
             "Progressively include weaker heatmap responses; stop at first predicate flip",
             ha="center", va="center", fontsize=13.5,
             color="#754315", fontweight="semibold")

    accept = FancyBboxPatch((.5135, .135), .170, .072,
                            boxstyle="round,pad=.01,rounding_size=.014",
                            transform=fig.transFigure, facecolor="#dff7e7",
                            edgecolor="#008000", linewidth=1.4)
    fig.add_artist(accept)
    fig.text(.5985, .171, "predicate deactivated  →  accept $B$",
             ha="center", va="center", fontsize=10.6,
             color="#008000", fontweight="semibold")
    # Align the equation with the neighboring "connected regions" subtitle.
    fig.text(.599, .2555, r"$p_{(1349,-)}(x_{\setminus B})=0$",
             ha="center", va="center", fontsize=11.4, color="#263548",
             zorder=20)

    # Right panel: validated box versus visualization-only intersection.
    image_axis(fig, [.733, .39, .105, .38], rgb, "Accepted region",
               r"validated: $B$", boxes=boxes)
    image_axis(fig, [.865, .39, .105, .38], overlay, "Foreground overlay",
               r"display only: $B\cap F$")
    arrow(fig, (.841, .565), (.861, .565), color="#3d5f87", lw=1.8)

    fig.text(.852, .866, "Foreground segmentation mask $F$",
             ha="center", va="center", fontsize=14.2,
             color="#27496d", fontweight="semibold")
    fig.text(.852, .235,
             "The removal test validates the box.\n"
             "Segmentation clarifies its visible content.",
             ha="center", va="center", fontsize=11.8,
             color="#26384d", linespacing=1.35)

    fig.text(.354, .025, "Attribution-guided box validation",
             ha="center", va="center", fontsize=17.5,
             fontweight="semibold", color="#182433")
    fig.text(.852, .025, "Foreground-assisted visualization",
             ha="center", va="center", fontsize=17.5,
             fontweight="semibold", color="#182433")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=.04)
    fig.savefig(PREVIEW, dpi=180, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    print(OUT)
    print(PREVIEW)


if __name__ == "__main__":
    main()
