"""DynaHybrid framework for paper — verticals broken around label bands."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

EDGE, ARROW, TEXT, MUTED = "#2B6A8A", "#2B6A8A", "#0a0a0a", "#222222"
FONT = "DejaVu Serif"

def box(ax, x, y, w, h, lw=2.0, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        linewidth=lw, edgecolor=EDGE, facecolor="white", zorder=z))

def outer(ax, x, y, w, h, lw=2.2):
    box(ax, x, y, w, h, lw=lw, z=2)

def txt(ax, x, y, s, fs=15, color=TEXT, weight="bold", **kw):
    ax.text(x, y, s, fontsize=fs, fontweight=weight, ha="center", va="center",
            color=color, fontfamily=FONT, zorder=10, **kw)

def seg(ax, x1, y1, x2, y2, lw=1.7, z=5):
    ax.plot([x1, x2], [y1, y2], color=ARROW, lw=lw, solid_capstyle="butt", solid_joinstyle="miter",
            zorder=z, clip_on=False)

def tip(ax, x, y, d, lw=1.7, ms=15):
    off = {"down": (0, 0.8), "up": (0, -0.8), "left": (0.8, 0), "right": (-0.8, 0)}[d]
    ax.annotate("", xy=(x, y), xytext=(x+off[0], y+off[1]),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=lw, mutation_scale=ms), zorder=6)

def main(output=None):
    fig, ax = plt.subplots(figsize=(20.0, 13.0))
    ax.set_xlim(0, 175); ax.set_ylim(0, 122)
    ax.set_aspect("equal"); ax.axis("off"); fig.patch.set_facecolor("white")

    outer(ax, 5, 99, 145, 19)
    ax.text(9, 115.5, "SERVER", fontsize=22, fontweight="bold", color=TEXT,
            ha="left", va="center", fontfamily=FONT, zorder=10)
    ax.text(146, 115.8, "ROUND t", fontsize=16, color="#444",
            ha="right", va="center", fontfamily=FONT, zorder=10)
    box(ax, 12, 101.5, 60, 12)
    txt(ax, 42, 110.0, "Global model:", fs=18)
    txt(ax, 42, 104.5, r"$w_t$", fs=25, weight="normal", style="italic")
    box(ax, 82, 101.5, 60, 12)
    txt(ax, 112, 110.0, "Select layers by relative gradient variance", fs=15)
    txt(ax, 112, 104.5, "for Hessian-based updates", fs=15)

    CLIENTS = [(6, "CLIENT 1"), (55, "CLIENT 2"), (112, "CLIENT N")]
    CW, CH, CY = 42, 56, 30
    centers = []
    for x, title in CLIENTS:
        outer(ax, x, CY, CW, CH)
        ax.text(x + 3, CY + CH - 3.0, title, fontsize=18, fontweight="bold",
                color=TEXT, ha="left", va="center", fontfamily=FONT, zorder=10)
        ld = (x + 3.0, CY + CH - 14.0, 13.0, 7.2)
        ls = (x + 18.5, CY + CH - 14.0, 19.5, 7.2)
        box(ax, *ld); txt(ax, ld[0]+ld[2]/2, ld[1]+ld[3]/2, "Local data", fs=15)
        box(ax, *ls)
        txt(ax, ls[0]+ls[2]/2, ls[1]+4.8, "Local steps", fs=15)
        txt(ax, ls[0]+ls[2]/2, ls[1]+1.9, "mini-batch gradients", fs=12, color=MUTED)
        ym = ld[1] + ld[3]/2
        seg(ax, ld[0]+ld[2]+0.2, ym, ls[0]-0.25, ym); tip(ax, ls[0], ym, "right", ms=14)
        pair_w, gap = 16.0, 1.5
        total = pair_w * 2 + gap
        pair_left = x + (CW - total) / 2
        ch = (pair_left, CY + 14.5, pair_w, 9.0)
        ot = (pair_left + pair_w + gap, CY + 14.5, pair_w, 9.0)
        box(ax, *ch)
        txt(ax, ch[0]+ch[2]/2, ch[1]+6.0, "Chosen layers", fs=14)
        txt(ax, ch[0]+ch[2]/2, ch[1]+2.8, "Hessian update", fs=13, color=MUTED)
        box(ax, *ot)
        txt(ax, ot[0]+ot[2]/2, ot[1]+6.0, "Other layers", fs=14)
        txt(ax, ot[0]+ot[2]/2, ot[1]+2.8, "SGD", fs=15, color=MUTED)
        ch_cx = ch[0]+ch[2]/2; ot_cx = ot[0]+ot[2]/2
        mid_cx = (ch_cx + ot_cx) / 2
        ls_cx = ls[0] + ls[2]/2
        y_below_ls = ls[1] - 2.5
        junc_y = ch[1] + ch[3] + 3.2
        seg(ax, ls_cx, ls[1]-0.15, ls_cx, y_below_ls)
        seg(ax, ls_cx, y_below_ls, mid_cx, y_below_ls)
        seg(ax, mid_cx, y_below_ls, mid_cx, junc_y)
        seg(ax, ch_cx, junc_y, ot_cx, junc_y)
        seg(ax, ch_cx, junc_y, ch_cx, ch[1]+ch[3]+0.2); tip(ax, ch_cx, ch[1]+ch[3], "down", ms=14)
        seg(ax, ot_cx, junc_y, ot_cx, ot[1]+ot[3]+0.2); tip(ax, ot_cx, ot[1]+ot[3], "down", ms=14)
        merge_y = CY + 11.0
        seg(ax, ch_cx, ch[1]-0.15, ch_cx, merge_y)
        seg(ax, ot_cx, ot[1]-0.15, ot_cx, merge_y)
        seg(ax, ch_cx, merge_y, ot_cx, merge_y)
        tip_y = merge_y - 4.4
        seg(ax, mid_cx, merge_y, mid_cx, tip_y + 0.15)
        tip(ax, mid_cx, tip_y, "down", ms=14)
        txt(ax, x + CW/2, CY + 4.2, "model + layer gradient means", fs=14)
        centers.append(x + CW/2)

    ax.text(104.5, 58, "···", fontsize=28, fontweight="bold", ha="center", va="center",
            color="#444", fontfamily=FONT, zorder=10)

    server_bot, client_top = 99.0, CY + CH
    bc_label_y = 93.5
    txt(ax, 80, bc_label_y, "broadcast global model + layer choices", fs=16, bbox=dict(facecolor="white", edgecolor="none", pad=1.6))
    for cx in centers:
        seg(ax, cx, server_bot, cx, bc_label_y + 2.8, lw=2.0)
        seg(ax, cx, bc_label_y - 2.8, cx, client_top + 0.25, lw=2.0)
        tip(ax, cx, client_top, "down", lw=2.0, ms=16)

    bottom_top = 18.5
    up_label_y = 24.5
    txt(ax, 80, up_label_y, "upload: complete models + trajectory-averaged layer gradients", fs=15, bbox=dict(facecolor="white", edgecolor="none", pad=1.6))
    for cx in centers:
        seg(ax, cx, CY, cx, up_label_y + 2.6, lw=2.0)
        seg(ax, cx, up_label_y - 2.6, cx, bottom_top + 0.25, lw=2.0)
        tip(ax, cx, bottom_top, "down", lw=2.0, ms=16)

    outer(ax, 5, 1.5, 145, 17)
    box(ax, 12, 4.0, 60, 12)
    txt(ax, 42, 12.5, "Weighted model averaging", fs=16)
    txt(ax, 36.5, 7.0, "next global model:", fs=14)
    txt(ax, 54.5, 7.0, r"$w_{t+1}$", fs=21, weight="normal")
    box(ax, 82, 4.0, 60, 12)
    txt(ax, 112, 12.5, "Normalized gradient disagreement", fs=15)
    txt(ax, 112, 7.0, "rank layers for the next round", fs=14)

    fb_x = 162
    seg(ax, 142, 10.0, fb_x, 10.0, lw=2.0)
    seg(ax, fb_x, 10.0, fb_x, 107.5, lw=2.0)
    seg(ax, fb_x, 107.5, 142.2, 107.5, lw=2.0)
    tip(ax, 142, 107.5, "left", lw=2.0, ms=16)

    plt.tight_layout(pad=0.3)
    out = Path(output or Path(__file__).with_name("figure1_user_base_fixed.pdf")).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out.with_suffix(".png"), format="png", bbox_inches="tight", pad_inches=0.15, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()



