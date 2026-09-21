# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

BASE = r"C:\Users\HUAWEI\Desktop\pfas"
rcParams["font.family"] = "DejaVu Sans"

def fig2():
    models = ["GNN\nscratch", "Random\nforest", "GNN pretrained\n(no PFAS seen)", "GNN pretrained\n+ fine-tuned"]
    scaffold, scaffold_sd = [0.607, 0.769, 0.798, 0.812], [0.044, 0, 0, 0.032]
    random_ = [0.751, 0.760, 0.804, 0.864]
    colors = ["#c44e52", "#8c8c8c", "#4c72b0", "#55a868"]
    x = np.arange(4); w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    for ax, vals, sds, title in [(axes[0], scaffold, scaffold_sd, "Scaffold split\n(chain-length withheld)"),
                                 (axes[1], random_, [0]*4, "Random split")]:
        ax.bar(x, vals, width=w, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
        if sds and any(sds):
            ax.errorbar(x, vals, yerr=sds, fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=4)
        for xi, v, sd in zip(x, vals, sds):
            ax.text(xi, v + sd + 0.018, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")
        ax.axhline(0.5, ls="--", lw=0.8, color="grey", zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.45, 0.95)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    axes[0].set_ylabel("Mean test AUROC", fontsize=10)
    fig.tight_layout()
    fig.savefig(BASE + r"\Fig2_main_comparison.png", dpi=300)
    plt.close(); print("Fig2 saved")

def fig4():
    from PIL import Image
    panels = [
        (r"\web_routing_pfoa.png",
         "(A) PFOA — detected as PFAS\nrouted to PFAS_Tox21_Panel (red 7-axis ADME-Tox radar)",
         "crimson"),
        (r"\web_routing_metoprolol.png",
         "(B) Metoprolol — not a PFAS\nstandard 5-axis drug-ADMET radar (unchanged)",
         "#4c72b0"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, (fn, title, color) in zip(axes, panels):
        img = np.array(Image.open(BASE + fn).convert("RGB"))
        h, w = img.shape[:2]
        img = img[int(h*0.045):int(h*0.985), int(w*0.015):int(w*0.985)]  # 裁掉浏览器标签 bleed
        ax.imshow(img)
        ax.set_title(title, fontsize=10.5, color=color, fontweight="bold", pad=10)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.06, left=0.015, right=0.985, top=0.76, bottom=0.02)
    fig.savefig(BASE + r"\Fig4_routing_demo.png", dpi=300)
    plt.close(); print("Fig4 saved (routing demo)")

def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    cats = ["PFAS test set\n(n = 86)", "Approved drugs\n(n = 368)"]
    correct, wrong = [82, 327], [4, 41]
    ax.barh(cats, correct, color="#55a868", edgecolor="black", lw=0.5, label="Correctly routed")
    ax.barh(cats, wrong, left=correct, color="#c44e52", edgecolor="black", lw=0.5, label="Missed / flagged")
    ax.text(41, 0, "82 (95.3%)", va="center", ha="center", fontsize=9.5, fontweight="bold", color="white")
    ax.text(88, 0, "4 missed", va="center", ha="left", fontsize=8.5, color="#c44e52", fontweight="bold")
    ax.text(163, 1, "327", va="center", ha="center", fontsize=9.5, fontweight="bold", color="white")
    ax.text(347, 1, "41 flagged", va="center", ha="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_xlim(0, 400); ax.invert_yaxis()
    ax.set_xlabel("Molecules", fontsize=10)
    ax.set_title("(A) Routing outcomes", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    groups = ["Perfluorocarbon drugs\n(routing arguably correct)",
              "Fluorinated inhalational\nanesthetics (PFAS-like)",
              "Borderline fluorinated\ntherapeutics (true FP)"]
    counts = [6, 8, 27]
    colors = ["#4c72b0", "#8172b3", "#c44e52"]
    bars = ax.barh(groups, counts, color=colors, edgecolor="black", lw=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_width()+0.6, b.get_y()+b.get_height()/2, str(c), va="center",
                fontsize=10, fontweight="bold")
    ax.set_xlim(0, 34); ax.invert_yaxis()
    ax.set_xlabel("Flagged approved drugs", fontsize=9.5)
    ax.set_title("(B) Flagged drugs form three\nchemically coherent groups", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(BASE + r"\Fig5_routing.png", dpi=300)
    plt.close(); print("Fig5 saved")

fig2(); fig4(); fig5()