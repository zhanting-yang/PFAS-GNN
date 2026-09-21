# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = r"C:\Users\HUAWEI\Desktop\pfas"
fig, ax = plt.subplots(figsize=(12, 8.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")

C_DATA, C_DATA_E = "#dbe7f3", "#4c72b0"
C_WARN = "#c44e52"
C_S = ["#f3d9d9", "#d9e4f3", "#d9efe0"]

def box(x, y, w, h, text, fc, ec, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color="#555555", lw=1.2))

box(3, 63, 22, 10, "PCBA large\ncollection\n60,027 cpds × 26 assays", C_DATA, C_DATA_E, 8)
box(3, 49, 22, 10, "PFAS panel\n(Cheng–Ng)\n1,012 PFAS × 26 assays", C_DATA, C_DATA_E, 8)
box(33, 55, 20, 11, "Canonical-SMILES\noverlap check", "#f5f5f5", "#555555", 8.5, True)
box(63, 55, 34, 11, "Decontamination:\nexclude all 1,012 PFAS\n→ pretraining pool ≈ 53k", C_DATA, C_DATA_E, 8.5)

ax.text(43, 51.5, "⚠ intersection = 1,012 / 1,012\ntotal pretraining–test leakage",
        ha="center", va="top", fontsize=8, color=C_WARN, fontweight="bold", linespacing=1.4)

box(33, 37, 20, 10, "PFAS training pool\n829 molecules\n(scaffold-aware split)", C_DATA, C_DATA_E, 8)

box(3, 19, 27, 13, "Strategy 1: scratch\nD-MPNN trained on\n829 PFAS only", C_S[0], C_WARN, 8.5, True)
box(36.5, 19, 27, 13, "Strategy 2: pretraining\nD-MPNN on ≈53k\nnon-PFAS (no PFAS seen)", C_S[1], C_DATA_E, 8.5, True)
box(70, 19, 27, 13, "Strategy 3: pretrain\n+ fine-tune\nretrain on PFAS pool", C_S[2], "#55a868", 8.5, True)

box(20, 6, 60, 9, "Evaluation: random vs. scaffold (chain-length withheld) splitting × 3 seeds\n"
    "metrics: AUROC / AUPRC   |   classical baseline: Morgan-FP random forest",
    "#e8e8e8", "#555555", 8.5, True)

box(2, 0.3, 96, 4.5, "Outputs:  generalization-gap quantification  ·  leakage-inflated comparison "
    "(apparent fine-tuning gain +0.36 = artifact)  ·  chemical-space-aware routing  ·  deployment "
    "as ADMET-AI v2 custom model (PFAS_Tox21_Panel)", "#fff3d9", "#b8860b", 7.5)

arrow(25, 68, 33, 62)      # PCBA large -> check
arrow(25, 54, 33, 58)      # PFAS panel -> check
arrow(53, 60.5, 63, 60.5)  # check -> decon
arrow(14, 49, 36, 42)      # PFAS panel -> pool
arrow(74, 55, 50, 32.2)    # decon -> s2
arrow(92, 55, 84, 32.2)    # decon -> s3
arrow(43, 37, 16, 32.2)    # pool -> s1
arrow(46, 37, 82, 32.2)    # pool -> s3
arrow(16.5, 19, 36, 15.2)  # s1 -> eval
arrow(50, 19, 50, 15.2)    # s2 -> eval
arrow(83.5, 19, 64, 15.2)  # s3 -> eval

ax.set_title("Figure 1", loc="left", fontsize=12, fontweight="bold", pad=8)
plt.tight_layout()
fig.savefig(BASE + r"\Fig1_workflow.png", dpi=300, bbox_inches="tight")
print("Fig1 saved")