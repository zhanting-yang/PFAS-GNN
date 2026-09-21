import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\HUAWEI\Desktop\pfas"
df = pd.read_csv(BASE + r"\web_demo_preds.csv", index_col=0)
names = ["PFOA", "PFOS", "Metoprolol"]

# 修正版 7 类映射（504339 = JMJD2A）
CATEGORY = {
    "Metabolism":           ["PCBA-883", "PCBA-884", "PCBA-891", "PCBA-1030"],
    "Gene regulation":      ["PCBA-2546", "PCBA-2551", "PCBA-504332", "PCBA-504444", "PCBA-504467", "PCBA-588855", "PCBA-651635"],
    "Cytotoxicity":         ["PCBA-686970", "PCBA-686978", "PCBA-686979"],
    "Membrane targets":     ["PCBA-1461", "PCBA-624417", "PCBA-938"],
    "Protein interaction":  ["PCBA-1468", "PCBA-1688", "PCBA-504339"],
    "Cell signaling":       ["PCBA-624288", "PCBA-720504"],
    "Pathogen/DNA stress":  ["PCBA-540276", "PCBA-720579", "PCBA-720580", "PCBA-624296"],
}

# 严格校验：任何一类缺列就明确报错
all_cols = [c for cols in CATEGORY.values() for c in cols]
missing = [c for c in all_cols if c not in df.columns]
if missing:
    raise SystemExit(f"CSV 缺少以下列，请核对：{missing}\n实际列名：{list(df.columns)}")

for i, mol in enumerate(df.index):
    name = names[i] if i < len(names) else f"Mol_{i+1}"
    cat_vals = {cat: df.loc[mol, cols].astype(float).mean() for cat, cols in CATEGORY.items()}
    labels = list(cat_vals.keys())
    values = np.concatenate([list(cat_vals.values()), [list(cat_vals.values())[0]]])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=1.8, color="crimson")
    ax.fill(angles, values, alpha=0.25, color="crimson")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"{name} — PFAS ADME-Tox Profile", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(BASE + f"\\radar_admet_{name}.png", dpi=200)
    plt.close()
    print(f"{name}: ", {k: round(v, 2) for k, v in cat_vals.items()})