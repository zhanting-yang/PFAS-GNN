# -*- coding: utf-8 -*-
"""Fig 3: 26 task × 4 model 热图（scaffold 测试集，种子7）"""
import subprocess, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*')

BASE = r"C:\Users\HUAWEI\Desktop\pfas"
CHEMPROP = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\Scripts\chemprop.exe"

test_df = pd.read_csv(BASE + r"\test.csv", index_col=0)
train_df = pd.read_csv(BASE + r"\train.csv", index_col=0)
task_cols = [c for c in test_df.columns if c.startswith("PCBA")]

CKPTS = {
    "GNN\nscratch":        BASE + r"\scratch_s7.pt",
    "GNN pretrained\n(no PFAS)": BASE + r"\PFAS_large_clean_random\model_0\best.pt",
    "GNN pretrained\n+ fine-tuned": BASE + r"\PFAS_small_scaffold\model_0\best.pt",
}

def gnn_predict(ckpt, tag):
    smi_file = BASE + f"\\fig3_{tag}_smiles.csv"
    out_file = BASE + f"\\fig3_{tag}_preds.csv"
    pd.DataFrame({"smiles": test_df.index}).to_csv(smi_file, index=False)
    subprocess.run(f'"{CHEMPROP}" predict --test-path {smi_file} '
                   f'--model-path {ckpt} --preds-path {out_file}', shell=True, check=True)
    return pd.read_csv(out_file, index_col=0)

def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)) if m else None

# ---- 三个 GNN 条件 ----
preds = {}
for tag, ckpt in CKPTS.items():
    if not os.path.exists(ckpt):
        raise SystemExit(f"模型文件不存在：{ckpt}\n请先完成第0步（抢救scratch并重跑微调）")
    preds[tag] = gnn_predict(ckpt, tag.split("\n")[0].split()[-1].lower())

# ---- RF 条件（同一 train/test 划分）----
X_tr = np.array([fp(s) for s in train_df.index])
X_te = np.array([fp(s) for s in test_df.index])
rf_pred = pd.DataFrame(index=test_df.index, columns=task_cols, dtype=float)
for tcol in task_cols:
    y = train_df[tcol].values
    m = ~pd.isna(y)
    yte = test_df[tcol].values
    mte = ~pd.isna(yte)
    if m.sum() < 20 or mte.sum() <= 10 or len(np.unique(yte[mte])) < 2:
        rf_pred.loc[mte, tcol] = np.nan
        continue
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_tr[m], y[m].astype(int))
    rf_pred.loc[mte, tcol] = clf.predict_proba(X_te[mte])[:, 1]
preds["Random\nforest"] = rf_pred.astype(float)

# ---- 计算逐任务 AUC ----
auc = pd.DataFrame(index=task_cols, columns=preds.keys(), dtype=float)
for tcol in task_cols:
    y = test_df[tcol].values
    for tag, p in preds.items():
        m = (~pd.isna(y)) & (~pd.isna(p[tcol].values))
        yy = y[m]
        if len(yy) > 10 and len(np.unique(yy)) == 2:
            auc.loc[tcol, tag] = roc_auc_score(yy, p.loc[m, tcol].astype(float))
        else:
            auc.loc[tcol, tag] = np.nan
auc.to_csv(BASE + r"\fig3_per_task_auc.csv")
print("逐任务AUC均值：\n", auc.mean().round(3))

# ---- 热图 ----
fig, ax = plt.subplots(figsize=(7.5, 9))
data = auc.values.astype(float)
im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(auc.columns))); ax.set_xticklabels(auc.columns, fontsize=9)
ax.set_yticks(range(len(auc.index)))
short_names = {"PCBA-883":"CYP2C9","PCBA-884":"CYP3A4","PCBA-891":"CYP2D6","PCBA-1030":"ALDH1A1",
               "PCBA-686970":"HT-1080","PCBA-686978":"DT40-A","PCBA-686979":"DT40-B","PCBA-938":"CNG"}
ax.set_yticklabels([f"{t} ({short_names.get(t,'')})" if t in short_names else t for t in auc.index], fontsize=7.5)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.55 < v < 0.85 else "white")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#999999")
        else:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.55 < v < 0.85 else "white")
fig.colorbar(im, ax=ax, shrink=0.5, label="AUROC")
fig.tight_layout()
fig.savefig(BASE + r"\Fig3_per_task_heatmap.png", dpi=300)
print("Fig3 saved")